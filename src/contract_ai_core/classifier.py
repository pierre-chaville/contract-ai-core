from __future__ import annotations

"""Clause classifier: assign clause categories to paragraphs for a given template.

Public API:
- ClauseClassifier.classify_paragraphs
- ClauseClassifier.classify_paragraphs_with_usage
"""

import logging
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from .schema import (
    ClassifiedParagraph,
    ContractTypeTemplate,
    DocumentClassification,
    Paragraph,
)
from .utilities import get_langchain_chat_model


@dataclass
class ClauseClassifierConfig:
    """Configuration for the clause classifier backend (LLM or rule-based)."""

    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = 10000
    # Retry policy for transient rate/limit errors
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0


class ClauseClassifier:
    """Classifies contract paragraphs into clause categories.

    Implement `classify_paragraphs` and `classify_document` to connect to your
    LLMs or rules. Keep deterministic behavior as much as possible.
    """

    def __init__(self, config: ClauseClassifierConfig | None = None) -> None:
        self.config = config or ClauseClassifierConfig()

    # ------------------------
    # Public API
    # ------------------------
    def classify_paragraphs(
        self,
        paragraphs: Sequence[Paragraph],
        template: ContractTypeTemplate,
        *,
        source_id: str | None = None,
    ) -> DocumentClassification:
        """Return a classification for each paragraph.

        Must map paragraphs to `clause_key`s present in `template.clauses` or
        leave them unclassified (None). Should also optionally produce
        `clause_to_paragraphs` for convenience.
        """
        if self.config.provider != "openai":
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        if not paragraphs:
            return DocumentClassification(paragraphs=[], clause_to_paragraphs={})

        classification, _usage = self.classify_paragraphs_with_usage(
            paragraphs, template, source_id=source_id
        )
        return classification

    def classify_paragraphs_with_usage(
        self,
        paragraphs: Sequence[Paragraph],
        template: ContractTypeTemplate,
        *,
        source_id: str | None = None,
    ) -> tuple[DocumentClassification, dict[str, int | None]]:
        clause_index_to_key, clauses_block = self._build_clauses_block(template)
        paragraphs_block = self._build_paragraphs_block(paragraphs)
        prompt = self._build_prompt(template, clauses_block, paragraphs_block)

        raw_output, usage = self._call_llm(prompt)
        parsed = self._parse_llm_output(raw_output)

        classified: list[ClassifiedParagraph] = []
        clause_to_paragraphs: dict[str, list[int]] = {}

        for para in paragraphs:
            line_index = para.index
            clause_id, confidence_pct = parsed.get(line_index, (0, None))

            if clause_id == 0:
                classified.append(
                    ClassifiedParagraph(
                        paragraph=para,
                        clause_key=None,
                        confidence=(confidence_pct / 100.0) if confidence_pct is not None else None,
                    )
                )
                continue

            clause_key = clause_index_to_key.get(clause_id)
            if clause_key is None:
                # Robustness: treat as unclassified if the LLM referenced an invalid index
                classified.append(
                    ClassifiedParagraph(
                        paragraph=para,
                        clause_key=None,
                        confidence=(confidence_pct / 100.0) if confidence_pct is not None else None,
                    )
                )
                continue

            classified.append(
                ClassifiedParagraph(
                    paragraph=para,
                    clause_key=clause_key,
                    confidence=(confidence_pct / 100.0) if confidence_pct is not None else None,
                )
            )
            clause_to_paragraphs.setdefault(clause_key, []).append(line_index)

        doc = DocumentClassification(
            paragraphs=classified, clause_to_paragraphs=clause_to_paragraphs
        )
        return doc, usage

    # ------------------------
    # Prompt construction
    # ------------------------
    def _build_clauses_block(self, template: ContractTypeTemplate) -> tuple[dict[int, str], str]:
        """Create a mapping from 1-based clause index to clause key and render the clause list.

        The rendered format is: `index: | title | description` with description optional.
        """
        index_to_key: dict[int, str] = {}
        lines: list[str] = []
        for i, clause in enumerate(template.clauses, start=1):
            index_to_key[i] = clause.key
            description = clause.description or ""
            lines.append(f"{i}: | {clause.title} | {description}")
        return index_to_key, "\n".join(lines)

    def _build_paragraphs_block(self, paragraphs: Sequence[Paragraph]) -> str:
        lines = [f"{p.index}: {p.text}" for p in paragraphs]
        return "\n".join(lines)

    def _build_prompt(
        self, template: ContractTypeTemplate, clauses_block: str, paragraphs_block: str
    ) -> str:
        instructions = (
            "You are an expert legal clause classifier.\n"
            "Given: (1) a list of clause categories and (2) a list of contract paragraphs,\n"
            f"The contract is a {template.description}.\n"
            "classify each paragraph to the single most likely clause category.\n\n"
            "Important rules:\n"
            "- Output exactly one line per paragraph provided.\n"
            "- Format for each line: '<line_number>: <clause_id> | <confidence>%'.\n"
            "- <line_number> is the number shown for the paragraph.\n"
            "- <clause_id> is the numeric id of the clause from the provided list; use 0 if none/other.\n"
            "- <confidence> is an integer from 0 to 100 representing your confidence.\n"
            "- Do not include explanations or extra text. Only the lines in the specified format.\n"
        )

        clauses_header = "CLAUSES (id: | title | description):\n" + clauses_block
        paragraphs_header = "PARAGRAPHS (line_number: text):\n" + paragraphs_block
        response_header = (
            "RESPONSE FORMAT (one per paragraph, no extra prose):\n"
            "<line_number>: <clause_id> | <confidence>%\n"
        )

        return "\n\n".join([instructions, clauses_header, paragraphs_header, response_header])

    # ------------------------
    # LLM backend
    # ------------------------
    def _call_llm(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        """Call an LLM via LangChain with deterministic configuration.

        Returns (content, usage) where usage has keys prompt_tokens, completion_tokens, total_tokens.
        """
        model_name = self.config.model or "gpt-4.1"
        temperature = float(self.config.temperature)
        max_tokens = self.config.max_tokens

        # Build the model once; reuse across retries
        llm = get_langchain_chat_model(
            self.config.provider,
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logging.getLogger(__name__).debug(
            "Calling LLM provider=%s model=%s temperature=%s",
            self.config.provider,
            model_name,
            temperature,
        )

        attempt = 1
        delay_seconds = max(0.0, float(self.config.initial_backoff_seconds))
        max_retries = max(1, int(self.config.max_retries))
        max_backoff = max(1.0, float(self.config.max_backoff_seconds))
        last_error: Exception | None = None

        while attempt <= max_retries:
            try:
                ai_msg = llm.invoke(
                    [
                        SystemMessage(content="You are a precise output-only model."),
                        HumanMessage(content=prompt),
                    ]
                )
                content = getattr(ai_msg, "content", "") or ""

                # Best-effort usage extraction across providers
                usage_dict: dict[str, int | None] = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                try:
                    usage_md = getattr(ai_msg, "usage_metadata", None)
                    if isinstance(usage_md, dict):
                        usage_dict["prompt_tokens"] = usage_md.get("input_tokens")  # type: ignore[arg-type]
                        usage_dict["completion_tokens"] = usage_md.get("output_tokens")  # type: ignore[arg-type]
                        usage_dict["total_tokens"] = usage_md.get("total_tokens")  # type: ignore[arg-type]
                    else:
                        resp_md = getattr(ai_msg, "response_metadata", None)
                        if isinstance(resp_md, dict):
                            token_usage = resp_md.get("token_usage") or resp_md.get("usage") or {}
                            if isinstance(token_usage, dict):
                                usage_dict["prompt_tokens"] = token_usage.get("prompt_tokens")  # type: ignore[arg-type]
                                usage_dict["completion_tokens"] = token_usage.get(
                                    "completion_tokens"
                                )  # type: ignore[arg-type]
                                usage_dict["total_tokens"] = token_usage.get("total_tokens")  # type: ignore[arg-type]
                except Exception:
                    pass

                return content.strip(), usage_dict
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                transient = any(
                    kw in msg
                    for kw in (
                        "rate limit",
                        "too many requests",
                        "overload",
                        "temporar",
                        "tpm",
                        "rpm",
                        "quota",
                        "retry",
                    )
                )
                if (attempt >= max_retries) or not transient:
                    raise
                jitter = random.uniform(0.0, delay_seconds * 0.25)
                time.sleep(delay_seconds + jitter)
                delay_seconds = min(delay_seconds * 2.0, max_backoff)
                attempt += 1

        if last_error is not None:
            raise last_error
        return "", {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    # ------------------------
    # Output parsing
    # ------------------------
    def _parse_llm_output(self, raw_output: str) -> dict[int, tuple[int, int | None]]:
        """Parse lines of the form 'line: clause_id | 95%'.

        Returns mapping: line_number -> (clause_id, confidence_percent or None)
        """
        result: dict[int, tuple[int, int | None]] = {}
        if not raw_output.strip():
            return result

        pattern = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*\|\s*([0-9]{1,3})%\s*$")
        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                # Try a bit more tolerant matching without confidence
                fallback = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", line)
                if fallback:
                    line_no = int(fallback.group(1))
                    clause_id = int(fallback.group(2))
                    result[line_no] = (clause_id, None)
                continue

            line_no = int(match.group(1))
            clause_id = int(match.group(2))
            confidence = int(match.group(3))
            # Clamp confidence between 0 and 100
            confidence = max(0, min(100, confidence))
            result[line_no] = (clause_id, confidence)

        return result
