from __future__ import annotations

"""Clause classifier: assign clause categories to paragraphs for a given template.

Public API:
- ClauseClassifier.classify_paragraphs
- ClauseClassifier.classify_paragraphs_with_usage
"""

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI

from .schema import (
    ClassifiedParagraph,
    ContractTypeTemplate,
    DocumentClassification,
    Paragraph,
)


@dataclass
class ClauseClassifierConfig:
    """Configuration for the clause classifier backend (LLM or rule-based)."""

    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = 10000


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

        raw_output, usage = self._call_openai(prompt)
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
    def _call_openai(self, prompt: str) -> tuple[str, dict[str, int | None]]:
        """Call OpenAI Chat Completions with a deterministic configuration.

        Returns (content, usage) where usage has keys prompt_tokens, completion_tokens, total_tokens.
        """
        # Lazy import to avoid hard dependency at package import time
        try:
            from dotenv import load_dotenv
        except Exception:  # pragma: no cover - optional dependency
            load_dotenv = None  # type: ignore

        if load_dotenv is not None:
            try:
                load_dotenv()
            except Exception:  # pragma: no cover
                pass

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # The SDK will also read env var, but we validate early for clearer errors
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(api_key=api_key)

        model_name = self.config.model or "gpt-4.1"
        temperature = float(self.config.temperature)
        max_tokens = self.config.max_tokens

        logging.getLogger(__name__).debug(
            "Calling OpenAI model=%s temperature=%s", model_name, temperature
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise output-only model."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        logging.getLogger(__name__).debug(
            "OpenAI usage prompt=%s completion=%s total=%s",
            getattr(usage, "prompt_tokens", None) if usage is not None else None,
            getattr(usage, "completion_tokens", None) if usage is not None else None,
            getattr(usage, "total_tokens", None) if usage is not None else None,
        )
        usage_dict: dict[str, int | None] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            ),
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        }
        return content.strip(), usage_dict

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
