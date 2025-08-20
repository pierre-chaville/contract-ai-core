from __future__ import annotations

from dataclasses import dataclass
import os
import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import (
    ContractTypeTemplate,
    DocumentClassification,
    Paragraph,
    ClassifiedParagraph,
    split_text_into_paragraphs,
)


@dataclass
class ClauseClassifierConfig:
    """Configuration for the clause classifier backend (LLM or rule-based)."""

    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = 10000


class ClauseClassifier:
    """Classifies contract paragraphs into clause categories.

    Implement `classify_paragraphs` and `classify_document` to connect to your
    LLMs or rules. Keep deterministic behavior as much as possible.
    """

    def __init__(self, config: Optional[ClauseClassifierConfig] = None) -> None:
        self.config = config or ClauseClassifierConfig()

    # ------------------------
    # Public API
    # ------------------------
    def classify_paragraphs(
        self,
        paragraphs: Sequence[Paragraph],
        template: ContractTypeTemplate,
        *,
        source_id: Optional[str] = None,
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

        clause_index_to_key, clauses_block = self._build_clauses_block(template)
        paragraphs_block = self._build_paragraphs_block(paragraphs)
        prompt = self._build_prompt(clauses_block, paragraphs_block)
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt) 

        raw_output, usage = self._call_openai(prompt)
        if source_id is not None:
            self._write_tokens_usage(
                source_id=source_id,
                provider=self.config.provider,
                model=self.config.model or "gpt-4.1-mini",
                usage=usage,
                num_paragraphs=len(paragraphs),
            )
        parsed = self._parse_llm_output(raw_output)

        classified: List[ClassifiedParagraph] = []
        clause_to_paragraphs: Dict[str, List[int]] = {}

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

        return DocumentClassification(paragraphs=classified, clause_to_paragraphs=clause_to_paragraphs)

    def classify_document(
        self,
        text: str,
        template: ContractTypeTemplate,
        *,
        source_id: Optional[str] = None,
    ) -> DocumentClassification:
        """Split the document into paragraphs and classify them."""
        paragraphs = split_text_into_paragraphs(text)
        return self.classify_paragraphs(paragraphs=paragraphs, template=template, source_id=source_id)

    # ------------------------
    # Prompt construction
    # ------------------------
    def _build_clauses_block(self, template: ContractTypeTemplate) -> Tuple[Dict[int, str], str]:
        """Create a mapping from 1-based clause index to clause key and render the clause list.

        The rendered format is: `index: | title | description` with description optional.
        """
        index_to_key: Dict[int, str] = {}
        lines: List[str] = []
        for i, clause in enumerate(template.clauses, start=1):
            index_to_key[i] = clause.key
            description = clause.description or ""
            lines.append(f"{i}: | {clause.title} | {description}")
        return index_to_key, "\n".join(lines)

    def _build_paragraphs_block(self, paragraphs: Sequence[Paragraph]) -> str:
        lines = [f"{p.index}: {p.text}" for p in paragraphs]
        return "\n".join(lines)

    def _build_prompt(self, clauses_block: str, paragraphs_block: str) -> str:
        instructions = (
            "You are an expert legal clause classifier.\n"
            "Given: (1) a list of clause categories and (2) a list of contract paragraphs,\n"
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
    def _call_openai(self, prompt: str) -> Tuple[str, Dict[str, Optional[int]]]:
        """Call OpenAI Chat Completions with a deterministic configuration.

        Returns (content, usage) where usage has keys prompt_tokens, completion_tokens, total_tokens.
        """
        # Lazy import to avoid hard dependency at package import time
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            load_dotenv = None  # type: ignore

        if load_dotenv is not None:
            try:
                load_dotenv()
            except Exception:  # pragma: no cover
                pass

        try:
            # Using OpenAI Python SDK v1
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - helpful error if not installed
            raise RuntimeError(
                "The 'openai' package is required to use ClauseClassifier with provider 'openai'. "
                "Install it with: pip install openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # The SDK will also read env var, but we validate early for clearer errors
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(api_key=api_key)

        model_name = self.config.model or "gpt-4.1-mini"
        print('model_name', model_name)
        temperature = float(self.config.temperature)
        max_tokens = self.config.max_tokens

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
        print('content', content)
        usage = getattr(response, "usage", None)
        usage_dict: Dict[str, Optional[int]] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        }
        return content.strip(), usage_dict

    # ------------------------
    # Output parsing
    # ------------------------
    def _parse_llm_output(self, raw_output: str) -> Dict[int, Tuple[int, Optional[int]]]:
        """Parse lines of the form 'line: clause_id | 95%'.

        Returns mapping: line_number -> (clause_id, confidence_percent or None)
        """
        result: Dict[int, Tuple[int, Optional[int]]] = {}
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

    # ------------------------
    # Token usage logging
    # ------------------------
    def _write_tokens_usage(
        self,
        *,
        source_id: str,
        provider: str,
        model: str,
        usage: Dict[str, Optional[int]] | None,
        num_paragraphs: int,
    ) -> None:
        """Append a JSONL record with token usage metadata into dataset/output/tokens.jsonl.

        On any error, silently no-op to avoid disrupting classification.
        """
        try:
            repo_root = Path(__file__).resolve().parents[2]
            out_dir = repo_root / "dataset" / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "tokens.jsonl"

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_id": source_id,
                "provider": provider,
                "model": model,
                "num_paragraphs": num_paragraphs,
                "usage": usage or {},
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return


