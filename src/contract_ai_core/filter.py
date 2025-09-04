from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .utilities import get_langchain_chat_model, split_text_into_paragraphs


@dataclass
class DocumentFilterConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class DocumentFilter:
    """Locate a contiguous span within a document using an LLM.

    Given a full document string and a short natural-language description of the
    section to find, returns the start and end line indices (inclusive) of the
    best-matching span. Lines are the paragraph indices produced by
    split_text_into_paragraphs.
    """

    def __init__(self, config: DocumentFilterConfig | None = None) -> None:
        self.config = config or DocumentFilterConfig()

    class _SpanOutput(BaseModel):
        start_line: int = Field(..., description="Start line index (inclusive)")
        end_line: int = Field(..., description="End line index (inclusive)")
        confidence: Optional[float] = Field(
            default=None, ge=0.0, le=1.0, description="Confidence in [0,1]"
        )
        explanation: Optional[str] = Field(
            default=None, description="Brief rationale for the chosen span"
        )

    def locate_span(
        self,
        *,
        document_text: str,
        query: str,
        default_on_fail: tuple[int, int] = (-1, -1),
    ) -> tuple[int, int, Optional[float], Optional[str]]:
        """Return (start_line, end_line, confidence, explanation).

        If the model cannot confidently identify a span, returns default_on_fail
        and a low confidence.
        """
        if self.config.provider not in ("openai", "azure", "anthropic"):
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        paragraphs = split_text_into_paragraphs(document_text)
        if not paragraphs:
            return (*default_on_fail, None, "Empty document")

        lines_block = "\n".join(f"{p.index}: {p.text}" for p in paragraphs)

        instruction = (
            "You are an expert legal analyst. Given the document paragraphs with line numbers, "
            "identify the single contiguous span that best matches the user's query."
        )
        guidance = (
            "If no suitable span can be identified with confidence, return start_line=-1 and end_line=-1 "
            "and a low confidence."
        )

        prompt = (
            instruction
            + "\n\nQUERY:\n"
            + query
            + "\n\nPARAGRAPHS (line_number: text):\n"
            + lines_block
            + "\n\n"
            + guidance
        )

        llm = get_langchain_chat_model(
            self.config.provider,
            self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            max_tokens=self.config.max_tokens,
        )

        structured_llm = llm.with_structured_output(DocumentFilter._SpanOutput)  # type: ignore[arg-type]
        try:
            out: DocumentFilter._SpanOutput = structured_llm.invoke(prompt)  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover
            logging.getLogger(__name__).error("Span location failed: %r", e)
            return (*default_on_fail, None, str(e))

        start_line = int(getattr(out, "start_line", -1) or -1)
        end_line = int(getattr(out, "end_line", -1) or -1)
        confidence: Optional[float] = getattr(out, "confidence", None)
        explanation: Optional[str] = getattr(out, "explanation", None)

        # Validate order
        if start_line < 0 or end_line < start_line:
            return (*default_on_fail, confidence, explanation)

        # Clamp to available indices
        max_idx = len(paragraphs) - 1
        start_line = max(0, min(start_line, max_idx))
        end_line = max(0, min(end_line, max_idx))

        return start_line, end_line, confidence, explanation
