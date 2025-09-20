from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .schema import ContractTypeTemplate, FilteringScope
from .utilities import get_langchain_chat_model, text_to_paragraphs


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
    text_to_paragraphs.
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

    class _NamedSpanOutput(BaseModel):
        name: str = Field(..., description="Scope name to match the input query list")
        start_line: int = Field(..., description="Start line index (inclusive)")
        end_line: int = Field(..., description="End line index (inclusive)")
        confidence: Optional[float] = Field(
            default=None, ge=0.0, le=1.0, description="Confidence in [0,1]"
        )
        explanation: Optional[str] = Field(
            default=None, description="Brief rationale for the chosen span"
        )

    class _BulkOutput(BaseModel):
        items: list["DocumentFilter._NamedSpanOutput"] = Field(
            default_factory=list, description="Results, one per requested scope"
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

        paragraphs = text_to_paragraphs(document_text)
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
        if start_line == -1 and end_line >= 0:
            start_line = 0
        if start_line < 0 or end_line < start_line:
            return (*default_on_fail, confidence, explanation)

        # Clamp to available indices
        max_idx = len(paragraphs) - 1
        start_line = max(0, min(start_line, max_idx))
        end_line = max(0, min(end_line, max_idx))

        return start_line, end_line, confidence, explanation

    def _locate_spans_bulk(
        self,
        *,
        document_text: str,
        named_queries: list[tuple[str, str]],
        default_on_fail: tuple[int, int] = (-1, -1),
    ) -> dict[str, tuple[int, int, Optional[float], Optional[str]]]:
        """Return mapping name -> (start_line, end_line, confidence, explanation) in one LLM call.

        named_queries: list of (name, query_description)
        """
        if self.config.provider not in ("openai", "azure", "anthropic"):
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        paragraphs = text_to_paragraphs(document_text)
        if not paragraphs:
            return {name: (*default_on_fail, None, "Empty document") for name, _ in named_queries}

        lines_block = "\n".join(f"{p.index}: {p.text}" for p in paragraphs)

        instruction = (
            "You are an expert legal analyst. Given the document paragraphs with line numbers, "
            "identify a contiguous span for each query in the QUERIES list."
        )
        guidance = (
            "For any query where no suitable span can be identified with confidence, return "
            f"start_line={default_on_fail[0]} and end_line={default_on_fail[1]} and a low confidence."
        )

        queries_block = "\n".join(
            f"- name: {name}\n  description: {desc}" for name, desc in named_queries
        )

        prompt = (
            instruction
            + "\n\nQUERIES (YAML-like list):\n"
            + queries_block
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

        structured_llm = llm.with_structured_output(DocumentFilter._BulkOutput)  # type: ignore[arg-type]
        try:
            out: DocumentFilter._BulkOutput = structured_llm.invoke(prompt)  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover
            logging.getLogger(__name__).error("Bulk span location failed: %r", e)
            return {name: (*default_on_fail, None, str(e)) for name, _ in named_queries}

        results: dict[str, tuple[int, int, Optional[float], Optional[str]]] = {}
        # Initialize defaults for all names
        for name, _ in named_queries:
            results[name] = (*default_on_fail, None, None)

        # Fill from model output
        for item in getattr(out, "items", []) or []:
            try:
                name = str(getattr(item, "name", "") or "").strip()
                if not name:
                    continue
                start_line = int(getattr(item, "start_line", -1) or -1)
                end_line = int(getattr(item, "end_line", -1) or -1)
                confidence: Optional[float] = getattr(item, "confidence", None)
                explanation: Optional[str] = getattr(item, "explanation", None)

                # Validate order
                if start_line == -1 and end_line >= 0:
                    start_line = 0
                if start_line < 0 or end_line < start_line:
                    results[name] = (*default_on_fail, confidence, explanation)
                    continue

                # Clamp to available indices
                max_idx = len(paragraphs) - 1
                start_line = max(0, min(start_line, max_idx))
                end_line = max(0, min(end_line, max_idx))

                results[name] = (start_line, end_line, confidence, explanation)
            except Exception:
                # Keep default for this name
                continue

        return results

    def locate_scopes(
        self,
        *,
        document_text: str,
        scopes: list[FilteringScope] | tuple[FilteringScope, ...],
        default_on_fail: tuple[int, int] = (-1, -1),
    ) -> dict[str, tuple[int, int, Optional[float], Optional[str]]]:
        """Return a mapping of scope.name -> (start_line, end_line, confidence, explanation).

        Each scope is evaluated independently using its description as the query.
        If a scope cannot be identified, the default_on_fail span is returned for it.
        """
        scopes_list = list(scopes)
        if len(scopes_list) > 1:
            # Single LLM call for all scopes
            named_queries = [(s.name, s.description) for s in scopes_list]
            return self._locate_spans_bulk(
                document_text=document_text,
                named_queries=named_queries,
                default_on_fail=default_on_fail,
            )
        # Fallback to single-call path
        results: dict[str, tuple[int, int, Optional[float], Optional[str]]] = {}
        for scope in scopes_list:
            start, end, conf, expl = self.locate_span(
                document_text=document_text,
                query=scope.description,
                default_on_fail=default_on_fail,
            )
            results[scope.name] = (start, end, conf, expl)
        return results

    def locate_template_scopes(
        self,
        *,
        document_text: str,
        template: ContractTypeTemplate,
        default_on_fail: tuple[int, int] = (-1, -1),
    ) -> dict[str, tuple[int, int, Optional[float], Optional[str]]]:
        """Convenience wrapper to evaluate all template.filtering_scopes.

        Returns mapping of scope name to (start_line, end_line, confidence, explanation).
        """
        return self.locate_scopes(
            document_text=document_text,
            scopes=list(template.filtering_scopes),
            default_on_fail=default_on_fail,
        )
