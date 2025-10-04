from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from .schema import (
    ContractMetadata,
    ContractTypeTemplate,
    DocumentAnalysis,
    DocumentClassification,
    ExtractedDatapoint,
    Paragraph,
    ReviewedGuideline,
)
from .utilities import get_langchain_chat_model


@dataclass
class AgentConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = 1200
    # Maximum number of paragraphs to include when falling back to full text
    max_full_text_paragraphs: int = 40


class Agent:
    """Conversational agent to answer questions about a contract.

    The agent selects relevant context (raw text, classified clauses, extracted datapoints,
    reviewed guidelines), then queries an LLM with the question and the context.
    Conversation is stateful; use reset() to clear the dialogue history.
    """

    def __init__(
        self,
        *,
        contract_type: ContractTypeTemplate | None,
        analysis: DocumentAnalysis,
        config: AgentConfig | None = None,
    ) -> None:
        self.contract_type: ContractTypeTemplate | None = contract_type
        self.analysis: DocumentAnalysis = analysis
        self.config: AgentConfig = config or AgentConfig()
        self._history: list[tuple[str, str]] = []  # (user_question, assistant_answer)

        self._llm = get_langchain_chat_model(
            provider=self.config.provider,
            model_name=self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            max_tokens=self.config.max_tokens,
        )

        # Precompute helpful lookups
        self._clause_key_to_text: dict[str, str] = self._build_clause_text_index(
            analysis.classified_clauses
        )
        self._dp_key_to_item: dict[str, ExtractedDatapoint] = {
            dp.key: dp
            for dp in (
                analysis.extracted_datapoints.datapoints if analysis.extracted_datapoints else []
            )
        }
        self._guidelines: list[ReviewedGuideline] = list(analysis.reviewed_guidelines or [])

        self._clause_key_to_title: dict[str, str] = {}
        self._clause_key_to_description: dict[str, str] = {}
        self._dp_key_to_title: dict[str, str] = {}
        self._guideline_key_to_text: dict[str, str] = {}
        if self.contract_type is not None:
            try:
                self._clause_key_to_title = {
                    c.key: (c.title or c.key) for c in (self.contract_type.clauses or [])
                }
            except Exception:
                self._clause_key_to_title = {}
            try:
                self._clause_key_to_description = {
                    c.key: (c.description or "") for c in (self.contract_type.clauses or [])
                }
            except Exception:
                self._clause_key_to_description = {}
            try:
                self._dp_key_to_title = {
                    d.key: (d.title or d.key) for d in (self.contract_type.datapoints or [])
                }
            except Exception:
                self._dp_key_to_title = {}
            try:
                self._guideline_key_to_text = {
                    g.key: (g.guideline) for g in (self.contract_type.guidelines or [])
                }
            except Exception:
                self._guideline_key_to_text = {}

    def reset(self) -> None:
        self._history.clear()

    def ask(self, question: str) -> str:
        categories = self._choose_context_categories(question)
        context_sections = self._collect_context(question, categories)
        prompt = self._build_prompt(question, context_sections)
        print("--------------------------------")
        print("prompt", prompt)
        result = self._llm.invoke(prompt)
        try:
            answer = getattr(result, "content", str(result))
        except Exception:
            answer = str(result)
        self._history.append((question, answer))
        return answer

    # -------------------- Internal helpers --------------------

    def _build_clause_text_index(
        self, classification: Optional[DocumentClassification]
    ) -> dict[str, str]:
        if not classification or not classification.paragraphs:
            return {}
        grouped: dict[str, list[str]] = {}
        for item in classification.paragraphs:
            ck = (item.clause_key or "").strip()
            if not ck:
                continue
            txt = (item.paragraph.text or "").strip()
            if not txt:
                continue
            grouped.setdefault(ck, []).append(txt)
        return {k: "\n\n".join(v) for k, v in grouped.items()}

    def _choose_context_categories(self, question: str) -> list[str]:
        """Use an LLM with structured output to choose relevant context categories.

        Returns a list subset of {"text", "clauses", "datapoints", "guidelines"}.
        Falls back to a simple heuristic if the structured call fails.
        """

        class CategorySelection(BaseModel):
            text: bool = Field(
                description="Look at raw document text when the query is broad or unspecific."
            )
            clauses: bool = Field(
                description="Look at classified clauses if the question mentions a clause or policy section."
            )
            datapoints: bool = Field(
                description="Use structured datapoint values when asking about entities, dates, enumerations, or metrics."
            )
            guidelines: bool = Field(
                description="Use reviewed guidelines when the question is about compliance, policies, or requirements."
            )
            rationale: Optional[str] = Field(
                default=None, description="Short explanation for the selection."
            )

        try:
            available = {
                "text": bool(
                    self.analysis.classified_clauses and self.analysis.classified_clauses.paragraphs
                ),
                "clauses": bool(self._clause_key_to_text),
                "datapoints": bool(self._dp_key_to_item),
                "guidelines": bool(self._guidelines),
            }
            avail_lines = [f"- {k}: {'yes' if v else 'no'}" for k, v in available.items()]
            prompt = (
                "You are selecting which context sources to consult to answer a question about a contract.\n"
                "Available sources (yes/no):\n" + "\n".join(avail_lines) + "\n\n"
                "Choose True/False for each: text, clauses, datapoints, guidelines.\n"
                "Prefer datapoints for direct values (dates, parties, enumerations), clauses for policy/section questions,\n"
                "guidelines for compliance checks, and text for broad/unspecific queries.\n\n"
                f"Question: {question.strip()}\n"
            )
            structured = self._llm.with_structured_output(
                CategorySelection, temperature=0.0, max_tokens=300
            )  # type: ignore[arg-type]
            sel: CategorySelection = structured.invoke(prompt)  # type: ignore[assignment]
            chosen = []
            if sel.text:
                chosen.append("text")
            if sel.clauses:
                chosen.append("clauses")
            if sel.datapoints:
                chosen.append("datapoints")
            if sel.guidelines:
                chosen.append("guidelines")
            if not chosen:
                chosen = ["text", "datapoints"]
            print("chosen", chosen)
            return chosen
        except Exception:
            # Heuristic fallback
            q = (question or "").lower()
            categories: list[str] = []
            mentions_guideline = any(
                x in q for x in ["guideline", "policy", "comply", "compliance"]
            )
            mentions_datapoint = any(
                x in q for x in ["datapoint", "value", "what is", "who is", "when is"]
            )
            clause_hit = any(
                (ck.lower() in q) or (title and title.lower() in q)
                for ck, title in self._clause_key_to_title.items()
            )
            if clause_hit:
                categories.append("clauses")
            elif not mentions_datapoint and not mentions_guideline:
                categories.append("text")
            if mentions_datapoint or self._dp_key_to_item:
                categories.append("datapoints")
            if mentions_guideline or self._guidelines:
                categories.append("guidelines")
            if not categories:
                categories = ["text", "datapoints", "guidelines"]
            return categories

    def _collect_context(self, question: str, categories: Iterable[str]) -> dict[str, str]:
        sections: dict[str, str] = {}

        if "text" in categories:
            full_text = self._build_full_text()
            if full_text:
                sections["FULL_TEXT"] = full_text

        if "clauses" in categories:
            clause_block = self._build_relevant_clauses_block(question)
            if clause_block:
                sections["CLAUSES"] = clause_block

        if "datapoints" in categories:
            dp_block = self._build_datapoints_block(question)
            if dp_block:
                sections["DATAPOINTS"] = dp_block

        if "guidelines" in categories:
            gl_block = self._build_guidelines_block(question)
            if gl_block:
                sections["GUIDELINES"] = gl_block

        meta = self._format_metadata(self.analysis.metadata)
        if meta:
            sections = {"METADATA": meta, **sections}
        return sections

    def _build_full_text(self) -> str:
        classification = self.analysis.classified_clauses
        if not classification or not classification.paragraphs:
            return ""
        paras: list[Paragraph] = [cp.paragraph for cp in classification.paragraphs]
        if not paras:
            return ""
        max_n = max(1, int(self.config.max_full_text_paragraphs))
        selected = paras[:max_n]
        return "\n\n".join(p.text for p in selected if p.text)

    def _build_relevant_clauses_block(self, question: str) -> str:
        clause_items: list[tuple[str, str, str, str]] = []  # (key, title, description, text)

        # Include all clauses
        for ck, text in self._clause_key_to_text.items():
            title = self._clause_key_to_title.get(ck, ck)
            description = self._clause_key_to_description.get(ck, "")
            clause_items.append((ck, title, description, text))

        if not clause_items:
            return ""

        lines: list[str] = ["List of all clauses:"]
        for _key, title, description, text in clause_items:
            lines.append(f"\nClause: {title}")
            if description:
                lines.append(f"Description: {description}")
            lines.append(f"Text:\n{text}")
        return "\n".join(lines)

    def _build_datapoints_block(self, question: str) -> str:
        # Include all datapoints
        items = list(self._dp_key_to_item.values())

        if not items:
            return ""
        lines: list[str] = ["Return values based on extracted datapoints when possible."]
        for dp in items:
            val = dp.value
            conf = (
                f"{int(round((dp.confidence or 0.0) * 100))}%" if dp.confidence is not None else ""
            )
            expl = dp.explanation or ""
            title = self._dp_key_to_title.get(dp.key, dp.key)
            lines.append(f"- {title}: {val} ({conf}) {expl}")
        return "\n".join(lines)

    def _build_guidelines_block(self, question: str) -> str:
        # Include all guidelines
        items = self._guidelines

        if not items:
            return ""

        lines: list[str] = ["Guideline evaluation results:"]
        for g in items:
            conf = f"{int(round((g.confidence or 0.0) * 100))}%" if g.confidence is not None else ""
            g_text = self._guideline_key_to_text.get(g.key, g.key)
            matched = "matched" if g.guideline_matched else "not matched"
            expl = g.explanation or ""
            lines.append(f"- {g.key}: {g_text} → {matched} ({conf}) {expl}")
        return "\n".join(lines)

    def _format_metadata(self, meta: Optional[ContractMetadata]) -> str:
        if not meta:
            return ""
        parts: list[str] = []
        if meta.title:
            parts.append(f"title={meta.title}")
        if meta.contract_type:
            parts.append(f"type={meta.contract_type}")
        if meta.execution_date:
            parts.append(f"execution_date={meta.execution_date}")
        if meta.effective_date:
            parts.append(f"effective_date={meta.effective_date}")
        if meta.counterparty_name:
            parts.append(f"counterparty={meta.counterparty_name}")
        return ", ".join(parts)

    def _build_prompt(self, question: str, sections: dict[str, str]) -> str:
        history_lines: list[str] = []
        for user_q, assistant_a in self._history[-6:]:
            history_lines.append(f"User: {user_q}")
            history_lines.append(f"Assistant: {assistant_a}")

        ordered = [
            (name, sections[name])
            for name in ["METADATA", "CLAUSES", "DATAPOINTS", "GUIDELINES", "FULL_TEXT"]
            if name in sections
        ]
        context_text = "\n\n".join([f"{name}:\n{val}" for name, val in ordered if val.strip()])

        sys_preamble = (
            "You are a helpful contract analysis assistant. Answer the user's question based only on the provided context. "
            "If the answer cannot be found, say so succinctly. Be precise and cite which context section(s) you used when helpful."
        )

        parts = [sys_preamble]
        if history_lines:
            parts.append("Conversation so far:\n" + "\n".join(history_lines))
        if context_text:
            parts.append("Context:\n" + context_text)
        parts.append("Question:\n" + question.strip())
        parts.append("Answer:")
        return "\n\n".join(parts)
