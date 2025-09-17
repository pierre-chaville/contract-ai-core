from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .utilities import get_langchain_chat_model


class FieldAnswer(BaseModel):
    """One field to fill in the docx template with provenance.

    To satisfy strict JSON schema requirements for structured output, scalar values should be
    placed in 'value'. For non-scalar (arrays/objects), serialize as JSON into 'value_json'.
    """

    key: str = Field(..., description="Template variable/field name (e.g., 'party_a_name')")
    value: str | int | float | bool | None = Field(
        default=None,
        description=(
            "Scalar value for the field (string, number, boolean, or null). For arrays/objects, leave "
            "this as null and use 'value_json' to provide a JSON string."
        ),
    )
    value_json: str | None = Field(
        default=None,
        description=(
            "If the value is an array or object (e.g., list of items for a for-loop), serialize it as a JSON string "
            "and set it here. For scalars, leave this null and use 'value'."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence in [0,1] for the chosen value.",
    )
    explanation: str | None = Field(
        default=None, description="Short rationale or source cue for the value."
    )


class AuthoringOutput(BaseModel):
    """LLM-structured output: fields to feed into docxtpl rendering."""

    fields: list[FieldAnswer] = Field(
        default_factory=list,
        description="List of fields with value, confidence, and explanation.",
    )


@dataclass
class AuthoringConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = 2000
    # Max characters of template text included in the prompt (to keep context concise)
    max_template_chars: int = 8000


class AuthoringDocument:
    """Generate a filled variable map for a docxtpl-based contract template.

    Inputs:
    - template_paragraphs: list of text paragraphs containing docxtpl/Jinja placeholders,
      e.g., "This Agreement is between {{ party_a_name }} and {{ party_b_name }}" or
      control structures like "{% if has_csa %}...{% endif %}" or "{% for fee in fees %} ... {% endfor %}".
    - context_text: free-text instruction describing how to fill placeholders and any business
      rules (conditions/loops, enumerations, defaulting rules, etc.).

    Output:
    - AuthoringOutput with fields: [{ key, value, confidence, explanation }]
      suitable for passing directly to docxtpl rendering (e.g., tpl.render(mapping)).
    """

    def __init__(
        self,
        *,
        template_paragraphs: list[str],
        context_text: str,
        config: AuthoringConfig | None = None,
    ) -> None:
        self.template_paragraphs = template_paragraphs or []
        self.context_text = context_text or ""
        self.config = config or AuthoringConfig()

        self._llm = get_langchain_chat_model(
            provider=self.config.provider,
            model_name=self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            max_tokens=self.config.max_tokens,
        )

    def infer_placeholders(self) -> dict[str, set[str]]:
        """Best-effort parse of placeholders and loop/conditional signals from docxtpl-like text.

        Returns a dict with keys:
        - variables: set of variable names found in {{ variable }}
        - loops: set of collection names found in {% for x in collection %}
        - conditions: set of variable/flag names found in {% if var %}
        """
        text = "\n".join(self.template_paragraphs)
        variables: set[str] = set(re.findall(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}", text))

        loops: set[str] = set()
        for m in re.finditer(r"\{\%\s*for\s+\w+\s+in\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\%\}", text):
            loops.add(m.group(1))

        conditions: set[str] = set()
        for m in re.finditer(r"\{\%\s*if\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\%\}", text):
            conditions.add(m.group(1))

        return {"variables": variables, "loops": loops, "conditions": conditions}

    def generate_fields(self) -> AuthoringOutput:
        """Call the LLM with a structured-output spec to fill fields for the template."""
        # Build concise template snippet
        template_text = "\n".join(self.template_paragraphs)
        if len(template_text) > self.config.max_template_chars:
            template_text = template_text[: self.config.max_template_chars] + "\n… (truncated)"

        ph = self.infer_placeholders()
        variables = sorted(ph.get("variables", set()))
        loops = sorted(ph.get("loops", set()))
        conditions = sorted(ph.get("conditions", set()))

        # Author instructions for the LLM
        guideline = (
            "You are drafting a contract by filling a docxtpl (Jinja) template.\n"
            "Return only structured data for variables the template expects.\n"
            "- For scalar placeholders (e.g., {{ party_name }}), return the atomic value in 'value'.\n"
            "- For arrays/objects (e.g., {% for fee in fees %}), serialize the data as a JSON string in 'value_json'.\n"
            "- For conditions (e.g., {% if has_csa %}), return booleans in 'value'.\n"
            "- Use concise, formal values appropriate for legal contracts.\n"
            "For each field, include: key, value (scalar) or value_json (for arrays/objects), confidence in [0,1], and a short explanation.\n"
        )

        # Summarize detected placeholders for clarity
        placeholder_summary_lines: list[str] = []
        if variables:
            placeholder_summary_lines.append("Variables: " + ", ".join(variables))
        if loops:
            placeholder_summary_lines.append("Loops (collections): " + ", ".join(loops))
        if conditions:
            placeholder_summary_lines.append("Conditions (flags): " + ", ".join(conditions))
        placeholder_summary = (
            "\n".join(placeholder_summary_lines) if placeholder_summary_lines else "(none detected)"
        )
        print("placeholder_summary", placeholder_summary)
        prompt = (
            guideline
            + "\nTEMPLATE (excerpt):\n"
            + template_text
            + "\n\nPLACEHOLDERS (parsed):\n"
            + placeholder_summary
            + "\n\nCONTEXT INSTRUCTIONS:\n"
            + (self.context_text or "(none)")
        )
        print("prompt", prompt)

        try:
            structured = self._llm.with_structured_output(  # type: ignore[arg-type]
                AuthoringOutput,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 2000,
            )
            out: AuthoringOutput = structured.invoke(prompt)  # type: ignore[assignment]
            print("out", out)
            return out
        except Exception as e:
            # Return empty bu   t valid structure on failure
            print("error", e)
            return AuthoringOutput(fields=[])
