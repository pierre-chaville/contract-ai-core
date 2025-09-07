from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Optional, Sequence, cast

from pydantic import BaseModel, Field, create_model

from .schema import LookupValue, Paragraph
from .utilities import get_langchain_chat_model


@dataclass
class ContractOrganizerConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    concurrency: int = 4
    max_retries: int = 5
    # Limit the amount of text sent by keeping only the first paragraphs whose
    # total word count stays under this threshold
    max_words: int = 500
    # Optional lookup lists used to constrain outputs for some fields
    lookup_contract_types: Optional[Sequence[LookupValue]] = None
    lookup_version_types: Optional[Sequence[LookupValue]] = None
    lookup_statuses: Optional[Sequence[LookupValue]] = None


class FieldResultString(BaseModel):
    value: Optional[str] = Field(default=None, description="Extracted value or null if unknown")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score in [0,1]"
    )
    explanation: Optional[str] = Field(
        default=None, description="Short rationale referencing supporting text"
    )


class OrganizedDocumentMetadata(BaseModel):
    filename: str
    contract_type: FieldResultString
    contract_date: FieldResultString
    amendment_date: FieldResultString
    amendment_number: FieldResultString
    version_type: FieldResultString
    status: FieldResultString


class ContractOrganizer:
    """Extract high-level contract metadata for multiple documents using an LLM."""

    def __init__(self, config: ContractOrganizerConfig | None = None) -> None:
        self.config = config or ContractOrganizerConfig()

    def organize(
        self,
        documents: Sequence[tuple[str, Sequence[Paragraph]]],
    ) -> list[OrganizedDocumentMetadata]:
        """Analyze a list of (filename, paragraphs) and return metadata for each.

        Extract fields with confidence and explanation:
        - contract_type
        - contract_date in the format YYYY-MM-DD
        - amendment_date (only if it is an amendment, else null) in the format YYYY-MM-DD
        - version_type (initial contract, amended and restated, amendment)
        - status (draft, executed, signed)
        """
        jobs: list[dict[str, object]] = []

        # Build a single OutputModel for all jobs
        FieldResultStringModel: type[BaseModel] = create_model(
            "FieldResultStringModel",
            value=(Optional[str], Field(default=None, description="Extracted value or null")),
            confidence=(
                Optional[float],
                Field(default=None, ge=0.0, le=1.0, description="Confidence in [0,1]"),
            ),
            explanation=(
                Optional[str],
                Field(default=None, description="Short rationale referencing supporting text"),
            ),
        )

        OutputModel: type[BaseModel] = create_model(
            "OrganizeOutput",
            contract_type=(FieldResultStringModel, Field(..., description="Type of contract")),
            contract_date=(FieldResultStringModel, Field(..., description="Primary contract date")),
            amendment_date=(
                FieldResultStringModel,
                Field(..., description="Amendment date if applicable, else null"),
            ),
            amendment_number=(
                FieldResultStringModel,
                Field(..., description="Amendment number if applicable, else null"),
            ),
            version_type=(
                FieldResultStringModel,
                Field(
                    ...,
                    description=(
                        "Version type: initial contract, amended and restated, or amendment"
                    ),
                ),
            ),
            status=(
                FieldResultStringModel,
                Field(..., description="Status: draft, executed, or signed"),
            ),
        )

        for filename, paragraphs in documents:
            # Keep only the beginning paragraphs up to max_words
            try:
                max_words = int(self.config.max_words)
            except Exception:
                max_words = 500
            selected: list[Paragraph] = []
            word_count = 0
            for p in paragraphs:
                words_in_p = len((p.text or "").split())
                if selected and (word_count + words_in_p) > max_words:
                    break
                if not selected and words_in_p > max_words:
                    # Ensure at least one paragraph is included
                    selected.append(p)
                    break
                selected.append(p)
                word_count += words_in_p

            text = "\n\n".join(p.text for p in selected)

            # Build ALLOWED VALUES section from config lookups
            def _build_allowed_section() -> str:
                lines: list[str] = []
                if self.config.lookup_contract_types:
                    lines.append("ALLOWED CONTRACT_TYPE codes (return the code exactly):")
                    lines.append("code | label | description")
                    for lv in self.config.lookup_contract_types:
                        lines.append(f"{lv.key} | {lv.label} | {lv.description}")
                    lines.append("")
                if self.config.lookup_version_types:
                    lines.append("ALLOWED VERSION_TYPE codes (return the code exactly):")
                    lines.append("code | label | description")
                    for lv in self.config.lookup_version_types:
                        lines.append(f"{lv.key} | {lv.label} | {lv.description}")
                    lines.append("")
                if self.config.lookup_statuses:
                    lines.append("ALLOWED STATUS codes (return the code exactly):")
                    lines.append("code | label | description")
                    for lv in self.config.lookup_statuses:
                        lines.append(f"{lv.key} | {lv.label} | {lv.description}")
                    lines.append("")
                return "\n".join(lines)

            allowed_section = _build_allowed_section()

            instruction = (
                "You are an expert legal analyst. Extract high-level contract metadata from the text.\n"
                "Return STRICTLY a JSON object matching the provided schema.\n"
                "For each field, include value, confidence in [0,1], and a short explanation.\n"
                "- contract_type: the category (e.g., NDA, employment, lease).\n"
                "- contract_date: the main date associated to the contract (usually the effective or execution date) in the format YYYY-MM-DD. When the document is an amendment, the contract date is the date of the initial contract (or the last amended and restated contract)\n"
                "- amendment_date: only if the document is an amendment, else null in the format YYYY-MM-DD.\n"
                "- amendment_number: only if the document is an amendment and if mentioned in the document, else null.\n"
                "- version_type: one of [initial contract, amended and restated, amendment].\n"
                "- status: one of [draft, executed, signed].\n"
                + (
                    "\nUse the ALLOWED codes provided below, you MUST choose one of those codes and return it exactly as the 'value' or 'role' for the parties.\n"
                    "Do NOT invent new values.\n\n" + allowed_section
                    if allowed_section
                    else ""
                )
            )

            prompt = instruction + f"FILENAME: {filename}\n\n" + "TEXT:\n" + text
            # print('--------------------------------')
            # print('prompt', prompt[:1000])
            jobs.append(
                {
                    "OutputModel": OutputModel,
                    "prompt": prompt,
                    "filename": filename,
                }
            )

        results = asyncio.run(
            self._run_document_organize(
                jobs=jobs,
                model_name=self.config.model or "gpt-4.1-mini",
                temperature=float(self.config.temperature),
                concurrency=int(self.config.concurrency),
            )
        )

        organized: list[OrganizedDocumentMetadata] = []
        for res in results:
            if not isinstance(res, dict):
                continue
            data_obj = res.get("data")
            data: dict[str, dict | None] = data_obj if isinstance(data_obj, dict) else {}
            filename = cast(str, res.get("filename") or "")

            # Extract and normalize per-field
            def _normalize_str_field(
                field_key: str, source: dict | None = data
            ) -> FieldResultString:
                item = (source or {}).get(field_key) or {}
                val = item.get("value")
                conf = item.get("confidence")
                expl = item.get("explanation")
                if isinstance(conf, (int, float)):
                    conf_f = max(0.0, min(1.0, float(conf)))
                else:
                    conf_f = None
                return FieldResultString(
                    value=str(val) if isinstance(val, str) and val.strip() else None,
                    confidence=conf_f,
                    explanation=expl if isinstance(expl, str) and expl.strip() else None,
                )

            organized.append(
                OrganizedDocumentMetadata(
                    filename=filename,
                    contract_type=_normalize_str_field("contract_type"),
                    contract_date=_normalize_str_field("contract_date"),
                    amendment_date=_normalize_str_field("amendment_date"),
                    amendment_number=_normalize_str_field("amendment_number"),
                    version_type=_normalize_str_field("version_type"),
                    status=_normalize_str_field("status"),
                )
            )

        return organized

    async def _run_document_organize(
        self,
        *,
        jobs: list[dict[str, object]],
        model_name: str,
        temperature: float,
        concurrency: int = 4,
    ) -> list[dict[str, object]]:
        try:
            max_tokens = int(self.config.max_tokens) if self.config.max_tokens is not None else None
        except Exception:
            max_tokens = None
        llm = get_langchain_chat_model(
            self.config.provider,
            model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        sem = asyncio.Semaphore(concurrency)

        async def run_one(job: dict[str, object]) -> dict[str, object]:
            OutputModel = job["OutputModel"]  # type: ignore[index]
            prompt = job["prompt"]  # type: ignore[index]
            filename = job.get("filename")  # type: ignore[index]

            structured_llm = llm.with_structured_output(
                OutputModel, temperature=temperature, max_tokens=8000
            )  # type: ignore[arg-type]

            attempt = 1
            delay_seconds = 1.0
            try:
                max_retries = int(self.config.max_retries)
            except Exception:
                max_retries = 5

            last_error: Exception | None = None
            while attempt <= max_retries:
                try:
                    async with sem:
                        output: BaseModel = await structured_llm.ainvoke(prompt)  # type: ignore[assignment, arg-type]
                    data = output.model_dump()  # type: ignore[attr-defined]
                    return {"data": data, "filename": filename}
                except Exception as e:  # pragma: no cover
                    last_error = e
                    if attempt >= max_retries:
                        break
                    jitter = random.uniform(0.0, delay_seconds * 0.25)
                    await asyncio.sleep(delay_seconds + jitter)
                    delay_seconds = min(delay_seconds * 2.0, 30.0)
                    attempt += 1

            logging.getLogger(__name__).error("Organizer job failed after retries: %r", last_error)
            return {
                "data": {},
                "filename": filename,
                "error": repr(last_error) if last_error else "unknown",
            }

        tasks = [asyncio.create_task(run_one(job)) for job in jobs]
        return await asyncio.gather(*tasks)
