from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import asyncio
import random

from .schema import (
    ContractTypeTemplate,
    DocumentClassification,
    ExtractionResult,
    ExtractedDatapoint,
    Paragraph,
    split_text_into_paragraphs,
)

from pydantic import BaseModel, Field, create_model

try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain.globals import set_debug
    set_debug(False)
except Exception as e:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    print('setting ChatOpenAI to None', e)

ScopeId = str


@dataclass(frozen=True)
class _ExtractionScope:
    kind: str  # 'clauses' | 'beginning' | 'document'
    clause_keys: Optional[Tuple[str, ...]] = None
    beginning_paragraphs: Optional[int] = None


@dataclass
class DatapointExtractorConfig:
    """Configuration for the datapoint extractor backend (LLM or rule-based)."""

    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    beginning_paragraphs: int = 20
    # Maximum number of concurrent scope extraction requests
    concurrency: int = 8
    # Number of retries for LLM calls on transient errors (e.g., TPM/rate limits)
    max_retries: int = 5


class DatapointExtractor:
    """Extracts datapoints from contract text, optionally using classifications."""

    def __init__(self, config: Optional[DatapointExtractorConfig] = None) -> None:
        self.config = config or DatapointExtractorConfig()

    def extract(
        self,
        text: str,
        template: ContractTypeTemplate,
        classified_clauses: Optional[DocumentClassification] = None,
    ) -> ExtractionResult:
        """Return extracted datapoints following the provided template.

        - Group datapoints by scope (same clause_keys, or beginning/document fallback)
        - Build scope text (selected clauses, beginning paragraphs, or whole document)
        - Use LangChain structured output to extract values for datapoints in each scope
        """
        if self.config.provider != "openai":
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required. Install with: pip install langchain langchain-openai"
            )

        paragraphs: List[Paragraph] = split_text_into_paragraphs(text)
        index_to_para: Dict[int, Paragraph] = {p.index: p for p in paragraphs}

        scopes: Dict[ScopeId, Tuple[_ExtractionScope, List[int]]] = {}
        scope_to_datapoints: Dict[ScopeId, List[int]] = {}

        # Assign datapoints to scopes
        for dp_idx, dp in enumerate(template.datapoints):
            if dp.clause_keys:
                keys_tuple = tuple(sorted({k for k in dp.clause_keys if k}))
                if keys_tuple:
                    scope_id = "clauses:" + ",".join(keys_tuple)
                    scope = _ExtractionScope(kind="clauses", clause_keys=keys_tuple)
                else:
                    # Empty clause_keys → treat as beginning by default
                    scope_id = "beginning"
                    scope = _ExtractionScope(kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs)
            else:
                # No clause_keys provided → default to beginning scope (first K paragraphs)
                scope_id = "beginning"
                scope = _ExtractionScope(kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs)

            if scope_id not in scopes:
                scopes[scope_id] = (scope, [])
                scope_to_datapoints[scope_id] = []
            scope_to_datapoints[scope_id].append(dp_idx)

        # Resolve paragraph indices per scope
        for scope_id, (scope, indices) in scopes.items():
            if scope.kind == "clauses" and scope.clause_keys and classified_clauses and classified_clauses.clause_to_paragraphs:
                gathered: List[int] = []
                for ck in scope.clause_keys:
                    gathered.extend(classified_clauses.clause_to_paragraphs.get(ck, []))
                indices.extend(sorted(set(gathered)))
            elif scope.kind == "beginning":
                k = scope.beginning_paragraphs or self.config.beginning_paragraphs
                indices.extend([p.index for p in paragraphs[:k]])
            else:
                # document: use full text later
                pass

        # Merge all empty-clauses scopes into a single document-scope job
        empty_clause_scope_ids: List[str] = [
            sid for sid, (scp, idxs) in scopes.items() if scp.kind == "clauses" and not idxs
        ]
        if empty_clause_scope_ids:
            merged_dp_indices: List[int] = []
            for sid in empty_clause_scope_ids:
                merged_dp_indices.extend(scope_to_datapoints.get(sid, []))
            merged_dp_indices = sorted(set(merged_dp_indices))
            # Remove empty clause scopes
            for sid in empty_clause_scope_ids:
                scopes.pop(sid, None)
                scope_to_datapoints.pop(sid, None)
            # Create or extend a single document scope holder
            doc_scope_id = "document"
            if doc_scope_id not in scopes:
                scopes[doc_scope_id] = (_ExtractionScope(kind="document"), [])
                scope_to_datapoints[doc_scope_id] = []
            scope_to_datapoints[doc_scope_id].extend(merged_dp_indices)

        # Build jobs for parallel execution
        jobs: List[Dict[str, object]] = []
        for scope_id, (scope, indices) in scopes.items():
            dp_indices = scope_to_datapoints[scope_id]
            datapoints = [template.datapoints[i] for i in dp_indices]

            # Build scope text
            if scope.kind == "beginning":
                k = scope.beginning_paragraphs or self.config.beginning_paragraphs
                selected = [p for p in paragraphs[:k]]
                scope_text = "\n\n".join(p.text for p in selected)
                evidence: Optional[List[int]] = [p.index for p in selected] if selected else []
            elif scope.kind == "clauses" and indices:
                scope_text = "\n\n".join(index_to_para[i].text for i in indices if i in index_to_para)
                evidence = indices
            else:
                scope_text = text
                evidence = None

            # Structured output model for this scope: each field returns {value, confidence, explanation}
            def _map_data_type_to_py_type(dt: Optional[str]):
                dt_norm = (dt or "").strip().lower()
                if dt_norm in ("str", "string", "text"):
                    return str
                if dt_norm in ("bool", "boolean"):
                    return bool
                if dt_norm in ("int", "integer"):
                    return int
                if dt_norm in ("float", "number", "double"):
                    return float
                if dt_norm in ("date", "datetime"):
                    # Dates are returned as ISO strings (YYYY-MM-DD)
                    return str
                if dt_norm == "enum":
                    # enums should return their code as a string
                    return str
                return str

            fields: Dict[str, Tuple[Optional[BaseModel], Field]] = {}
            for dp in datapoints:
                desc = dp.description or f"Extract value for '{dp.title}'."
                py_type = _map_data_type_to_py_type(getattr(dp, "data_type", None))
                # Create a typed FieldResult model per datapoint so `value` matches expected type
                FieldResultModel: BaseModel = create_model(
                    f"FieldResult_{dp.key}",
                    value=(Optional[py_type], Field(default=None, description="Extracted value or null if not found.")),
                    confidence=(
                        Optional[float],
                        Field(default=None, description="Model confidence in [0,1]; null if not available.", ge=0.0, le=1.0),
                    ),
                    explanation=(Optional[str], Field(default=None, description="Short rationale for the extracted value.")),
                )  # type: ignore[assignment]
                fields[dp.key] = (FieldResultModel, Field(..., description=desc))

            OutputModel: BaseModel = create_model("ScopeExtraction", **fields)  # type: ignore[assignment]

            instruction = (
                "You are an expert in the field of contract law. Your job is to extract the information from the contract.\n"
                f"The contract is a {template.description}.\n"
                "Extract the requested fields based only on the provided text in json format. Do not include any other text in your response.\n"
                "Return, for each field, an object with keys 'value', 'confidence', and 'explanation'.\n"
                "- value: the extracted value, or null if not present.\n"
                "- confidence: a float between 0 and 1, or null if you cannot judge.\n"
                "- explanation: short rationale or source cue for the value.\n"
                "For fields of type date, return the date in the format YYYY-MM-DD\n"
            )

            # Provide a concise list of fields to extract (only essential info)
            field_specs: List[str] = []
            for dp in datapoints:
                title = dp.title or dp.key
                desc = dp.description or ""
                data_type = dp.data_type or ""
                if dp.data_type == "enum":
                    data_type = f"enum: {dp.enum_key}"
                field_specs.append(f"- {dp.key}"  + (f" [{data_type}]" if data_type else "") + (f": {desc}" if desc else ""))

            # ENUMS section: include any enum lists referenced by datapoints in this scope
            enums_text = ""
            template_enums = getattr(template, "enums", None)
            if template_enums:
                enum_by_key = {e.key: e for e in template_enums}
                enum_keys = sorted({getattr(dp, "enum_key", None) for dp in datapoints if getattr(dp, "enum_key", None)})
                enum_lines: List[str] = []
                for ek in enum_keys:
                    enum_def = enum_by_key.get(ek)
                    if not enum_def:
                        continue
                    title = enum_def.title or ek
                    enum_lines.append(f"- {ek} ({title}):")
                    for opt in enum_def.options:
                        desc = opt.description or ""
                        enum_lines.append(f"  - {opt.code}: {desc}")
                if enum_lines:
                    enums_text = (
                        "For fields of type enum, use ONLY the provided codes (e.g., DRAFT, not Draft or draft)\n"
                        "ENUMS (for fields referencing an enum, return the code(s) exactly as listed):\n"
                        + "\n".join(enum_lines)
                        + "\n\n"
                    )

            prompt = (
                instruction
                + (enums_text if enums_text else "")
                + "FIELDS TO EXTRACT:\n"
                + "\n".join(field_specs)
                + "\n\nTEXT:\n"
                + scope_text
            )

            jobs.append(
                {
                    "OutputModel": OutputModel,
                    "prompt": prompt,
                    "datapoints": datapoints,
                    "evidence": evidence,
                }
            )

        # Execute jobs concurrently (up to 8)
        results = asyncio.run(
            self._run_scope_extractions(
                jobs=jobs,
                model_name=self.config.model or "gpt-4.1-mini",
                temperature=float(self.config.temperature),
                concurrency=int(self.config.concurrency),
            )
        )

        # Build extracted list from results (robust to missing/failed jobs)
        extracted: List[ExtractedDatapoint] = []
        for res in results:
            if not isinstance(res, dict):
                continue
            data_obj = res.get("data")
            data: Dict[str, Optional[dict]] = data_obj if isinstance(data_obj, dict) else {}
            datapoints = res.get("datapoints") or []
            evidence = res.get("evidence")
            for dp in datapoints:  # type: ignore[assignment]
                item = (data or {}).get(dp.key) or {}
                value = item.get("value")
                conf = item.get("confidence")
                expl = item.get("explanation")
                if isinstance(conf, (int, float)):
                    conf = max(0.0, min(1.0, float(conf)))
                else:
                    conf = None
                extracted.append(
                    ExtractedDatapoint(
                        key=dp.key,
                        value=value if value is not None else None,
                        explanation=expl if isinstance(expl, str) and expl.strip() else None,
                        evidence_paragraph_indices=evidence,
                        confidence=conf,
                    )
                )

        return ExtractionResult(datapoints=extracted)

    async def _run_scope_extractions(
        self,
        *,
        jobs: List[Dict[str, object]],
        model_name: str,
        temperature: float,
        concurrency: int = 8,
    ) -> List[Dict[str, object]]:
        """Run scope extractions concurrently and return raw model_dump data per job."""
        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required. Install with: pip install langchain langchain-openai"
            )
        # Load .env if available to populate OPENAI_API_KEY
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass

        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)
        sem = asyncio.Semaphore(concurrency)

        async def run_one(job: Dict[str, object]) -> Dict[str, object]:
            OutputModel = job["OutputModel"]  # type: ignore[index]
            prompt = job["prompt"]  # type: ignore[index]
            datapoints = job["datapoints"]  # type: ignore[index]
            evidence = job["evidence"]  # type: ignore[index]

            # print('--------------------------------')
            # print('OutputModel', OutputModel.model_json_schema())
            # print('datapoints', [dp.key for dp in datapoints])
            # print('evidence', evidence)
            # print('prompt', prompt)
            # print('--------------------------------')

            structured_llm = llm.with_structured_output(OutputModel, temperature=temperature, max_tokens=8000)  # type: ignore[arg-type]

            attempt = 1
            delay_seconds = 1.0
            try:
                max_retries = int(self.config.max_retries)
            except Exception:
                max_retries = 5

            last_error: Optional[Exception] = None
            while attempt <= max_retries:
                try:
                    async with sem:
                        output: BaseModel = await structured_llm.ainvoke(prompt)  # type: ignore[assignment]
                    data = output.model_dump()  # type: ignore[attr-defined]
                    return {"data": data, "datapoints": datapoints, "evidence": evidence}
                except Exception as e:  # pragma: no cover
                    last_error = e
                    if attempt >= max_retries:
                        break
                    # Exponential backoff with jitter
                    jitter = random.uniform(0.0, delay_seconds * 0.25)
                    await asyncio.sleep(delay_seconds + jitter)
                    delay_seconds = min(delay_seconds * 2.0, 30.0)
                    attempt += 1

            # Return an empty result for this job so the caller can proceed
            print('run_one error after retries:', repr(last_error))
            return {"data": {}, "datapoints": datapoints, "evidence": evidence, "error": repr(last_error) if last_error else "unknown"}

        tasks = [asyncio.create_task(run_one(job)) for job in jobs]
        return await asyncio.gather(*tasks)


