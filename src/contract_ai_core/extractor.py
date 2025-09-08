from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Optional, cast

from pydantic import BaseModel, Field, create_model

from .schema import (
    ContractTypeTemplate,
    DatapointDefinition,
    DocumentClassification,
    ExtractedDatapoint,
    ExtractionResult,
    Paragraph,
)

try:
    from langchain.globals import set_debug

    set_debug(False)
except Exception as e:  # pragma: no cover
    logging.getLogger(__name__).warning("LangChain globals unavailable: %r", e)

from .utilities import get_langchain_chat_model

ScopeId = str


@dataclass(frozen=True)
class _ExtractionScope:
    kind: str  # 'clauses' | 'beginning' | 'document'
    clause_keys: tuple[str, ...] | None = None
    beginning_paragraphs: int | None = None


@dataclass
class DatapointExtractorConfig:
    """Configuration for the datapoint extractor backend (LLM or rule-based)."""

    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    beginning_paragraphs: int = 20
    # Maximum number of concurrent scope extraction requests
    concurrency: int = 8
    # Number of retries for LLM calls on transient errors (e.g., TPM/rate limits)
    max_retries: int = 5


class DatapointExtractor:
    """Extracts datapoints from contract text, optionally using classifications."""

    def __init__(self, config: DatapointExtractorConfig | None = None) -> None:
        self.config = config or DatapointExtractorConfig()

    def extract(
        self,
        paragraphs: list[Paragraph],
        template: ContractTypeTemplate,
        classified_clauses: DocumentClassification | None = None,
    ) -> ExtractionResult:
        """Return extracted datapoints following the provided template.

        - Group datapoints by scope (same clause_keys, or beginning/document fallback)
        - Build scope text (selected clauses, beginning paragraphs, or whole document)
        - Use LangChain structured output to extract values for datapoints in each scope
        """
        if self.config.provider not in ("openai", "azure", "anthropic"):
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        index_to_para: dict[int, Paragraph] = {p.index: p for p in paragraphs}

        scopes: dict[ScopeId, tuple[_ExtractionScope, list[int]]] = {}
        scope_to_datapoints: dict[ScopeId, list[int]] = {}

        # Build lookup for structures by key for complex object datapoints
        structures_by_key: dict[str, Any] = {}
        try:
            for s in getattr(template, "structures", []) or []:
                key = getattr(s, "structure_key", None)
                if isinstance(key, str) and key:
                    structures_by_key[key] = s
        except Exception:
            structures_by_key = {}

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
                    scope = _ExtractionScope(
                        kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs
                    )
            else:
                # No clause_keys provided → default to beginning scope (first K paragraphs)
                scope_id = "beginning"
                scope = _ExtractionScope(
                    kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs
                )

            if scope_id not in scopes:
                scopes[scope_id] = (scope, [])
                scope_to_datapoints[scope_id] = []
            scope_to_datapoints[scope_id].append(dp_idx)

        # Resolve paragraph indices per scope
        for _scope_id, (scope, indices) in scopes.items():
            if (
                scope.kind == "clauses"
                and scope.clause_keys
                and classified_clauses
                and classified_clauses.clause_to_paragraphs
            ):
                gathered: list[int] = []
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
        empty_clause_scope_ids: list[str] = [
            sid for sid, (scp, idxs) in scopes.items() if scp.kind == "clauses" and not idxs
        ]
        if empty_clause_scope_ids:
            logging.getLogger(__name__).debug(
                "Merging %d empty clause scopes into document scope", len(empty_clause_scope_ids)
            )
            merged_dp_indices: list[int] = []
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
        jobs: list[dict[str, object]] = []
        for scope_id, (scope, indices) in scopes.items():
            dp_indices = scope_to_datapoints[scope_id]
            datapoints = [template.datapoints[i] for i in dp_indices]

            # Build scope text
            if scope.kind == "beginning":
                k = scope.beginning_paragraphs or self.config.beginning_paragraphs
                selected = [p for p in paragraphs[:k]]
                scope_text = "\n\n".join(p.text for p in selected)
                evidence: list[int] | None = [p.index for p in selected] if selected else []
            elif scope.kind == "clauses" and indices:
                scope_text = "\n\n".join(
                    index_to_para[i].text for i in indices if i in index_to_para
                )
                evidence = indices
            else:
                scope_text = "\n\n".join(p.text for p in paragraphs)
                evidence = None

            # Structured output model for this scope: each field returns {value, confidence, explanation}
            def _map_data_type_to_py_type(dt: Any | None) -> Any:
                dt_norm = (dt or "").strip().lower()
                if dt_norm == "str":
                    return str
                if dt_norm == "bool":
                    return bool
                if dt_norm == "int":
                    return int
                if dt_norm == "float":
                    return float
                if dt_norm == "date":
                    # Dates are returned as ISO strings (YYYY-MM-DD)
                    return str
                if dt_norm == "enum":
                    # enums should return their code as a string
                    return str
                if dt_norm == "list[str]":
                    return list[str]
                if dt_norm == "list[bool]":
                    return list[bool]
                if dt_norm == "list[int]":
                    return list[int]
                if dt_norm == "list[float]":
                    return list[float]
                if dt_norm == "list[date]":
                    # Dates are returned as ISO strings (YYYY-MM-DD)
                    return list[str]
                if dt_norm == "list[enum]":
                    # enums should return their code as a string
                    return list[str]
                return str

            def _parse_structure_type(dt: Any | None) -> tuple[str, str | None]:
                """Return (kind, structure_key) where kind in {simple, object, list_object}. Tolerates spaces and optional brackets."""
                raw = (dt or "").strip().lower()
                # list[object:struct]
                m = re.match(r"^list\[\s*object\s*:\s*\[?([a-z0-9_\-]+)\]?\s*\]$", raw)
                if m:
                    return "list_object", m.group(1)
                # object:struct
                m = re.match(r"^object\s*:\s*\[?([a-z0-9_\-]+)\]?$", raw)
                if m:
                    return "object", m.group(1)
                return "simple", None

            def _map_element_type(el_type: str | None) -> Any:
                et = (el_type or "").strip().lower()
                # Reuse the scalar/list mapping above
                return _map_data_type_to_py_type(et)

            # Store dynamic field definitions for create_model; Field(...) returns FieldInfo at runtime
            # Use Any for the tuple elements to satisfy mypy
            fields: dict[str, tuple[Any, Any]] = {}
            for dp in datapoints:
                desc = dp.description or f"Extract value for '{dp.title}'."
                raw_dt = getattr(dp, "data_type", None)
                kind, struct_key = _parse_structure_type(raw_dt)

                if (
                    kind in ("object", "list_object")
                    and struct_key
                    and (struct_key in structures_by_key)
                ):
                    # Build nested result model per element
                    struct_def = structures_by_key[struct_key]
                    # Create per-element result models
                    element_fields: dict[str, tuple[type[Any], Any]] = {}
                    for el in getattr(struct_def, "elements", []) or []:
                        el_desc = (
                            getattr(el, "description", None)
                            or f"Extract '{getattr(el, 'title', getattr(el, 'key', ''))}'."
                        )
                        el_py_type = _map_element_type(getattr(el, "data_type", None))
                        ElementResultModel: type[BaseModel] = create_model(
                            f"FieldResult_{dp.key}_{getattr(el, 'key', 'element')}",
                            value=(
                                Optional[el_py_type],
                                Field(
                                    default=None,
                                    description="Extracted value or null if not found.",
                                ),
                            ),
                            confidence=(
                                Optional[float],
                                Field(
                                    default=None,
                                    description="Model confidence in [0,1]; null if not available.",
                                    ge=0.0,
                                    le=1.0,
                                ),
                            ),
                            explanation=(
                                Optional[str],
                                Field(
                                    default=None,
                                    description="Short rationale for the extracted value.",
                                ),
                            ),
                        )
                        element_fields[getattr(el, "key", "element")] = (
                            ElementResultModel,
                            Field(..., description=el_desc),
                        )

                    # Object model aggregating element results
                    ObjectModel: type[BaseModel] = create_model(
                        f"Object_{dp.key}_{struct_key}",
                        **element_fields,  # type: ignore[arg-type]
                    )

                    # FieldResult model for datapoint: value is object or list[object]
                    if kind == "list_object":
                        value_type: Any = list[ObjectModel]  # type: ignore[valid-type]
                    else:
                        value_type = ObjectModel

                    FieldResultModel: type[BaseModel] = create_model(
                        f"FieldResult_{dp.key}",
                        value=(
                            Optional[value_type],
                            Field(
                                default=None,
                                description="Extracted object(s) or null if not found.",
                            ),
                        ),
                        confidence=(
                            Optional[float],
                            Field(
                                default=None,
                                description="Model confidence in [0,1]; null if not available.",
                                ge=0.0,
                                le=1.0,
                            ),
                        ),
                        explanation=(
                            Optional[str],
                            Field(
                                default=None, description="Short rationale for the extracted value."
                            ),
                        ),
                    )
                else:
                    # Simple/enum/list scalar types
                    py_type = _map_data_type_to_py_type(raw_dt)
                    FieldResultModel = create_model(
                        f"FieldResult_{dp.key}",
                        value=(
                            Optional[py_type],
                            Field(
                                default=None, description="Extracted value or null if not found."
                            ),
                        ),
                        confidence=(
                            Optional[float],
                            Field(
                                default=None,
                                description="Model confidence in [0,1]; null if not available.",
                                ge=0.0,
                                le=1.0,
                            ),
                        ),
                        explanation=(
                            Optional[str],
                            Field(
                                default=None, description="Short rationale for the extracted value."
                            ),
                        ),
                    )

                fields[dp.key] = (FieldResultModel, Field(..., description=desc))

            _field_defs = cast(dict[str, tuple[type[Any], Any]], fields)
            OutputModel: type[BaseModel] = create_model("ScopeExtraction", **_field_defs)

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
            field_specs: list[str] = []
            for dp in datapoints:
                title = dp.title or dp.key
                desc = dp.description or ""
                data_type = dp.data_type or ""
                kind, struct_key = _parse_structure_type(data_type)
                if (dp.data_type or "") == "enum":
                    data_type = f"enum: {dp.enum_key}"
                else:
                    # Preserve original string for specs in case of structure
                    if kind == "object" and struct_key:
                        data_type = f"object:{struct_key}"
                    elif kind == "list_object" and struct_key:
                        data_type = f"list[object:{struct_key}]"
                field_specs.append(
                    f"- {dp.key}"
                    + (f" [{data_type}]" if data_type else "")
                    + (f": {desc}" if desc else "")
                )

            # ENUMS section: include any enum lists referenced by datapoints (and structure elements) in this scope
            enums_text = ""
            template_enums = getattr(template, "enums", None)
            if template_enums:
                enum_by_key = {e.key: e for e in template_enums}
                enum_keys_set: set[str] = set()
                for dp in datapoints:
                    # direct enum on datapoint
                    ek = getattr(dp, "enum_key", None)
                    if isinstance(ek, str) and ek:
                        enum_keys_set.add(ek)
                    # if structure type, include enum keys from its elements
                    knd, skey = _parse_structure_type(getattr(dp, "data_type", None))
                    if skey and skey in structures_by_key:
                        try:
                            for el in getattr(structures_by_key[skey], "elements", []) or []:
                                el_ek = getattr(el, "enum_key", None)
                                if isinstance(el_ek, str) and el_ek:
                                    enum_keys_set.add(el_ek)
                        except Exception:
                            pass
                enum_keys = sorted(enum_keys_set)
                enum_lines: list[str] = []
                for ek in enum_keys:
                    enum_def = enum_by_key.get(ek)
                    if not enum_def:
                        continue
                    title = enum_def.title if enum_def.title is not None else ek
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

        # Execute jobs concurrently
        results = asyncio.run(
            self._run_scope_extractions(
                jobs=jobs,
                model_name=self.config.model or "gpt-4.1-mini",
                temperature=float(self.config.temperature),
                concurrency=int(self.config.concurrency),
            )
        )

        # Build extracted list from results (robust to missing/failed jobs)
        extracted: list[ExtractedDatapoint] = []
        for res in results:
            if not isinstance(res, dict):
                continue
            data_obj = res.get("data")
            data: dict[str, dict | None] = data_obj if isinstance(data_obj, dict) else {}
            dp_obj = res.get("datapoints")
            if isinstance(dp_obj, list):
                datapoints = cast(list[DatapointDefinition], dp_obj)
            else:
                datapoints = []
            ev_obj = res.get("evidence")
            evidence = (
                cast(list[int] | None, ev_obj)
                if (isinstance(ev_obj, list) or ev_obj is None)
                else None
            )
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
        jobs: list[dict[str, object]],
        model_name: str,
        temperature: float,
        concurrency: int = 8,
    ) -> list[dict[str, object]]:
        """Run scope extractions concurrently and return raw model_dump data per job.

        Retries with exponential backoff are applied per job to handle transient rate limits.
        """
        # Build LLM using common utility (supports openai/azure/anthropic)
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
            datapoints = job["datapoints"]  # type: ignore[index]
            evidence = job["evidence"]  # type: ignore[index]

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
            logging.getLogger(__name__).error("Job failed after retries: %r", last_error)
            return {
                "data": {},
                "datapoints": datapoints,
                "evidence": evidence,
                "error": repr(last_error) if last_error else "unknown",
            }

        tasks = [asyncio.create_task(run_one(job)) for job in jobs]
        return await asyncio.gather(*tasks)
