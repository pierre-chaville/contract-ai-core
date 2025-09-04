from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, cast

from pydantic import BaseModel, Field, create_model

from .schema import (
    ContractTypeTemplate,
    DocumentClassification,
    ExtractionScope,
    GuidelineDefinition,
    Paragraph,
    ReviewedGuideline,
)
from .utilities import get_langchain_chat_model

ScopeId = str


@dataclass(frozen=True)
class _ReviewScope:
    kind: str  # 'clauses' | 'beginning' | 'document'
    clause_keys: tuple[str, ...] | None = None
    beginning_paragraphs: int | None = None


@dataclass
class GuidelineReviewerConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    beginning_paragraphs: int = 20
    # Maximum number of concurrent clause-scope review requests
    concurrency: int = 8
    # Number of retries for LLM calls on transient errors (e.g., TPM/rate limits)
    max_retries: int = 5


class GuidelineReviewer:
    """Reviews contract text against template guidelines using an LLM backend."""

    def __init__(self, config: GuidelineReviewerConfig | None = None) -> None:
        self.config = config or GuidelineReviewerConfig()

    def review(
        self,
        *,
        paragraphs: list[Paragraph],
        template: ContractTypeTemplate,
        classified_clauses: DocumentClassification | None = None,
    ) -> list[ReviewedGuideline]:
        """Return guideline reviews for the provided text and template.

        Steps:
        - Group guidelines by scope (same clause_keys; beginning; document)
        - Build scope text (selected clauses, beginning paragraphs, or whole document)
        - Use LangChain structured output to determine whether guidelines are met
        """
        if self.config.provider not in ("openai", "azure", "anthropic"):
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        index_to_para: dict[int, Paragraph] = {p.index: p for p in paragraphs}

        scopes: dict[ScopeId, tuple[_ReviewScope, list[int]]] = {}
        scope_to_guidelines: dict[ScopeId, list[int]] = {}

        def _scope_id_for_guideline(g: GuidelineDefinition) -> tuple[ScopeId, _ReviewScope]:
            # Determine scope based on guideline.scope and clause_keys
            if g.scope == ExtractionScope.CLAUSE:
                keys_tuple = tuple(sorted({k for k in (g.clause_keys or []) if k}))
                if keys_tuple:
                    sid = "clauses:" + ",".join(keys_tuple)
                    scope = _ReviewScope(kind="clauses", clause_keys=keys_tuple)
                else:
                    sid = "beginning"
                    scope = _ReviewScope(
                        kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs
                    )
            elif g.scope == ExtractionScope.BEGINNING:
                sid = "beginning"
                scope = _ReviewScope(
                    kind="beginning", beginning_paragraphs=self.config.beginning_paragraphs
                )
            else:
                sid = "document"
                scope = _ReviewScope(kind="document")
            return sid, scope

        # Assign guidelines to scopes
        for g_idx, g in enumerate(template.guidelines):
            sid, scope = _scope_id_for_guideline(g)
            if sid not in scopes:
                scopes[sid] = (scope, [])
                scope_to_guidelines[sid] = []
            scope_to_guidelines[sid].append(g_idx)

        # Resolve paragraph indices per scope
        for _sid, (scope, indices) in scopes.items():
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
            merged_g_indices: list[int] = []
            for sid in empty_clause_scope_ids:
                merged_g_indices.extend(scope_to_guidelines.get(sid, []))
            merged_g_indices = sorted(set(merged_g_indices))
            for sid in empty_clause_scope_ids:
                scopes.pop(sid, None)
                scope_to_guidelines.pop(sid, None)
            doc_scope_id = "document"
            if doc_scope_id not in scopes:
                scopes[doc_scope_id] = (_ReviewScope(kind="document"), [])
                scope_to_guidelines[doc_scope_id] = []
            scope_to_guidelines[doc_scope_id].extend(merged_g_indices)

        # Build jobs (separate lists so we can parallelize clauses only)
        clause_jobs: list[dict[str, object]] = []
        other_jobs: list[dict[str, object]] = []
        for scope_id, (scope, indices) in scopes.items():
            g_indices = scope_to_guidelines[scope_id]
            gls = [template.guidelines[i] for i in g_indices]

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

            # Structured output model for this scope: each field returns
            # {guideline_matched, confidence, fallback_guideline_matched, confidence_fallback, explanation}
            fields: dict[str, tuple[type[Any], Any]] = {}
            for g in gls:
                desc_lines: list[str] = [
                    f"Primary guideline: {g.guideline}",
                ]
                if g.fallback_guideline:
                    desc_lines.append(f"Fallback guideline: {g.fallback_guideline}")
                if g.priority:
                    desc_lines.append(f"Priority: {g.priority}")
                field_desc = "\n".join(desc_lines)

                ResultModel: type[BaseModel] = create_model(
                    f"GuidelineResult_{g.key}",
                    guideline_matched=(
                        Optional[bool],
                        Field(
                            default=None, description="True if the primary guideline is satisfied."
                        ),
                    ),
                    confidence=(
                        Optional[float],
                        Field(
                            default=None,
                            description="Confidence in [0,1] for the primary guideline decision.",
                            ge=0.0,
                            le=1.0,
                        ),
                    ),
                    fallback_guideline_matched=(
                        Optional[bool],
                        Field(
                            default=None,
                            description="True if primary is not satisfied but fallback is satisfied.",
                        ),
                    ),
                    confidence_fallback=(
                        Optional[float],
                        Field(
                            default=None,
                            description="Confidence in [0,1] for the fallback guideline decision.",
                            ge=0.0,
                            le=1.0,
                        ),
                    ),
                    explanation=(
                        Optional[str],
                        Field(
                            default=None,
                            description=(
                                "Short rationale referencing the text supporting the decision(s)."
                            ),
                        ),
                    ),
                )
                fields[g.key] = (ResultModel, Field(..., description=field_desc))

            OutputModel: type[BaseModel] = create_model("ScopeGuidelineReview", **fields)

            instruction = (
                "You are an expert legal reviewer. Assess whether each guideline is satisfied by the provided text.\n"
                "For each item:\n"
                "- guideline_matched: true/false/null for the PRIMARY guideline.\n"
                "- confidence: float 0..1 or null.\n"
                "- If primary is not satisfied, check the fallback (if provided):\n"
                "  - fallback_guideline_matched: true/false/null.\n"
                "  - confidence_fallback: float 0..1 or null.\n"
                "- explanation: brief justification referring to the text.\n"
                "Only use the provided text. Return STRICTLY the JSON for the schema.\n"
            )

            prompt = instruction + "\nTEXT:\n" + scope_text

            job = {
                "OutputModel": OutputModel,
                "prompt": prompt,
                "guidelines": gls,
                "evidence": evidence,
            }

            if scope.kind == "clauses":
                clause_jobs.append(job)
            else:
                other_jobs.append(job)

        # Execute clause-scope jobs concurrently (limited), others sequentially
        clause_results = (
            asyncio.run(
                self._run_scope_reviews(
                    jobs=clause_jobs,
                    model_name=self.config.model or "gpt-4.1-mini",
                    temperature=float(self.config.temperature),
                    concurrency=int(self.config.concurrency),
                )
            )
            if clause_jobs
            else []
        )

        other_results: list[dict[str, object]] = []
        for job in other_jobs:
            other_results.extend(
                asyncio.run(
                    self._run_scope_reviews(
                        jobs=[job],
                        model_name=self.config.model or "gpt-4.1-mini",
                        temperature=float(self.config.temperature),
                        concurrency=1,
                    )
                )
            )

        # Build reviewed list from results
        reviewed: list[ReviewedGuideline] = []
        for res in clause_results + other_results:
            if not isinstance(res, dict):
                continue
            data_obj = res.get("data")
            data: dict[str, dict | None] = data_obj if isinstance(data_obj, dict) else {}
            g_obj = res.get("guidelines")
            guidelines = cast(list[GuidelineDefinition], g_obj) if isinstance(g_obj, list) else []
            ev_obj = res.get("evidence")
            evidence = (
                cast(list[int] | None, ev_obj)
                if (isinstance(ev_obj, list) or ev_obj is None)
                else None
            )

            for g in guidelines:
                item = (data or {}).get(g.key) or {}
                primary = item.get("guideline_matched")
                conf = item.get("confidence")
                fallback = item.get("fallback_guideline_matched")
                conf_fb = item.get("confidence_fallback")
                expl = item.get("explanation")

                # Normalize types
                primary_b = bool(primary) if isinstance(primary, bool) else None
                fallback_b = bool(fallback) if isinstance(fallback, bool) else None
                if isinstance(conf, (int, float)):
                    conf_f = max(0.0, min(1.0, float(conf)))
                else:
                    conf_f = None
                if isinstance(conf_fb, (int, float)):
                    conf_fb_f = max(0.0, min(1.0, float(conf_fb)))
                else:
                    conf_fb_f = None
                explanation = expl if isinstance(expl, str) and expl.strip() else None

                reviewed.append(
                    ReviewedGuideline(
                        key=g.key,
                        guideline_matched=primary_b if primary_b is not None else False,
                        confidence=conf_f,
                        fallback_guideline_matched=fallback_b if fallback_b is not None else False,
                        confidence_fallback=conf_fb_f,
                        explanation=explanation,
                        evidence_paragraph_indices=evidence,
                    )
                )

        return reviewed

    async def _run_scope_reviews(
        self,
        *,
        jobs: list[dict[str, object]],
        model_name: str,
        temperature: float,
        concurrency: int = 8,
    ) -> list[dict[str, object]]:
        """Run scope reviews concurrently and return raw model_dump data per job.

        Retries with exponential backoff are applied per job to handle transient rate limits.
        """
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
            guidelines = job["guidelines"]  # type: ignore[index]
            evidence = job.get("evidence")  # type: ignore[index]

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
                    return {"data": data, "guidelines": guidelines, "evidence": evidence}
                except Exception as e:  # pragma: no cover
                    last_error = e
                    if attempt >= max_retries:
                        break
                    jitter = random.uniform(0.0, delay_seconds * 0.25)
                    await asyncio.sleep(delay_seconds + jitter)
                    delay_seconds = min(delay_seconds * 2.0, 30.0)
                    attempt += 1

            logging.getLogger(__name__).error(
                "Guideline review job failed after retries: %r", last_error
            )
            return {
                "data": {},
                "guidelines": guidelines,
                "error": repr(last_error) if last_error else "unknown",
            }

        tasks = [asyncio.create_task(run_one(job)) for job in jobs]
        return await asyncio.gather(*tasks)
