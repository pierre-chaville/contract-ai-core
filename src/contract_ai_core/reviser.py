from __future__ import annotations

"""Contract reviser: plan and apply amendment instructions to produce a restated contract.

Public API:
- ContractReviser.analyze_amendments: parse amendment paragraphs into structured instructions
- ContractReviser.find_revisions_targets: locate target spans in the original contract
- ContractReviser.apply_revisions: generate revised paragraphs for each instruction
- ContractReviser.generate_amended_and_restated: end-to-end pipeline helper
"""

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .schema import (
    ContractTypeTemplate,
    Paragraph,
    RevisedContract,
    RevisedSection,
    RevisionInstruction,
    RevisionInstructionTarget,
)

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


@dataclass
class ContractReviserConfig:
    """Configuration for the contract reviser backend."""

    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None


class ContractReviser:
    """Produces an amended and restated contract according to a template."""

    def __init__(self, config: ContractReviserConfig | None = None) -> None:
        self.config = config or ContractReviserConfig()

    def analyze_amendments(
        self, amendment_paragraphs: Sequence[Paragraph], template: ContractTypeTemplate
    ) -> Sequence[RevisionInstruction]:
        """Return a sequence of high-level revision instructions.

        For example, replace this section ... with this text: ...
        """
        if self.config.provider != "openai":
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required. Install with: pip install langchain langchain-openai"
            )

        # Build amendment lines block
        lines_block = "\n".join(f"{p.index}: {p.text}" for p in amendment_paragraphs)

        class AmendmentItem(BaseModel):
            amendment_start_line: int = Field(
                ..., description="Start line index (inclusive) in the amendment text"
            )
            amendment_end_line: int = Field(
                ..., description="End line index (inclusive) in the amendment text"
            )
            amendment_span_text: str = Field(..., description="Text of the amendment span")
            target_section: str = Field(
                ..., description="Target section identifier in the base contract"
            )
            confidence_target: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0,1]")
            change_explanation: str = Field(..., description="Short rationale of the change")

        class AmendmentsOutput(BaseModel):
            amendments: list[AmendmentItem]

        # Load .env and API key
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")

        llm = ChatOpenAI(
            model=self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            api_key=api_key,
        )
        structured_llm = llm.with_structured_output(AmendmentsOutput)  # type: ignore[arg-type]

        instruction = (
            "You are an expert legal analyst. Given amendment paragraphs with line numbers, identify "
            "each elementary amendment. For every amendment, return: start line, end line (inclusive), "
            "target section identifier in the base contract as a numbering string (e.g., 'Part 1 (c) (ii)'), "
            "a confidence in [0,1], and a brief explanation. If multiple discrete changes appear, split them into "
            "separate amendment items."
        )
        prompt = instruction + "\n\nAmendment paragraphs (line_number: text):\n" + lines_block

        output: AmendmentsOutput = structured_llm.invoke(prompt)  # type: ignore[assignment]
        results: list[RevisionInstruction] = []
        for item in output.amendments:
            results.append(
                RevisionInstruction(
                    amendment_start_line=item.amendment_start_line,
                    amendment_end_line=item.amendment_end_line,
                    amendment_span_text=item.amendment_span_text,
                    target_section=item.target_section,
                    confidence_target=item.confidence_target,
                    change_explanation=item.change_explanation,
                )
            )
        return results

    def find_revisions_targets(
        self,
        contracts_paragraphs: Sequence[Paragraph],
        instructions: Sequence[RevisionInstruction],
    ) -> Sequence[RevisionInstructionTarget]:
        """for each revision instructions, find the relevant target paragraphs in the original text."""
        if self.config.provider != "openai":
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required. Install with: pip install langchain langchain-openai"
            )
        # Build contract lines block
        contract_block = "\n".join(f"{p.index}: {p.text}" for p in contracts_paragraphs)

        # Build compact JSON-like description of targets to locate
        targets_desc = [
            {
                "target_section": ins.target_section,
                "change_explanation": ins.change_explanation,
            }
            for ins in instructions
        ]

        class TargetLocationItem(BaseModel):
            target_section: str = Field(..., description="Target section identifier as provided")
            start_line: int = Field(..., description="Start line index (inclusive) in the contract")
            end_line: int = Field(..., description="End line index (inclusive) in the contract")
            confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0,1]")
            explanation: str = Field(..., description="Brief rationale for the chosen span")

        class TargetLocations(BaseModel):
            locations: list[TargetLocationItem]

        # Load .env and API key
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")

        llm = ChatOpenAI(
            model=self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            api_key=api_key,
        )
        structured_llm = llm.with_structured_output(TargetLocations)  # type: ignore[arg-type]

        instruction_text = (
            "You are an expert legal analyst. Given the base contract paragraphs (with line numbers) "
            "and a list of target sections to locate, identify for each target the start and end line "
            "indices (inclusive) in the contract that best correspond to the target section."
        )
        guidance_text = "If a target cannot be confidently located, return start_line=-1 and end_line=-1 with a low confidence."

        import json as _json

        prompt = (
            instruction_text
            + "\n\nTargets to locate (JSON):\n"
            + _json.dumps(targets_desc, ensure_ascii=False)
            + "\n\nContract paragraphs (line_number: text):\n"
            + contract_block
            + "\n\n"
            + guidance_text
        )

        output: TargetLocations = structured_llm.invoke(prompt)  # type: ignore[assignment]

        # Map results back to input instructions by target_section
        results: list[RevisionInstructionTarget] = []
        # Build lookup from target_section to item
        loc_by_section = {loc.target_section: loc for loc in output.locations}

        for ins in instructions:
            loc = loc_by_section.get(ins.target_section)
            if loc is None or loc.start_line is None or loc.end_line is None:
                target_indices: Sequence[int] | None = None
                conf = 0.0
                explanation = "Not found"
            else:
                if loc.start_line >= 0 and loc.end_line >= loc.start_line:
                    target_indices = list(range(loc.start_line, loc.end_line + 1))
                else:
                    target_indices = None
                conf = float(loc.confidence)
                explanation = loc.explanation
            if target_indices is None:
                logging.getLogger(__name__).warning(
                    "No target indices for section: %s", ins.target_section
                )
                target_indices = []
            results.append(
                RevisionInstructionTarget(
                    amendment_start_line=ins.amendment_start_line,
                    amendment_end_line=ins.amendment_end_line,
                    amendment_span_text=ins.amendment_span_text,
                    target_section=ins.target_section,
                    confidence_target=ins.confidence_target,
                    change_explanation=ins.change_explanation,
                    target_paragraph_indices=target_indices,
                    confidence_target_paragraph_indices=conf,
                    target_paragraph_explanation=explanation,
                )
            )

        return results

    def apply_revisions(
        self,
        section_paragraphs: Sequence[Paragraph],
        instructions: Sequence[RevisionInstructionTarget],
    ) -> Sequence[RevisedSection]:
        """Apply the revision instructions on the section to produce the revised section."""
        if self.config.provider != "openai":
            raise NotImplementedError(f"Unsupported provider: {self.config.provider}")

        if ChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is required. Install with: pip install langchain langchain-openai"
            )

        # Prepare jobs per instruction
        logging.getLogger(__name__).info("Applying %d revision jobs", len(instructions))
        jobs = []
        for ins in instructions:
            target_idxs = list(ins.target_paragraph_indices) if ins.target_paragraph_indices else []
            initial = [
                section_paragraphs[i] for i in target_idxs if 0 <= i < len(section_paragraphs)
            ]

            # Build context only with target span lines
            target_block = (
                "\n".join(f"{p.index}: {p.text}" for p in initial)
                if initial
                else "<no-target-found>"
            )

            class ApplyOutput(BaseModel):
                revised_paragraphs: list[str] = Field(
                    ..., description="List of revised paragraphs text for the target span, ordered."
                )
                confidence_revision: float = Field(
                    ..., ge=0.0, le=1.0, description="Confidence in [0,1] for the applied revision"
                )
                revision_explanation: str = Field(
                    ..., description="Brief explanation of the applied revision"
                )

            jobs.append(
                {
                    "ins": ins,
                    "initial": initial,
                    "target_block": target_block,
                    "model": ApplyOutput,
                }
            )

        # Load API key once
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")

        async def run_job(job):
            ApplyOutput = job["model"]
            llm = ChatOpenAI(
                model=self.config.model or "gpt-4.1-mini",
                temperature=float(self.config.temperature),
                api_key=api_key,
            )
            structured_llm = llm.with_structured_output(ApplyOutput)  # type: ignore[arg-type]

            ins: RevisionInstructionTarget = job["ins"]
            instruction_text = (
                "You are an expert legal editor. Given ONLY the target span paragraphs (with line numbers), "
                "update that span to implement the described amendment. Do not introduce unrelated changes. "
                "Preserve style, defined terms, and numbering within the span."
            )
            guidance_text = (
                f"TARGET SECTIONr: {ins.target_section}\n"
                f"AMENDMENT INSTRUCTION: {ins.amendment_span_text}\n"
                # "If no target span is provided, return an empty list for revised_paragraphs and low confidence."
            )
            prompt = (
                instruction_text
                + "\n\nTARGET SPAN (line_number: text):\n"
                + job["target_block"]
                + "\n\n"
                + guidance_text
            )
            # Debug logging can be enabled if needed
            # logging.getLogger(__name__).debug("Running job for section %s", ins.target_section)
            out: ApplyOutput = await structured_llm.ainvoke(prompt)  # type: ignore[assignment]
            return job, out

        # Run with limited concurrency
        async def runner():
            sem = asyncio.Semaphore(8)

            async def wrapped(job):
                async with sem:
                    return await run_job(job)

            tasks = [asyncio.create_task(wrapped(j)) for j in jobs]
            return await asyncio.gather(*tasks)

        results = asyncio.run(runner()) if jobs else []

        revised_sections: list[RevisedSection] = []
        for job, out in results:
            ins: RevisionInstructionTarget = job["ins"]
            initial: list[Paragraph] = job["initial"]

            # Build revised Paragraph objects, assigning indices starting from the first target index
            if initial:
                start_idx = initial[0].index
            else:
                start_idx = -1
            revised_paras: list[Paragraph] = []
            if start_idx >= 0:
                for offset, txt in enumerate(out.revised_paragraphs):
                    revised_paras.append(Paragraph(index=start_idx + offset, text=txt))
            # logging.getLogger(__name__).debug("Revised paragraphs for %s: %d", ins.target_section, len(revised_paras))
            revised_sections.append(
                RevisedSection(
                    amendment_start_line=ins.amendment_start_line,
                    amendment_end_line=ins.amendment_end_line,
                    amendment_span_text=ins.amendment_span_text,
                    target_section=ins.target_section,
                    confidence_target=ins.confidence_target,
                    change_explanation=ins.change_explanation,
                    target_paragraph_indices=ins.target_paragraph_indices,
                    confidence_target_paragraph_indices=ins.confidence_target_paragraph_indices,
                    target_paragraph_explanation=ins.target_paragraph_explanation,
                    initial_paragraphs=initial if initial else None,
                    revised_paragraphs=revised_paras if revised_paras else None,
                    confidence_revision=out.confidence_revision,
                    revision_explanation=out.revision_explanation,
                )
            )

        return revised_sections

    def generate_amended_and_restated(
        self,
        contract: Sequence[Paragraph],
        amendment: Sequence[Paragraph],
        template: ContractTypeTemplate,
    ) -> RevisedContract:
        """End-to-end helper: plan and apply revisions to output final contract."""
        # 1) Analyze amendment to produce high-level instructions
        instructions = self.analyze_amendments(amendment_paragraphs=amendment, template=template)

        # 2) Locate target spans in the base contract
        targeted = self.find_revisions_targets(
            contracts_paragraphs=contract, instructions=instructions
        )

        # 3) Apply revisions to produce revised sections
        revised_sections = self.apply_revisions(section_paragraphs=contract, instructions=targeted)

        # 4) Materialize new contract content by replacing each targeted span with revised paragraphs
        new_content_list: list[Paragraph] = list(contract)

        # Build list of applicable changes with ranges
        changes = []
        for rs in revised_sections:
            if not rs.target_paragraph_indices or not rs.revised_paragraphs:
                continue
            start = min(rs.target_paragraph_indices)
            end = max(rs.target_paragraph_indices)
            changes.append((start, end, rs.revised_paragraphs))

        # Apply changes from bottom to top to preserve indices
        changes.sort(key=lambda x: x[0], reverse=True)
        for start, end, revised in changes:
            # Guard against out-of-bounds
            if start < 0 or end >= len(new_content_list) or end < start:
                continue
            new_content_list[start : end + 1] = revised

        # Re-index final paragraphs
        reindexed: list[Paragraph] = [
            Paragraph(index=i, text=p.text) for i, p in enumerate(new_content_list)
        ]

        return RevisedContract(new_content=reindexed, applied_instructions=revised_sections)
