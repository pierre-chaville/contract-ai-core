from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FrozenBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ClauseDefinition(FrozenBaseModel):
    """Definition of a clause expected in a given contract type template."""

    key: str = Field(..., description="Stable key identifier for the clause (e.g., 'termination').")
    title: str = Field(..., description="Human-readable clause title (e.g., 'Termination').")
    description: str | None = Field(
        default=None,
        description="Optional explanation of the clause's scope and intent.",
    )
    # required: bool = Field(
    #     default=True,
    #     description="Whether this clause is required for this contract type.",
    # )


class EnumOption(FrozenBaseModel):
    """One permissible value for an enum, identified by a stable code and a description."""

    code: str = Field(
        ..., description="Stable enum code to be used in outputs (e.g., 'NY', 'USD')."
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description or definition for this code.",
    )


class EnumDefinition(FrozenBaseModel):
    """Reusable enumeration definition that can be referenced by datapoints via 'enum_key'."""

    key: str = Field(
        ..., description="Stable key for this enum list (e.g., 'governing_law_codes')."
    )
    title: str | None = Field(
        default=None,
        description="Optional title for this enum list.",
    )
    # description: Optional[str] = Field(
    #     default=None,
    #     description="Optional description of the enum's intended use.",
    # )
    options: Sequence[EnumOption] = Field(
        ..., description="List of permissible enum options (code and description)."
    )


class ExtractionScope(str, Enum):
    """Defines where in the document the datapoint was extracted from"""

    CLAUSE = "clause"
    DOCUMENT = "document"
    BEGINNING = "beginning"


class DatapointDefinition(FrozenBaseModel):
    """Definition of a datapoint to be extracted from a contract."""

    key: str = Field(
        ..., description="Stable key identifier for the datapoint (e.g., 'effective_date')."
    )
    title: str = Field(..., description="Human-readable datapoint name (e.g., 'Effective Date').")
    description: str | None = Field(
        default=None,
        description="Optional guidance on how to interpret or extract this datapoint.",
    )
    data_type: str = Field(
        default="string",
        description="Logical data type: e.g., 'string', 'number', 'date', 'party', 'money'.",
    )
    # required: bool = Field(
    #     default=False,
    #     description="Whether this datapoint is required for this contract type.",
    # )
    scope: ExtractionScope = Field(
        default=ExtractionScope.CLAUSE,
        description="Where in the document the datapoint was extracted from.",
    )
    # Optional enum constraints for structured outputs
    enum_key: str | None = Field(
        default=None,
        description=(
            "If set, references a named enum in template.enums. When present, outputs should use "
            "one of the enum option codes."
        ),
    )
    enum_multi_select: bool = Field(
        default=False,
        description="Whether multiple enum codes can be selected for this datapoint.",
    )
    clause_keys: Sequence[str] | None = Field(
        default=None,
        description="Clause key(s) where this datapoint is typically located.",
    )


class ContractTypeTemplate(FrozenBaseModel):
    """Template describing expected clauses and datapoints for a contract type."""

    key: str = Field(..., description="Stable key identifier for the contract type (e.g., 'NDA').")
    name: str = Field(..., description="Template name (e.g., 'NDA', 'Loan Agreement').")
    use_case: str = Field(
        ...,
        description="Use case for the contract type (e.g., 'extraction of clauses', 'amendment of clauses').",
    )
    description: str | None = Field(
        ..., description="Optional description of the contract type template."
    )
    clauses: Sequence[ClauseDefinition] = Field(
        ..., description="List of clause definitions expected in this contract type."
    )
    datapoints: Sequence[DatapointDefinition] = Field(
        ..., description="List of datapoint definitions to extract for this contract type."
    )
    enums: Sequence[EnumDefinition] = Field(
        default=None,
        description=("Rreusable enum definitions that datapoints can reference via 'enum_key'."),
    )


class Paragraph(FrozenBaseModel):
    """A paragraph of the contract with an index for stable referencing."""

    index: int = Field(..., description="Zero-based index of the paragraph within the document.")
    text: str = Field(..., description="Raw paragraph text.")


class ClassifiedParagraph(FrozenBaseModel):
    """Classification result for a single paragraph."""

    paragraph: Paragraph = Field(..., description="The paragraph being classified.")
    clause_key: str | None = Field(
        default=None,
        description="Predicted clause key or None if unclassified/other.",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score in [0,1] for the classification, if available.",
    )


class DocumentClassification(FrozenBaseModel):
    """Aggregate classification results for a document."""

    paragraphs: Sequence[ClassifiedParagraph] = Field(
        ..., description="Per-paragraph classifications across the document."
    )
    # Optional map from clause_key to paragraph indices for quick lookup
    clause_to_paragraphs: dict[str, list[int]] | None = Field(
        default=None,
        description="Optional index for quick retrieval of paragraphs by clause key.",
    )


class ExtractedDatapoint(FrozenBaseModel):
    """Value extracted for a datapoint along with optional provenance info."""

    key: str = Field(..., description="Datapoint key (matches a definition in the template).")
    value: Any = Field(
        default=None,
        description="Extracted value; type depends on datapoint data_type (str/bool/int/float).",
    )
    explanation: str | None = Field(
        default=None,
        description="Short explanation or rationale for the extracted value.",
    )
    # Indices of paragraphs that support/justify the extracted value
    evidence_paragraph_indices: Sequence[int] | None = Field(
        default=None,
        description="Paragraph indices that provide evidence for this value.",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score in [0,1] for the extraction, if available.",
    )


class ExtractionResult(FrozenBaseModel):
    """Aggregate datapoint extraction results."""

    datapoints: Sequence[ExtractedDatapoint] = Field(
        ..., description="All extracted datapoints for the document."
    )


class RevisionInstruction(FrozenBaseModel):
    """Instruction for revising content: target section."""

    amendment_start_line: int = Field(
        ..., description="The line number where the amendment starts."
    )
    amendment_end_line: int = Field(..., description="The line number where the amendment ends.")
    amendment_span_text: str = Field(..., description="Text of the amendment span")
    target_section: str = Field(
        ..., description="Section to revise in the contract, e.g., 'Part 1 (a)(ii)."
    )
    confidence_target: float = Field(
        ..., description="Confidence score in [0,1] for the target section."
    )
    change_explanation: str = Field(..., description="Explanation of the intended change.")


class RevisionInstructionTarget(RevisionInstruction):
    """Instruction for revising content: target paragraph indices."""

    target_paragraph_indices: Sequence[int] | None = Field(
        default=None,
        description="Target paragraph indices to revise.",
    )
    confidence_target_paragraph_indices: float = Field(
        ..., description="Confidence score in [0,1] for the target paragraph indices."
    )
    target_paragraph_explanation: str = Field(..., description="Explanation of the target section.")


class RevisedSection(RevisionInstructionTarget):
    """Section revised with instructions."""

    initial_paragraphs: Sequence[Paragraph] | None = Field(
        default=None,
        description="Initial paragraphs to revise in the contract.",
    )
    revised_paragraphs: Sequence[Paragraph] | None = Field(
        default=None,
        description="Revised paragraphs in the contract.",
    )
    confidence_revision: float = Field(
        ..., description="Confidence score in [0,1] for the revision."
    )
    revision_explanation: str = Field(..., description="Explanation of the revision.")


class RevisedContract(FrozenBaseModel):
    """The resulting amended and restated contract content with metadata."""

    new_content: Sequence[Paragraph] = Field(
        ..., description="Final amended and restated contract text as paragraphs."
    )
    applied_instructions: Sequence[RevisedSection] = Field(
        ..., description="List of revised sections that were applied."
    )


def split_text_into_paragraphs(text: str) -> list[Paragraph]:
    """Split text into paragraphs with markdown cleanup and heuristic merging.

    Cleanup rules:
    - Collapse multiple consecutive empty lines into a single empty line
    - If a line contains '|' but neither the previous nor next line contains '|',
      replace '|' with a space (break stray table artifacts)
    - If a line contains only '|' and '-' characters and neither neighbor line
      contains '|', drop the line (remove markdown table separator rows)

    Paragraph merge rules (merge current line into the previous paragraph only if):
    - The previous line does not end with one of . : ! ?
    - The current line does not start with a list/enumeration marker like
      '-', '1)', '1.', 'a)', '(a)', '(1)', etc.
    - Neither the previous line nor the current line contains a '|'
    """

    if not text:
        return []

    # Normalize newlines and split
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cleaned_lines: list[str] = []
    n = len(raw_lines)
    only_pipes_dashes_re = re.compile(r"^[\s\|\-]+$")
    only_stars_re = re.compile(r"^[\s\*]+$")
    star_rule_re = re.compile(r"^\*+(?:\s*\*+){2,}$")

    def remove_md_emphasis(s: str) -> str:
        # Remove strong/italic/strike emphasis markers: **text**, *text*, __text__, _text_, ~~text~~
        s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
        s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)
        s = re.sub(r"~~(.*?)~~", r"\1", s)
        return s

    i = 0
    while i < n:
        line = raw_lines[i]
        stripped = line.strip()

        # Remove markdown emphasis markers inline
        line = remove_md_emphasis(line)

        # Remove lines that are only stars (e.g., markdown separators) or star rules like * * *
        if stripped and (only_stars_re.match(stripped) or star_rule_re.match(stripped)):
            i += 1
            continue

        # Remove any line consisting solely of '|' and '-' (table separators), unconditionally
        if stripped and only_pipes_dashes_re.match(stripped):
            i += 1
            continue

        # Convert all pipes to tabs to normalize table-like content into columns
        if "|" in line:
            line = line.replace("|", "\t")

        cleaned_lines.append(line)
        i += 1

    # Collapse multiple consecutive empty lines into a single empty line
    collapsed_lines: list[str] = []
    empty_streak = 0
    for line in cleaned_lines:
        if line.strip() == "":
            empty_streak += 1
            if empty_streak > 1:
                continue
        else:
            empty_streak = 0
        collapsed_lines.append(line)

    # Merge lines into paragraphs using heuristics
    paragraphs: list[str] = []
    buffer = ""
    prev_line_text: str | None = None

    list_marker_re = re.compile(r"^\s*(?:-+|\d+\)|\d+\.|[A-Za-z]\)|\([A-Za-z]\)|\(\d+\))\s+")

    for line in collapsed_lines:
        stripped = line.strip()
        if stripped == "":
            if buffer:
                paragraphs.append(buffer.strip())
                buffer = ""
                prev_line_text = None
            continue

        if not buffer:
            buffer = stripped
            prev_line_text = line
            continue

        # Decide merge vs start new paragraph
        assert prev_line_text is not None
        prev_last_char = buffer.rstrip()[-1] if buffer.rstrip() else ""
        prev_ends_sentence = prev_last_char in ".:!?"
        curr_starts_list = bool(list_marker_re.match(stripped))
        prev_has_pipe_now = "|" in prev_line_text
        curr_has_pipe_now = "|" in line

        can_merge = (
            (not prev_ends_sentence)
            and (not curr_starts_list)
            and (not prev_has_pipe_now)
            and (not curr_has_pipe_now)
        )

        if can_merge:
            buffer = f"{buffer.rstrip()} {stripped}"
        else:
            paragraphs.append(buffer.strip())
            buffer = stripped

        prev_line_text = line

    if buffer:
        paragraphs.append(buffer.strip())

    return [Paragraph(index=i, text=block) for i, block in enumerate(paragraphs) if block]
