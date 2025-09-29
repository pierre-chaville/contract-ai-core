from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Optional

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
    sort_order: int | None = Field(
        default=None,
        description="Optional ordering hint for display; lower sorts first.",
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
    required: bool = Field(
        default=False,
        description="Whether this datapoint is required for this contract type.",
    )
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
    clause_keys: Sequence[str] | None = Field(
        default=None,
        description="Clause key(s) where this datapoint is typically located.",
    )
    sort_order: int | None = Field(
        default=None,
        description="Optional ordering hint for display; lower sorts first.",
    )


class StructureDefinition(FrozenBaseModel):
    """Definition of a structure used to extract a datapoint from a contract."""

    structure_key: str = Field(
        ..., description="Stable key identifier for the structure (e.g., 'rating_trigger')."
    )
    title: str = Field(..., description="Human-readable structure name (e.g., 'Effective Date').")
    description: str | None = Field(
        default=None,
        description="Optional guidance on how to interpret or extract this structure.",
    )
    elements: Sequence[StructureElementDefinition] = Field(
        ..., description="List of elements of the structure."
    )


class StructureElementDefinition(FrozenBaseModel):
    """Definition of an element of a structure used to extract a datapoint from a contract."""

    structure_key: str = Field(
        ..., description="Stable key identifier for the structure (e.g., 'rating_trigger')."
    )
    key: str = Field(
        ..., description="Stable key identifier for the element (e.g., 'effective_date')."
    )
    title: str = Field(..., description="Human-readable element name (e.g., 'Effective Date').")
    description: str | None = Field(
        default=None,
        description="Optional guidance on how to interpret or extract this element.",
    )
    data_type: str = Field(
        default="string",
        description="Logical data type: e.g., 'str', 'int', 'date', 'float', 'enum', 'list[str]', 'list[int]', 'list[date]', 'list[float]', 'list[enum]'.",
    )
    required: bool = Field(
        default=False,
        description="Whether this element is required for this contract type.",
    )
    # Optional enum constraints for structured outputs
    enum_key: str | None = Field(
        default=None,
        description=(
            "If set, references a named enum in template.enums. When present, outputs should use "
            "one of the enum option codes."
        ),
    )
    sort_order: int | None = Field(
        default=None,
        description="Optional ordering hint for display; lower sorts first.",
    )


class GuidelineDefinition(FrozenBaseModel):
    """Definition of a guideline for a contract type."""

    key: str = Field(
        ..., description="Stable key identifier for the guideline (e.g., 'guideline_1')."
    )
    fallback_from_key: Optional[str] = Field(
        default=None,
        description="If not empty, this guideline is a fallback from the given key.",
    )
    guideline: str = Field(
        ..., description="Human-readable guideline description (e.g., 'Guideline 1 description')."
    )
    action: str = Field(
        ...,
        description="Action to be taken when the guideline is not matched (e.g., 'request approval from Risk Management').",
    )
    priority: str = Field(
        default="medium",
        description="Priority of the guideline. Higher priority guidelines are preferred when multiple are applicable.",
    )
    scope: ExtractionScope = Field(
        default=ExtractionScope.CLAUSE,
        description="Where in the document the guideline is typically located.",
    )
    clause_keys: Sequence[str] | None = Field(
        default=None,
        description="Clause key(s) where this guideline is typically located.",
    )
    sort_order: int | None = Field(
        default=None,
        description="Optional ordering hint for display; lower sorts first.",
    )


class FilteringScope(FrozenBaseModel):
    """A named filtering scope used to limit which parts of a contract are considered.

    For example, for a CSA you might define scopes like 'preamble' for the
    beginning with parties and date, and 'from_paragraph_11' for everything
    from paragraph 11 to the end.
    """

    name: str = Field(
        ..., description="Short, human-readable name of the scope (e.g., 'preamble')."
    )
    description: str = Field(
        ..., description="Detailed description of what to include/exclude within this scope."
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
    filtering_scopes: Sequence[FilteringScope] = Field(
        ...,
        description="List of named filtering scopes (each with name and description).",
    )
    prompt_scope_amendment: str = Field(
        ...,
        description="This is the prompt scope filter for the amendment.",
    )
    clauses: Sequence[ClauseDefinition] = Field(
        ..., description="List of clause definitions expected in this contract type."
    )
    datapoints: Sequence[DatapointDefinition] = Field(
        ..., description="List of datapoint definitions to extract for this contract type."
    )
    structures: Sequence[StructureDefinition] = Field(
        ..., description="List of structure definitions to extract for this contract type."
    )
    guidelines: Sequence[GuidelineDefinition] = Field(
        ..., description="List of guideline definitions for this contract type."
    )
    enums: Optional[Sequence[EnumDefinition]] = Field(
        default=None,
        description=("Rreusable enum definitions that datapoints can reference via 'enum_key'."),
    )


class ContractMetadata(FrozenBaseModel):
    """Descriptive and operational metadata for a legal contract record."""

    # Core Identification
    contract_id: str = Field(..., description="Unique identifier (primary key).")
    contract_number: str | None = Field(
        default=None, description="Business/legal reference number."
    )
    title: str | None = Field(default=None, description="Contract title or name.")
    contract_type: str | None = Field(
        default=None, description="Category (e.g., employment, vendor, lease, NDA)."
    )
    contract_type_version: str | None = Field(
        default=None, description="Version of contract type (e.g., 1992 or 2002 for ISDA)."
    )
    contract_date: str | None = Field(default=None, description="Primary contract date.")
    last_amendment_date: str | None = Field(
        default=None, description="Date of last amendment if applicable, else null."
    )
    number_amendments: int | None = Field(default=None, description="Number of amendments.")
    status: str | None = Field(default=None, description="Status: draft, executed, or signed.")
    party_name_1: str | None = Field(default=None, description="Name of Party 1 (Party A).")
    party_role_1: str | None = Field(default=None, description="Role of Party 1 (e.g., Party A).")
    party_name_2: str | None = Field(default=None, description="Name of Party 2 (Party B).")
    party_role_2: str | None = Field(default=None, description="Role of Party 2 (e.g., Party B).")

    # Business Context
    department: str | None = Field(default=None, description="Owning business unit.")
    contract_owner: str | None = Field(default=None, description="Responsible employee.")
    business_purpose: str | None = Field(
        default=None, description="Brief description of contract purpose."
    )


class LookupValue(FrozenBaseModel):
    """A generic lookup value record (for enums, statuses, vocabularies)."""

    category: str = Field(..., description="Lookup category/group (e.g., 'status', 'law_codes').")
    key: str = Field(..., description="Stable key or code for this value.")
    label: str = Field(..., description="Human-readable display label.")
    description: str | None = Field(
        default=None, description="Optional longer description/tooltip."
    )
    sort_order: int | None = Field(
        default=None, description="Optional ordering hint (lower sorts first)."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Arbitrary extra attributes for this value."
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


class ReviewedGuideline(FrozenBaseModel):
    """Value reviewed for a guideline along with optional provenance info."""

    key: str = Field(..., description="Guideline key (matches a definition in the template).")
    guideline_matched: bool = Field(..., description="Whether the guideline was matched.")
    confidence: float | None = Field(
        default=None,
        description="Confidence score in [0,1] for the guideline match, if available.",
    )
    explanation: str | None = Field(
        default=None,
        description="Short explanation or rationale for the extracted guideline value.",
    )
    evidence_paragraph_indices: Sequence[int] | None = Field(
        default=None,
        description="Paragraph indices that provide evidence for this guideline review.",
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


class DocumentAnalysis(FrozenBaseModel):
    """Aggregate view of a processed contract document.

    Combines high-level metadata with extracted artifacts produced by the pipeline:
    - Classified clauses at the paragraph level
    - Extracted datapoints (values + evidence)
    - Reviewed guidelines (compliance checks)
    """

    metadata: ContractMetadata | None = Field(
        default=None,
        description="Descriptive metadata for the contract (id, title, parties, dates, etc.).",
    )
    classified_clauses: DocumentClassification | None = Field(
        default=None,
        description="Per-paragraph clause classifications and optional clause index.",
    )
    extracted_datapoints: ExtractionResult | None = Field(
        default=None,
        description="All datapoints extracted from the document, including evidence and confidence.",
    )
    reviewed_guidelines: Sequence[ReviewedGuideline] | None = Field(
        default=None,
        description="Results of guideline reviews with match status, evidence, and confidence.",
    )
