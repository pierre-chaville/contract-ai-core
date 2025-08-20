from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .schema import (
    ContractTypeTemplate,
    ExtractionResult,
    RevisionInstruction,
    RevisedContract,
)


@dataclass
class ContractReviserConfig:
    """Configuration for the contract reviser backend."""

    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None


class ContractReviser:
    """Produces an amended and restated contract according to a template."""

    def __init__(self, config: Optional[ContractReviserConfig] = None) -> None:
        self.config = config or ContractReviserConfig()

    def plan_revisions(
        self,
        original_text: str,
        template: ContractTypeTemplate,
        datapoints: ExtractionResult,
    ) -> Sequence[RevisionInstruction]:
        """Return a sequence of high-level revision instructions.

        For example, add missing required clauses, normalize terms, and update
        sections with extracted datapoints.
        """
        raise NotImplementedError("plan_revisions is not implemented yet")

    def apply_revisions(
        self,
        original_text: str,
        instructions: Sequence[RevisionInstruction],
    ) -> RevisedContract:
        """Apply the revision instructions to produce the revised content."""
        raise NotImplementedError("apply_revisions is not implemented yet")

    def generate_amended_and_restated(
        self,
        original_text: str,
        template: ContractTypeTemplate,
        datapoints: ExtractionResult,
    ) -> RevisedContract:
        """End-to-end helper: plan and apply revisions to output final contract."""
        raise NotImplementedError(
            "generate_amended_and_restated is not implemented yet"
        )


