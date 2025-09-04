from contract_ai_core.reviser import ContractReviser
from contract_ai_core.schema import (
    ClauseDefinition,
    ContractTypeTemplate,
    DatapointDefinition,
    Paragraph,
    RevisedSection,
)


def test_generate_amended_and_restated_applies_valid_changes_and_skips_invalid() -> None:
    # Minimal template
    template = ContractTypeTemplate(
        key="T",
        name="T",
        use_case="amendment",
        description="Test",
        prompt_scope_filter="",
        prompt_scope_amendment="",
        clauses=[ClauseDefinition(key="a", title="A")],
        datapoints=[DatapointDefinition(key="x", title="X")],
        guidelines=[],
        enums=[],
    )

    # Base contract paragraphs
    contract = [Paragraph(index=i, text=f"P{i}") for i in range(5)]

    # Valid change replacing indices [1,2]
    r_valid = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s-valid",
        confidence_target=1.0,
        change_explanation="",
        target_paragraph_indices=[1, 2],
        confidence_target_paragraph_indices=1.0,
        target_paragraph_explanation="",
        initial_paragraphs=[contract[1], contract[2]],
        revised_paragraphs=[Paragraph(index=1, text="X"), Paragraph(index=2, text="Y")],
        confidence_revision=1.0,
        revision_explanation="",
    )

    # Change with empty target indices -> should be ignored
    r_empty_target = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s-empty",
        confidence_target=0.5,
        change_explanation="",
        target_paragraph_indices=[],
        confidence_target_paragraph_indices=0.1,
        target_paragraph_explanation="",
        initial_paragraphs=None,
        revised_paragraphs=[Paragraph(index=0, text="Z")],
        confidence_revision=0.1,
        revision_explanation="",
    )

    # Change with missing revised paragraphs -> should be ignored
    r_missing_revised = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s-missing",
        confidence_target=0.5,
        change_explanation="",
        target_paragraph_indices=[3, 4],
        confidence_target_paragraph_indices=0.2,
        target_paragraph_explanation="",
        initial_paragraphs=[contract[3], contract[4]],
        revised_paragraphs=None,
        confidence_revision=0.2,
        revision_explanation="",
    )

    # Out-of-bounds change -> ignored by bounds guard
    r_oob = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s-oob",
        confidence_target=0.5,
        change_explanation="",
        target_paragraph_indices=[10, 11],
        confidence_target_paragraph_indices=0.3,
        target_paragraph_explanation="",
        initial_paragraphs=None,
        revised_paragraphs=[Paragraph(index=10, text="OO"), Paragraph(index=11, text="PP")],
        confidence_revision=0.3,
        revision_explanation="",
    )

    cr = ContractReviser()
    # Bypass upstream LLM-dependent stages
    cr.analyze_amendments = lambda amendment_paragraphs, template: []  # type: ignore
    cr.find_revisions_targets = lambda contracts_paragraphs, instructions: []  # type: ignore
    cr.apply_revisions = lambda section_paragraphs, instructions: [  # type: ignore
        r_valid,
        r_empty_target,
        r_missing_revised,
        r_oob,
    ]

    res = cr.generate_amended_and_restated(contract=contract, amendment=[], template=template)

    # Only the valid change should be applied; others are ignored.
    texts = [p.text for p in res.new_content]
    assert texts == ["P0", "X", "Y", "P3", "P4"]

    # Applied instructions are preserved in the result
    assert res.applied_instructions == [r_valid, r_empty_target, r_missing_revised, r_oob]
