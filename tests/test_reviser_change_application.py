from contract_ai_core.reviser import ContractReviser
from contract_ai_core.schema import (
    ClauseDefinition,
    ContractTypeTemplate,
    DatapointDefinition,
    Paragraph,
)


def test_materialize_changes_bottom_up_ordering():
    # Minimal template
    template = ContractTypeTemplate(
        key="T",
        name="T",
        use_case="amendment",
        description="Test",
        clauses=[ClauseDefinition(key="a", title="A")],
        datapoints=[DatapointDefinition(key="x", title="X")],
        enums=[],
    )
    # Base contract paragraphs
    contract = [Paragraph(index=i, text=f"P{i}") for i in range(5)]
    # Fake revised sections simulating two changes
    from contract_ai_core.schema import RevisedSection
    r1 = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s1",
        confidence_target=1.0,
        change_explanation="",
        target_paragraph_indices=[1, 2],
        confidence_target_paragraph_indices=1.0,
        target_paragraph_explanation="",
        initial_paragraphs=[contract[1], contract[2]],
        revised_paragraphs=[Paragraph(index=1, text="A"), Paragraph(index=2, text="B")],
        confidence_revision=1.0,
        revision_explanation="",
    )
    r2 = RevisedSection(
        amendment_start_line=0,
        amendment_end_line=0,
        amendment_span_text="",
        target_section="s2",
        confidence_target=1.0,
        change_explanation="",
        target_paragraph_indices=[3, 4],
        confidence_target_paragraph_indices=1.0,
        target_paragraph_explanation="",
        initial_paragraphs=[contract[3], contract[4]],
        revised_paragraphs=[Paragraph(index=3, text="C"), Paragraph(index=4, text="D")],
        confidence_revision=1.0,
        revision_explanation="",
    )

    # Monkeypatch ContractReviser steps to return our fake revised_sections
    cr = ContractReviser()
    cr.analyze_amendments = lambda amendment_paragraphs, template: []  # type: ignore
    cr.find_revisions_targets = lambda contracts_paragraphs, instructions: []  # type: ignore
    cr.apply_revisions = lambda section_paragraphs, instructions: [r1, r2]  # type: ignore

    res = cr.generate_amended_and_restated(contract=contract, amendment=[], template=template)
    texts = [p.text for p in res.new_content]
    assert texts == ["P0", "A", "B", "C", "D"]


