import pytest
from contract_ai_core.classifier import ClauseClassifier
from contract_ai_core.schema import ClauseDefinition, ContractTypeTemplate, Paragraph


def test_classify_paragraphs_with_usage_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Template with two clauses
    template = ContractTypeTemplate(
        key="NDA",
        name="NDA",
        use_case="classification",
        description="Non-Disclosure Agreement",
        prompt_scope_filter="",
        prompt_scope_amendment="",
        clauses=[
            ClauseDefinition(key="termination", title="Termination"),
            ClauseDefinition(key="confidentiality", title="Confidentiality"),
        ],
        datapoints=[],
        guidelines=[],
        enums=None,
    )

    # Three paragraphs
    paragraphs = [
        Paragraph(index=0, text="This agreement may be terminated by either party."),
        Paragraph(index=1, text="Miscellaneous content that doesn't match any clause."),
        Paragraph(index=2, text="The parties shall keep information confidential."),
    ]

    # Fake LLM output: map paragraph 0 -> clause 1, paragraph 1 -> none (0), paragraph 2 -> clause 2
    raw_output = "0: 1 | 90%\n1: 0 | 10%\n2: 2 | 80%"
    usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}

    # Patch the backend call to avoid real network
    monkeypatch.setattr(ClauseClassifier, "_call_llm", lambda self, prompt: (raw_output, usage))

    cc = ClauseClassifier()
    doc, got_usage = cc.classify_paragraphs_with_usage(paragraphs, template)

    # Usage is forwarded
    assert got_usage == usage

    # We get one classification per paragraph
    assert len(doc.paragraphs) == 3

    # Paragraph 0 -> termination (id=1), confidence 0.90
    assert doc.paragraphs[0].paragraph.index == 0
    assert doc.paragraphs[0].clause_key == "termination"
    assert doc.paragraphs[0].confidence == 0.90

    # Paragraph 1 -> none (id=0), confidence still normalized
    assert doc.paragraphs[1].paragraph.index == 1
    assert doc.paragraphs[1].clause_key is None
    assert doc.paragraphs[1].confidence == 0.10

    # Paragraph 2 -> confidentiality (id=2)
    assert doc.paragraphs[2].paragraph.index == 2
    assert doc.paragraphs[2].clause_key == "confidentiality"
    assert doc.paragraphs[2].confidence == 0.80

    # clause_to_paragraphs is correctly populated
    assert doc.clause_to_paragraphs == {
        "termination": [0],
        "confidentiality": [2],
    }
