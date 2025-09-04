from typing import Any

import pytest
from contract_ai_core.extractor import DatapointExtractor, DatapointExtractorConfig
from contract_ai_core.schema import (
    ClauseDefinition,
    ContractTypeTemplate,
    DatapointDefinition,
    DocumentClassification,
)


def test_extract_with_mocked_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Minimal template with one clause and three datapoints (string/date/enum)
    template = ContractTypeTemplate(
        key="NDA",
        name="NDA",
        use_case="extraction",
        description="Non-Disclosure Agreement",
        prompt_scope_filter="",
        prompt_scope_amendment="",
        clauses=[
            ClauseDefinition(key="termination", title="Termination"),
        ],
        datapoints=[
            DatapointDefinition(
                key="termination_reason",
                title="Termination Reason",
                data_type="string",
                clause_keys=["termination"],
            ),
            DatapointDefinition(key="effective_date", title="Effective Date", data_type="date"),
            DatapointDefinition(
                key="governing_law", title="Governing Law", data_type="enum", enum_key="law_codes"
            ),
        ],
        guidelines=[],
        enums=None,
    )

    paragraphs = [
        # index is position in list
        type("P", (), {"index": 0, "text": "This agreement is made on January 2, 2020."})(),
        type("P", (), {"index": 1, "text": "Either party may terminate upon breach."})(),
        type("P", (), {"index": 2, "text": "All information shall remain confidential."})(),
    ]

    # Classified clauses map the 'termination' clause to paragraph index 1
    classified = DocumentClassification(paragraphs=[], clause_to_paragraphs={"termination": [1]})

    # Prepare deterministic outputs from the backend
    expected_values = {
        "termination_reason": ("Breach", 0.8),
        "effective_date": ("2020-01-02", 0.9),
        "governing_law": ("NY", 1.0),
    }

    async def fake_run_scope_extractions(
        self: Any,
        *,
        jobs: list[dict[str, object]],
        model_name: str,
        temperature: float,
        concurrency: int = 8,
    ) -> list[dict[str, object]]:
        results = []
        for job in jobs:
            datapoints = job["datapoints"]
            evidence = job["evidence"]
            data: dict[str, dict] = {}
            for dp in datapoints:  # type: ignore[attr-defined]
                value, conf = expected_values.get(dp.key, (None, None))  # type: ignore[attr-defined]
                data[dp.key] = {"value": value, "confidence": conf, "explanation": None}
            results.append({"data": data, "datapoints": datapoints, "evidence": evidence})
        return results

    # Patch the concurrent runner to avoid network
    monkeypatch.setattr(DatapointExtractor, "_run_scope_extractions", fake_run_scope_extractions)

    extractor = DatapointExtractor(DatapointExtractorConfig(beginning_paragraphs=2))
    result = extractor.extract(paragraphs, template, classified_clauses=classified)

    # Validate we received values and confidences mapped per datapoint
    by_key = {dp.key: dp for dp in result.datapoints}

    assert by_key["termination_reason"].value == "Breach"
    assert by_key["termination_reason"].confidence == 0.8
    assert by_key["termination_reason"].evidence_paragraph_indices == [1]

    assert by_key["effective_date"].value == "2020-01-02"
    assert by_key["effective_date"].confidence == 0.9
    # Beginning scope uses first K paragraph indices (here 0 and 1)
    assert by_key["effective_date"].evidence_paragraph_indices == [0, 1]

    assert by_key["governing_law"].value == "NY"
    assert by_key["governing_law"].confidence == 1.0
    # Also part of beginning scope evidence
    assert by_key["governing_law"].evidence_paragraph_indices == [0, 1]
