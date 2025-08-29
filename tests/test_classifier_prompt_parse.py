from contract_ai_core.classifier import ClauseClassifier


def test_parse_llm_output_tolerant_without_confidence():
    cc = ClauseClassifier()
    raw = "1: 2\n2: 0\n3: 5"
    parsed = cc._parse_llm_output(raw)
    assert parsed[1] == (2, None)
    assert parsed[2] == (0, None)
    assert parsed[3] == (5, None)


def test_parse_llm_output_with_confidence():
    cc = ClauseClassifier()
    raw = "1: 3 | 95%\n2: 0 | 5%"
    parsed = cc._parse_llm_output(raw)
    assert parsed[1] == (3, 95)
    assert parsed[2] == (0, 5)


