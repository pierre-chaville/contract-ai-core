def test_can_import_package():
    import importlib

    pkg = importlib.import_module("contract_ai_core")
    assert hasattr(pkg, "ClauseClassifier")
    assert hasattr(pkg, "DatapointExtractor")
    assert hasattr(pkg, "ContractReviser")


