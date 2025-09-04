## Contract AI Core – Documentation

This repository is organized into three parts:

1. Core library (`src/contract_ai_core/`)
   - Provides the main building blocks to classify clauses, extract datapoints, and generate amended-and-restated contracts.
2. Dataset (`dataset/`)
   - A small curated corpus (from public EDGAR filings) with ~20 ISDA Master Agreements and ~20 amendments.
3. Dataset tools (`dataset/*.py`)
   - Command-line utilities to run the core models on the dataset, compute metrics, and visualize amendment instructions.

Use the links below to dive into each part:

- Core Library: `[core_library.md](core_library.md)`
- Dataset: `[dataset.md](dataset.md)`
- Dataset Tools: `[dataset_tools.md](dataset_tools.md)`

### Quick start

- Documents are expected as Markdown (`.md`) or plain text (`.txt`).
- The helper `split_text_into_paragraphs(text: str)` converts raw `.md/.txt` into normalized paragraphs suitable for the models.
- Example minimal flow:

```python
from contract_ai_core.schema import split_text_into_paragraphs
from contract_ai_core.classifier import ClauseClassifier
from contract_ai_core.extractor import DatapointExtractor
from contract_ai_core.reviser import ContractReviser, ContractReviserConfig

text = open("path/to/contract.md", "r", encoding="utf-8").read()
paragraphs = split_text_into_paragraphs(text)

# 1) Classify clauses
clf = ClauseClassifier()
classification = clf.classify_document(paragraphs, template)

# 2) Extract datapoints
ext = DatapointExtractor()
result = ext.extract(paragraphs, template, classified_clauses=classification)

# 3) Apply an amendment
reviser = ContractReviser(ContractReviserConfig(model="gpt-4.1-mini"))
revised = reviser.generate_amended_and_restated(contract=paragraphs, amendment=amend_paragraphs, template=template)
```
