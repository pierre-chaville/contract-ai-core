## Core Library

Location: `src/contract_ai_core/`

### Overview

The core provides three primary components:

- `ClauseClassifier` (`classifier.py`): classifies paragraphs into clause categories defined by a `ContractTypeTemplate`.
- `DatapointExtractor` (`extractor.py`): extracts typed datapoints (str/bool/int/float/date/enum) from contract text.
- `ContractReviser` (`reviser.py`): analyzes an amendment and produces an amended-and-restated version of the base contract with structured instructions.

Documents should be Markdown (`.md`) or plain text (`.txt`). Use `split_text_into_paragraphs(text)` from `schema.py` to normalize text into paragraph units.

### Data models (schema)

Defined in `schema.py`:
- `Paragraph`: `{ index: int, text: str }` paragraph representation.
- `ContractTypeTemplate`: contains clause definitions and datapoint definitions.
- `DocumentClassification`: paragraph classifications and optional `clause_to_paragraphs` map.
- `ExtractedDatapoint`: `{ key, value, confidence, explanation, evidence_paragraph_indices }`.
- Revision models: `RevisedContract`, `RevisedSection`, etc., for the reviser pipeline.

### ClauseClassifier

Usage:

```python
from contract_ai_core.classifier import ClauseClassifier
from contract_ai_core.schema import split_text_into_paragraphs, ContractTypeTemplate

paragraphs = split_text_into_paragraphs(text)
template = ContractTypeTemplate.model_validate(load_template_json())

clf = ClauseClassifier()
doc_cls = clf.classify_document(paragraphs, template)
```

Returns a `DocumentClassification` with per-paragraph labels and optional index map.

### DatapointExtractor

Typed extraction with concurrency and rate-limit resiliency.

Key features:
- Groups datapoints by scope (clauses/beginning/document).
- Merges empty clause scopes into one document scope to avoid duplicate large calls.
- Concurrency configurable via `DatapointExtractorConfig.concurrency`.
- Exponential backoff/retries via `DatapointExtractorConfig.max_retries`.
- Value types follow `data_type` in the template: `str`, `bool`, `int`, `float`, `date`, `enum`.

Usage:

```python
from contract_ai_core.extractor import DatapointExtractor, DatapointExtractorConfig

ext = DatapointExtractor(DatapointExtractorConfig(model="gpt-4.1-mini", concurrency=2, max_retries=5))
result = ext.extract(paragraphs, template, classified_clauses=doc_cls)

for dp in result.datapoints:
    print(dp.key, dp.value, dp.confidence)
```

### ContractReviser

Pipeline: analyze amendments → find targets in base contract → apply revisions → produce restated content plus structured instructions.

Usage:

```python
from contract_ai_core.reviser import ContractReviser, ContractReviserConfig
from contract_ai_core.schema import split_text_into_paragraphs

contract_paras = split_text_into_paragraphs(contract_text)
amendment_paras = split_text_into_paragraphs(amendment_text)

reviser = ContractReviser(ContractReviserConfig(model="gpt-4.1-mini"))
revised = reviser.generate_amended_and_restated(contract=contract_paras, amendment=amendment_paras, template=template)

print(len(revised.new_content), "paragraphs in restated contract")
for r in revised.applied_instructions:
    print(r.target_section, r.change_explanation)
```
