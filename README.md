## contract-ai-core

AI-driven legal contract processing. This repo includes a core Python library and dataset tools to:

- Classify contract paragraphs into clause categories
- Extract structured datapoints from contracts
- Generate an amended-and-restated contract from an amendment

### Highlights

- **Template-driven**: Contract type templates define expected clauses and datapoints
- **Composable pipeline**: Classifier → Extractor → Reviser
- **LLM-ready**: Designed to plug in your preferred LLM provider and prompt strategy

### New: Filtering scopes and metrics

This repo includes a lightweight document filtering pipeline to locate named spans (scopes) in contracts.

- Scopes are defined per contract type in the template as a list of `{name, description}`.
- The filter returns, for each scope, `start_line`, `end_line`, `confidence`, and `explanation`.
- Batch inference calls a single LLM to find all scopes at once for efficiency.

CLI:

```bash
# Run scope detection for organizer files and write JSON outputs
python tools/run_filter.py --model gpt-4.1-mini --provider openai

# Compute filter metrics vs gold and write result.json per type/model
python tools/run_filter_metrics.py --type ISDA --model gpt-4.1-mini
```

Outputs:

- Predictions: `dataset/output/filter/<contract_type>/<model>/<stem>.json`
- Metrics: `dataset/metrics/filter/<contract_type>/<model>/result.json`

Sampling (optional):

```bash
# Build sample files per contract version from gold scopes
python tools/run_sample.py --type ISDA
```

This writes text snippets by version to `dataset/samples/<contract_type>/<contract_version>/<stem>.txt`.

### Project layout

```
contract-ai-core/
  documentation/
  tools/
  dataset/
  src/contract_ai_core/
  tests/
  README.md
  requirements.txt
```

The Python package is importable as `contract_ai_core`.

### Quick start

1) Python 3.10+ is recommended.

2) Create a virtual environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Configure environment variables. Copy `.env.example` to `.env` and set your key(s):

```
OPENAI_API_KEY=your-openai-key
```

You can load environment variables at runtime using `python-dotenv` if desired.

### Library overview

- `schema.py`: Data models for templates, classification results, extracted datapoints, and revision artifacts
- `classifier.py`: Clause classification for paragraphs/documents
- `extractor.py`: Datapoint extraction (typed values, concurrency, retries)
- `reviser.py`: Amendment analysis and restated contract generation

See full docs under `documentation/`:

- `[documentation/index.md](documentation/index.md)`
- `[documentation/core_library.md](documentation/core_library.md)`
- `[documentation/dataset.md](documentation/dataset.md)`
- `[documentation/tools.md](documentation/tools.md)`

### Usage sketch

```python
from contract_ai_core.schema import split_text_into_paragraphs
from contract_ai_core.classifier import ClauseClassifier
from contract_ai_core.extractor import DatapointExtractor, DatapointExtractorConfig
from contract_ai_core.reviser import ContractReviser, ContractReviserConfig

text = open("./dataset/documents/ISDA/example.md", "r", encoding="utf-8").read()
paragraphs = split_text_into_paragraphs(text)

# Classify
clf = ClauseClassifier()
doc_cls = clf.classify_document(paragraphs, template)

# Extract
ext = DatapointExtractor(DatapointExtractorConfig(model="gpt-4.1-mini", concurrency=2))
result = ext.extract(paragraphs, template, classified_clauses=doc_cls)

# Revise
reviser = ContractReviser(ContractReviserConfig(model="gpt-4.1-mini"))
revised = reviser.generate_amended_and_restated(contract=paragraphs, amendment=amend_paragraphs, template=template)
```

Documents should be `.md`/`.txt`. Use `split_text_into_paragraphs` to normalize paragraphs.

### Metrics

Classification metrics:

- `tools/run_classification_metrics.py` writes `summary.yaml`, `confusion_matrix.csv/.png`
- The summary includes `num_examples`, overall `accuracy`, `cohen_kappa`, high-confidence stats,
  `macro_f1`, and a `per_category` list with: `category`, `accuracy`, `count`, `f1`, `precision`, `recall`.

Extraction metrics:

- `tools/run_extraction_metrics.py` writes `summary.yaml` (single consolidated file)
- The summary includes overall scores and a `per_key` list with: `key`, `accuracy`, `count`, `f1`, `precision`, `recall`.
- Object and list-of-object datapoints are flattened to element-level keys:
  - `object:<structure>` → `datapoint_key.element_key`
  - `list[object:<structure>]` → `datapoint_key[<i>].element_key`

Filter metrics:

- `tools/run_filter_metrics.py` aggregates accuracy/IoU per scope and provides histograms and per-scope averages.

### Streamlit pages

The `tools/pages` directory contains Streamlit pages to browse inputs/outputs:

- `50_Metrics.py`: Metrics dashboard with tabs for Migration, Filter, Clauses, and Datapoints
  - Summaries render as field: value lines with percentages; per-category/per-key tables where applicable
- `20_Contracts.py`: Adds a Text tab to show the full contract and an interactive Compare tab
- `15_Contract_filter.py`: Explore filter outputs by type/model with raw text and per-scope spans

### Environment and secrets

Set your LLM credentials (e.g., `OPENAI_API_KEY`) in `.env`. Load them at runtime if needed:

```python
from dotenv import load_dotenv
load_dotenv()  # loads variables from .env
```

### Testing

Add tests under `tests/`. As implementations evolve, include unit tests for:

- Classification behavior against fixture contracts
- Datapoint extraction accuracy and schema validation
- Revision generation: structural and textual assertions

### Development

Install dev tools and enable pre-commit hooks:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
pytest -q
```

### Roadmap ideas

- Prompt templates and guardrails per clause/datapoint
- Caching and idempotency for repeated runs
- Evaluation harness with golden datasets
- Multi-provider LLM adapters
