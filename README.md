## contract-ai-core

Core library for AI-driven legal contract processing. This package provides the building blocks (no UI or APIs) to:

- Classify contract paragraphs into clause categories
- Extract structured datapoints from contracts
- Generate an amended and restated contract using a contract type template

The repository is scaffolded with interfaces and types, ready for concrete LLM-backed implementations.

Status: initial scaffold (no implementations).

### Highlights

- **Template-driven**: Contract type templates define expected clauses and datapoints
- **Composable pipeline**: Classifier → Extractor → Reviser
- **LLM-ready**: Designed to plug in your preferred LLM provider and prompt strategy

### Project layout

```
contract-ai-core/
  docs/
  src/
    contract_core_ai/
      __init__.py
      classifier.py
      extractor.py
      reviser.py
      schema.py
  tests/
  README.md
  requirements.txt
  .env.example
```

Note: The Python package is located under `src/contract_core_ai` and is importable as `contract_core_ai`.

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
- `classifier.py`: Interfaces for paragraph and document clause classification
- `extractor.py`: Interfaces for extracting datapoints (using template and optionally classifications)
- `reviser.py`: Interfaces for proposing and applying revisions to produce an amended and restated contract

### Usage sketch (to be implemented by you)

```python
from pathlib import Path

from contract_core_ai import (
    ContractTypeTemplate, ClauseDefinition, DatapointDefinition,
    ClauseClassifier, DatapointExtractor, ContractReviser,
)

# 1) Build or load a ContractTypeTemplate (see schema.py)
# template = ContractTypeTemplate(...)

# 2) Classify
# classifier = ClauseClassifier(model="gpt-4o")
# classified = classifier.classify_document(text=Path("contract.txt").read_text(encoding="utf-8"), template=template)

# 3) Extract datapoints
# extractor = DatapointExtractor(model="gpt-4o")
# datapoints = extractor.extract(text, template=template, classified_clauses=classified)

# 4) Generate amended and restated contract
# reviser = ContractReviser(model="gpt-4o")
# revised = reviser.generate_amended_and_restated(original_text=text, template=template, datapoints=datapoints)
# print(revised.content)
```

All public methods are currently stubs (`NotImplementedError`) and are intended as extension points for your implementations.

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


