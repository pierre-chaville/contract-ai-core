## Dataset Tools

Location: `dataset/*.py`

### Classification

- `run_classification.py` – Classifies paragraphs into clauses for a given template.

Usage:
```bash
python dataset/run_classification.py --template ISDA --model gpt-4.1
```

Metrics:
- `run_classification_metrics.py` – Computes accuracy/F1 per clause based on gold labels.

### Extraction

- `run_extraction.py` – Extracts datapoints based on the template definitions.

Usage:
```bash
python dataset/run_extraction.py --template ISDA --model gpt-4.1-mini
```

Metrics and reporting:
- `run_extraction_metrics.py` – Computes relaxed accuracy, per-key accuracy and F1. Handles types: str, bool, int, float, date, enum, money.

Outputs are written under `dataset/output/datapoints/<template>/<model>/`.

### Amendments (Reviser)

- `run_reviser.py` – Generates amended-and-restated contracts from pairs of base contract and amendment documents.
  - Writes restated contracts to `dataset/output/amendments/restated/<model>/*.md`.
  - Writes applied instruction JSON to `dataset/output/amendments/instructions/<model>/*.json`.
  - Each instruction JSON row includes: amendment span text, target section, confidence, initial and revised paragraphs, and explanations.

Viewer:
- `view_reviser.py` – Streamlit app to browse instruction JSONs and restated text.

Run the viewer:
```bash
streamlit run dataset/view_reviser.py
```
