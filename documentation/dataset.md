## Dataset

Location: `dataset/`

### Contents

- `documents/`
  - `contract_types/`: JSON templates describing clauses and datapoints (e.g., `ISDA.json`).
  - `amendments/`: raw Markdown source of amendments and initial documents (see repo structure).
- `gold/`
  - `datapoints/<template_key>/*.csv`: Gold labels for datapoints per document.
- `output/`
  - `datapoints/<template_key>/<model_name>/*.csv`: Extracted datapoints.
  - `amendments/`
    - `restated/<model_name>/*.md`: Restated contracts after applying revisions.
    - `instructions/<model_name>/*.json`: Structured applied-instruction logs.

All documents are Markdown or plain text. Use `split_text_into_paragraphs` to standardize paragraphs.

### Source

The dataset is curated from public filings on EDGAR (U.S. SEC). It includes approximately:
- 20 ISDA Master Agreements
- 20 amendments

These examples are intended for experimentation, benchmarking, and demos.
