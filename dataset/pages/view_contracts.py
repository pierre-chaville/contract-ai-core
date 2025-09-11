from __future__ import annotations

import sys

# Contracts Viewer (Streamlit page)
# Browse individual contracts under dataset/documents/contracts/<template>/ and view:
# - Paragraphs of the contract
# - Clauses (text + clause + confidence) from classification results
# - Full text
# - Datapoints extracted
# Use the sidebar to select a template and model, and navigate with "Next contract".
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


def get_repo_root() -> Path:
    # Find the 'dataset' directory in parents, then return its parent
    here = Path(__file__).resolve()
    dataset_dir: Optional[Path] = None
    for p in here.parents:
        if p.name == "dataset":
            dataset_dir = p
            break
    return dataset_dir.parent if dataset_dir else here.parents[2]


def list_templates() -> list[str]:
    repo_root = get_repo_root()
    # Prefer keys defined in contract_types JSONs; fallback to folder names under documents/contracts
    ct_dir = repo_root / "dataset" / "contract_types"
    keys: set[str] = set()
    try:
        for p in sorted(ct_dir.glob("*.json")):
            keys.add(p.stem)
    except Exception:
        pass
    docs_dir = repo_root / "dataset" / "documents" / "contracts"
    if docs_dir.exists():
        for p in docs_dir.iterdir():
            if p.is_dir():
                keys.add(p.name)
    return sorted(keys)


def list_models_for_template(template_key: str) -> list[str]:
    repo_root = get_repo_root()
    models: set[str] = set()
    for category in ("clauses", "datapoints"):
        base = repo_root / "dataset" / "output" / category / template_key
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    models.add(d.name)
    return sorted(models)


def find_contract_files(template_key: str) -> list[Path]:
    repo_root = get_repo_root()
    base = repo_root / "dataset" / "documents" / "contracts" / template_key
    files = []
    if base.exists():
        files.extend(sorted(base.glob("*.txt")))
        files.extend(sorted(base.glob("*.md")))
    return files


def load_template_dict(template_key: str) -> dict[str, Any]:
    # Add dataset/ to sys.path so we can import utilities
    dataset_dir = get_repo_root() / "dataset"
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    from utilities import load_template  # type: ignore

    return load_template(template_key)


def read_text_best_effort(path: Path) -> str:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def simple_split_paragraphs(text: str) -> list[str]:
    # Split on blank lines; strip trailing spaces
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]


def load_classification_csv(
    template_key: str, model_name: str, stem: str
) -> Optional[pd.DataFrame]:
    repo_root = get_repo_root()
    path = repo_root / "dataset" / "output" / "clauses" / template_key / model_name / f"{stem}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        return None


def load_datapoints_csv(template_key: str, model_name: str, stem: str) -> Optional[pd.DataFrame]:
    repo_root = get_repo_root()
    path = (
        repo_root / "dataset" / "output" / "datapoints" / template_key / model_name / f"{stem}.csv"
    )
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Contracts Viewer", layout="wide")
    st.title("Contracts")

    # Sidebar selectors
    st.sidebar.header("Selection")
    templates = list_templates()
    if not templates:
        st.info("No templates found.")
        return
    if "contracts_template" not in st.session_state:
        st.session_state.contracts_template = templates[0]
    template_key = st.sidebar.selectbox(
        "Template",
        templates,
        index=templates.index(st.session_state.get("contracts_template", templates[0])),
    )
    if template_key != st.session_state.get("contracts_template"):
        st.session_state.contracts_template = template_key
        st.session_state.contract_idx = 0

    models = list_models_for_template(template_key)
    if not models:
        st.sidebar.info("No models found under output for this template.")
        model_name = None
    else:
        if "contracts_model" not in st.session_state:
            st.session_state.contracts_model = models[0]
        model_name = st.sidebar.selectbox(
            "Model", models, index=models.index(st.session_state.get("contracts_model", models[0]))
        )
        if model_name != st.session_state.get("contracts_model"):
            st.session_state.contracts_model = model_name
            st.session_state.contract_idx = 0

    # Contracts list
    files = find_contract_files(template_key)
    if not files:
        st.warning(f"No contract files found in dataset/documents/contracts/{template_key}.")
        return

    if "contract_idx" not in st.session_state:
        st.session_state.contract_idx = 0
    idx = int(st.session_state.contract_idx) % len(files)
    current_path = files[idx]

    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.subheader(
            f"Template: {template_key} — File {idx + 1} / {len(files)}: {current_path.name}"
        )
    with col_b:
        if st.button("Next contract ▶"):
            st.session_state.contract_idx = (int(st.session_state.contract_idx) + 1) % len(files)
            st.rerun()

    # Tabs
    tab_clauses, tab_datapoints = st.tabs(["Clauses", "Datapoints"])

    with tab_clauses:
        if not model_name:
            st.info("Select a model to view clause classification results.")
        else:
            df_cls = load_classification_csv(template_key, model_name, current_path.stem)
            if df_cls is None or df_cls.empty:
                st.info("No clause results found for this contract.")
            else:
                # Expect columns: index, clause_key, confidence (percent int), text
                text_series = df_cls.get("text", pd.Series(dtype=str))
                clause_series = df_cls.get("clause_key", pd.Series(dtype=str))
                # Map clause keys to titles from the template
                try:
                    tmpl = load_template_dict(template_key)
                    clauses = tmpl.get("clauses", []) or []
                    key_to_title = {
                        str(c.get("key")): c.get("title") or str(c.get("key")) for c in clauses
                    }
                except Exception:
                    key_to_title = {}
                clause_title_series = clause_series.map(lambda k: key_to_title.get(str(k), str(k)))
                conf_series = df_cls.get("confidence", pd.Series(dtype=object)).map(
                    lambda v: f"{int(v)}%" if pd.notna(v) and str(v).strip() != "" else ""
                )
                # Build clause cells: if same key as previous non-empty, show only (confidence) with left border
                keys_list = clause_series.tolist()
                titles_list = clause_title_series.tolist()
                conf_list = conf_series.tolist()
                repeat_flags: list[bool] = []
                prev_key: str | None = None
                for k in keys_list:
                    k_str = str(k).strip() if k is not None else ""
                    is_repeat = prev_key is not None and k_str != "" and k_str == prev_key
                    repeat_flags.append(bool(is_repeat))
                    if k_str != "":
                        prev_key = k_str

                clause_cells: list[str] = []
                text_cells: list[str] = []
                for is_rep, title, conf in zip(repeat_flags, titles_list, conf_list, strict=False):
                    conf_str = str(conf).strip()
                    if is_rep:
                        content = f"({conf_str})" if conf_str else ""
                        clause_cells.append(f'<div class="repeat-cell">{content}</div>')
                        # Mark text cell as repeated to draw right border
                        text_cells.append('<div class="text-repeat-cell">{}</div>')
                    else:
                        title_str = str(title).strip()
                        content = f"{title_str} ({conf_str})" if conf_str else title_str
                        clause_cells.append(f'<div class="normal-cell">{content}</div>')
                        text_cells.append('<div class="text-normal-cell">{}</div>')

                # Merge raw text into wrappers (avoid HTML breaking)
                safe_text = (
                    text_series.astype(str).str.replace("<", "&lt;").str.replace(">", "&gt;")
                )
                text_cells = [
                    tpl.format(txt)
                    for tpl, txt in zip(text_cells, safe_text.tolist(), strict=False)
                ]

                df_view = pd.DataFrame(
                    {
                        "text": pd.Series(text_cells),
                        "clause": clause_cells,
                    }
                )
                # Render a single HTML table with fixed column widths and no borders; repeated rows show a right border on the text column
                html = df_view.to_html(index=False, escape=False)
                html = html.replace(
                    '<table border="1" class="dataframe">',
                    (
                        "<style>"
                        ".wrapped-table{width:100%;table-layout:fixed;border-collapse:collapse;}"
                        ".wrapped-table th,.wrapped-table td{border:none;padding:0.5rem;vertical-align:top;word-wrap:break-word;white-space:normal;}"
                        ".wrapped-table td .text-repeat-cell{border-right:4px solid #888;padding-right:0.5rem;}"
                        ".wrapped-table col.text-col{width:50%;}"
                        ".wrapped-table col.clause-col{width:50%;}"
                        "</style>"
                        '<table class="wrapped-table">'
                        "<colgroup>"
                        '<col class="text-col"/>'
                        '<col class="clause-col"/>'
                        "</colgroup>"
                    ),
                )
                st.markdown(html, unsafe_allow_html=True)

    with tab_datapoints:
        if not model_name:
            st.info("Select a model to view datapoint extraction results.")
        else:
            df_dp = load_datapoints_csv(template_key, model_name, current_path.stem)
            if df_dp is None or df_dp.empty:
                st.info("No datapoints results found for this contract.")
            else:
                # Expect columns: key, title, confidence (percent), value, evidence, explanation
                df_view = df_dp.copy()
                if "confidence" in df_view.columns:
                    df_view["confidence"] = df_view["confidence"].map(
                        lambda v: f"{int(v)}%" if pd.notna(v) and str(v).strip() != "" else ""
                    )
                st.dataframe(df_view, use_container_width=True)


if __name__ == "__main__":
    main()
