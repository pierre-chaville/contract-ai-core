from __future__ import annotations

"""Streamlit viewer for organizer results.

Browse results.csv by model, inspect original text and extracted metadata.
Run: streamlit run dataset/view_organizer.py
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_organizer_models() -> list[str]:
    repo_root = get_repo_root()
    base_dir = repo_root / "dataset" / "output" / "organizer"
    base_dir.mkdir(parents=True, exist_ok=True)
    models: list[str] = []
    try:
        for p in base_dir.iterdir():
            if p.is_dir() and (p / "results.csv").exists():
                models.append(p.name)
    except Exception:
        pass
    return sorted(models)


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


def find_source_file(filename: str) -> Optional[Path]:
    repo_root = get_repo_root()
    # Search under dataset/documents/organizer recursively; support .txt and .md
    base_dir = repo_root / "dataset" / "documents" / "organizer"
    if not base_dir.exists():
        return None
    # Try exact match in any nested folder
    for p in base_dir.rglob(filename):
        if p.is_file():
            return p
    # Fallback: try stem match across common extensions
    stem = Path(filename).stem
    for ext in (".txt", ".md"):
        for p in base_dir.rglob(stem + ext):
            if p.is_file():
                return p
    return None


def format_pct(conf: float | None) -> str:
    try:
        return f"{int(round(float(conf) * 100))}%" if conf is not None else ""
    except Exception:
        return ""


def main() -> None:
    st.set_page_config(page_title="Contract Organizer Viewer", layout="wide")
    st.title("Contract Organizer - Results Viewer")

    # Sidebar: model selector
    st.sidebar.header("Model")
    models = get_organizer_models()
    if "model_name" not in st.session_state and models:
        st.session_state.model_name = models[0]
    if models:
        default_index = (
            models.index(st.session_state.get("model_name", models[0]))
            if st.session_state.get("model_name") in models
            else 0
        )
        selected_model = st.sidebar.selectbox("Model", models, index=default_index)
        if selected_model != st.session_state.get("model_name"):
            st.session_state.model_name = selected_model
            st.session_state.file_idx = 0
    else:
        st.sidebar.info("No model folders found under dataset/output/organizer/")

    model_name = st.session_state.get("model_name")
    if not model_name:
        st.info("Select a model to begin.")
        return

    repo_root = get_repo_root()
    results_path = repo_root / "dataset" / "output" / "organizer" / model_name / "results.csv"
    if not results_path.exists():
        st.warning(f"results.csv not found for model '{model_name}'.")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        st.error(f"Failed to read results.csv: {exc}")
        return

    if df.empty:
        st.info("No rows in results.csv.")
        return

    # Navigation
    if "file_idx" not in st.session_state:
        st.session_state.file_idx = 0

    num_files = len(df)
    current_idx = int(st.session_state.file_idx) % num_files
    current_row = df.iloc[current_idx]
    filename = str(current_row.get("filename", ""))

    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.subheader(f"Model: {model_name} — File {current_idx + 1} / {num_files}: {filename}")
    with col_b:
        if st.button("Next file ▶"):
            st.session_state.file_idx = (int(st.session_state.file_idx) + 1) % num_files
            st.rerun()

    # Two columns: left text, right extracted fields
    left, right = st.columns(2)

    with left:
        src_path = find_source_file(filename)
        if not src_path:
            st.warning("Original file not found under dataset/documents/organizer.")
        else:
            text = read_text_best_effort(src_path)
            st.text_area("Document text", value=text, height=700)

    with right:
        st.subheader("Extracted fields")

        def show_field(title: str, value_key: str, conf_key: str, expl_key: str) -> None:
            value = current_row.get(value_key, "")
            conf = current_row.get(conf_key, None)
            if pd.isna(conf):
                conf = None
            try:
                conf_pct = format_pct(float(conf) / 100.0) if conf is not None else ""
            except Exception:
                conf_pct = ""
            expl = current_row.get(expl_key, "")
            st.markdown(f"**{title}:** {value}")
            st.caption(
                (f"confidence: {conf_pct}" if conf_pct else "") + (f" — {expl}" if expl else "")
            )

        show_field(
            "Contract type",
            "contract_type",
            "contract_type_confidence",
            "contract_type_explanation",
        )
        show_field(
            "Contract date",
            "contract_date",
            "contract_date_confidence",
            "contract_date_explanation",
        )
        show_field(
            "Amendment date",
            "amendment_date",
            "amendment_date_confidence",
            "amendment_date_explanation",
        )
        show_field(
            "Version type", "version_type", "version_type_confidence", "version_type_explanation"
        )
        show_field("Status", "status", "status_confidence", "status_explanation")

        parties = str(current_row.get("parties", "") or "")
        p_conf = current_row.get("parties_confidence", None)
        if pd.isna(p_conf):
            p_conf = None
        try:
            p_conf_pct = format_pct(float(p_conf) / 100.0) if p_conf is not None else ""
        except Exception:
            p_conf_pct = ""
        p_expl = current_row.get("parties_explanation", "")
        st.markdown("**Parties:**")
        st.text(parties)
        st.caption(
            (f"confidence: {p_conf_pct}" if p_conf_pct else "") + (f" — {p_expl}" if p_expl else "")
        )


if __name__ == "__main__":
    main()
