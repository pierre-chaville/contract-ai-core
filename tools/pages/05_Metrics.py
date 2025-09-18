from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml  # type: ignore


def get_repo_root() -> Path:
    # Locate repo root by finding 'dataset' directory upward
    here = Path(__file__).resolve()
    dataset_dir = None
    for p in here.parents:
        if p.name == "dataset":
            dataset_dir = p
            break
    return dataset_dir.parent if dataset_dir else here.parents[2]


def list_models(base: Path) -> list[str]:
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    except Exception:
        return []


def load_csv_rows(path: Path) -> list[list[str]]:
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            return [row for row in reader]
    except Exception:
        return []


def render_migration_tab() -> None:
    repo_root = get_repo_root()
    base_dir = repo_root / "dataset" / "metrics" / "organizer"
    st.subheader("Organizer (Migration) metrics")
    models = list_models(base_dir)
    if not models:
        st.info("No organizer metrics found under dataset/metrics/organizer/")
        return
    model = st.selectbox("Model", models, index=0, key="metrics_migration_model")
    csv_path = base_dir / model / "summary.csv"
    mm_path = base_dir / model / "mismatch.csv"
    if csv_path.exists():
        rows = load_csv_rows(csv_path)
        if rows:
            st.markdown("**Summary**")
            # Use first row as header to label columns
            headers = rows[0]
            data_rows = rows[1:]
            records = [
                {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
                for row in data_rows
            ]
            st.dataframe(records, hide_index=True, use_container_width=True)
        else:
            st.info("Summary CSV is empty.")
    else:
        st.warning(f"Summary not found: {csv_path}")
    if mm_path.exists():
        st.markdown("**Mismatches**")
        rows = load_csv_rows(mm_path)
        if rows:
            st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)
        else:
            st.caption("No mismatches or file empty.")


def render_clauses_tab() -> None:
    repo_root = get_repo_root()
    st.subheader("Clauses metrics")
    # Expected layout: dataset/metrics/classification/<template>/<model>/
    templates_base = repo_root / "dataset" / "metrics" / "classification"
    templates = list_models(templates_base)
    if not templates:
        st.info("No classification metrics found.")
        return
    template = st.selectbox("Template", templates, index=0, key="metrics_clauses_template")
    models = list_models(templates_base / template)
    if not models:
        st.info("No models for this template.")
        return
    model = st.selectbox("Model", models, index=0, key="metrics_clauses_model")
    summary_path = templates_base / template / model / "summary.yaml"
    cm_csv = templates_base / template / model / "confusion_matrix.csv"
    cm_png = templates_base / template / model / "confusion_matrix.png"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = yaml.safe_load(f)
        st.markdown("**Summary**")
        st.json(summary)
    else:
        st.warning(f"Summary not found: {summary_path}")
    if cm_png.exists():
        st.image(str(cm_png), caption="Confusion matrix")
    elif cm_csv.exists():
        rows = load_csv_rows(cm_csv)
        st.markdown("**Confusion matrix**")
        st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)


def render_datapoints_tab() -> None:
    repo_root = get_repo_root()
    st.subheader("Datapoints metrics")
    # Expected layout: dataset/metrics/extraction/<template>/<model>/
    templates_base = repo_root / "dataset" / "metrics" / "extraction"
    templates = list_models(templates_base)
    if not templates:
        st.info("No extraction metrics found.")
        return
    template = st.selectbox("Template", templates, index=0, key="metrics_datapoints_template")
    models = list_models(templates_base / template)
    if not models:
        st.info("No models for this template.")
        return
    model = st.selectbox("Model", models, index=0, key="metrics_datapoints_model")
    summary_path = templates_base / template / model / "summary.yaml"
    per_key_f1 = templates_base / template / model / "per_key_f1.csv"
    per_key_acc = templates_base / template / model / "per_key_accuracy.csv"
    mm_csv = templates_base / template / model / "mismatches.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = yaml.safe_load(f)
        st.markdown("**Summary**")
        st.json(summary)
    else:
        st.warning(f"Summary not found: {summary_path}")
    if per_key_f1.exists():
        st.markdown("**Per-key F1**")
        rows = load_csv_rows(per_key_f1)
        st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)
    if per_key_acc.exists():
        st.markdown("**Per-key accuracy**")
        rows = load_csv_rows(per_key_acc)
        st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)
    if mm_csv.exists():
        st.markdown("**Mismatches**")
        rows = load_csv_rows(mm_csv)
        st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Metrics", layout="wide")
    st.title("Metrics Dashboard")
    tab_mig, tab_cls, tab_dp = st.tabs(["Migration", "Clauses", "Datapoints"])
    with tab_mig:
        render_migration_tab()
    with tab_cls:
        render_clauses_tab()
    with tab_dp:
        render_datapoints_tab()


if __name__ == "__main__":
    main()
