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


def list_contract_types(base: Path) -> list[str]:
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    except Exception:
        return []


def render_filter_tab() -> None:
    repo_root = get_repo_root()
    st.subheader("Filter metrics")
    # Expected layout: dataset/metrics/filter/<contract_type>/<model>/result.json
    base = repo_root / "dataset" / "metrics" / "filter"
    ctypes = list_contract_types(base)
    if not ctypes:
        st.info("No filter metrics found.")
        return
    ct = st.selectbox("Contract type", ctypes, index=0, key="metrics_filter_contract_type")
    models = list_models(base / ct)
    if not models:
        st.info("No models for this contract type.")
        return
    model = st.selectbox("Model", models, index=0, key="metrics_filter_model")
    result_path = base / ct / model / "result.json"
    if not result_path.exists():
        st.warning(f"Result not found: {result_path}")
        return
    try:
        import json

        data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read result.json: {e}")
        return
    # Render overview
    st.markdown("### Summary")
    overall = data.get("overall", {})
    counts = data.get("counts", {})

    def _pct(x: float | int | None) -> str:
        try:
            v = float(x if x is not None else 0.0)
            if 0.0 <= v <= 1.0:
                v *= 100.0
            return f"{int(round(v))}%"
        except Exception:
            return ""

    files_val = counts.get("files", 0)
    pairs_val = counts.get("pairs", 0)
    acc_pct = _pct(overall.get("avg_accuracy", 0.0))
    iou_pct = _pct(overall.get("avg_iou", 0.0))
    st.markdown(f"**files:** {files_val}")
    st.markdown(f"**pairs:** {pairs_val}")
    st.markdown(f"**avg_accuracy:** {acc_pct}")
    st.markdown(f"**avg_iou:** {iou_pct}")
    # Histogram
    hist = overall.get("histogram_iou") or data.get("overall", {}).get("histogram_iou") or {}
    if hist:
        st.markdown("### IoU histogram")
        # Display as simple table for clarity
        rows = [
            {"range": k, "count": hist.get(k, 0)}
            for k in ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0"]
        ]
        st.table(rows)
    # Per-scope
    ps = data.get("per_scope", {}) or {}
    if ps:
        st.markdown("### Per-scope metrics")
        scope_rows = []
        for name, vals in sorted(ps.items()):
            scope_rows.append(
                {
                    "scope": name,
                    "count": vals.get("count", 0),
                    "avg_accuracy": _pct(vals.get("avg_accuracy", 0.0)),
                    "avg_iou": _pct(vals.get("avg_iou", 0.0)),
                }
            )
        st.table(scope_rows)


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
            st.markdown("### Summary")
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
        st.markdown("### Mismatches")
        rows = load_csv_rows(mm_path)
        if rows:
            # Use first row as header to label columns
            headers = rows[0]
            data_rows = rows[1:]
            records = [
                {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
                for row in data_rows
            ]
            st.dataframe(records, hide_index=True, use_container_width=True)
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

        def _pct(v: object) -> str:
            try:
                x = float(v)  # type: ignore[arg-type]
                if 0.0 <= x <= 1.0:
                    x *= 100.0
                return f"{int(round(x))}%"
            except Exception:
                return str(v)

        if isinstance(summary, dict):
            # num_examples as number
            if "Number examples" in summary:
                st.markdown(f"**Number examples:** {summary.get('Number examples', 0)}")
            # Other top-level fields except per_category and num_examples
            for k, v in summary.items():
                if k in ("Number examples", "per_category"):
                    continue
                if isinstance(v, dict):
                    st.markdown(f"**{k}:**")
                    for sk, sv in v.items():
                        st.markdown(f"- {sk}: {_pct(sv)}")
                else:
                    st.markdown(f"**{k}:** {_pct(v)}")
            # per_category as table
            pc = summary.get("per_category")
            if isinstance(pc, list) and pc:
                st.markdown("**per_category**")
                # Format percentage fields, keep count as is
                rows = []
                for it in pc:
                    if not isinstance(it, dict):
                        continue
                    rows.append(
                        {
                            "category": it.get("category", ""),
                            "accuracy": _pct(it.get("accuracy", 0.0)),
                            "count": it.get("count", 0),
                            "f1": _pct(it.get("f1", 0.0)),
                            "precision": _pct(it.get("precision", 0.0)),
                            "recall": _pct(it.get("recall", 0.0)),
                        }
                    )
                if rows:
                    st.table(rows)
        else:
            st.write(summary)
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
        st.markdown("### Summary")

        def _pct(v: object) -> str:
            try:
                x = float(v)  # type: ignore[arg-type]
                if 0.0 <= x <= 1.0:
                    x *= 100.0
                return f"{int(round(x))}%"
            except Exception:
                return str(v)

        if isinstance(summary, dict):
            for k, v in summary.items():
                if k == "per_key":
                    continue
                if isinstance(v, dict):
                    st.markdown(f"**{k}:**")
                    for sk, sv in v.items():
                        st.markdown(f"- {sk}: {_pct(sv)}")
                else:
                    st.markdown(f"**{k}:** {_pct(v)}")
            # Per-key metrics table if present in summary
            pc = summary.get("per_key")
            if isinstance(pc, list) and pc:
                st.markdown("### Per-key metrics")
                rows = []
                for it in pc:
                    if not isinstance(it, dict):
                        continue
                    rows.append(
                        {
                            "key": it.get("key", ""),
                            "accuracy": _pct(it.get("accuracy", 0.0)),
                            "count": it.get("count", 0),
                            "f1": _pct(it.get("f1", 0.0)),
                            "precision": _pct(it.get("precision", 0.0)),
                            "recall": _pct(it.get("recall", 0.0)),
                        }
                    )
                if rows:
                    st.table(rows)
        else:
            st.write(summary)
    else:
        st.warning(f"Summary not found: {summary_path}")
    # Fallback to legacy CSVs if per_key not in summary
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                _sum = yaml.safe_load(f)
            has_per_key = isinstance(_sum, dict) and isinstance(_sum.get("per_key"), list)
        except Exception:
            has_per_key = False
    else:
        has_per_key = False
    if not has_per_key:
        if per_key_f1.exists():
            st.markdown("### Per-key F1")
            rows = load_csv_rows(per_key_f1)
            st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)
        if per_key_acc.exists():
            st.markdown("### Per-key accuracy")
            rows = load_csv_rows(per_key_acc)
            st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)
    if mm_csv.exists():
        st.markdown("### Mismatches")
        rows = load_csv_rows(mm_csv)
        st.dataframe(rows[1:], column_config=None, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Metrics", layout="wide")
    st.title("Metrics Dashboard")
    tab_mig, tab_filter, tab_cls, tab_dp = st.tabs(["Migration", "Filter", "Clauses", "Datapoints"])
    with tab_mig:
        render_migration_tab()
    with tab_filter:
        render_filter_tab()
    with tab_cls:
        render_clauses_tab()
    with tab_dp:
        render_datapoints_tab()


if __name__ == "__main__":
    main()
