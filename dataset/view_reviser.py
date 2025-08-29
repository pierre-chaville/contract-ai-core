from __future__ import annotations
"""Streamlit viewer for amendment instructions and restated documents.

Browse instruction JSONs by model, inspect original/amendment/restated content.
Run: streamlit run dataset/view_reviser.py
"""

import json
import html
from pathlib import Path
from contract_ai_core.schema import split_text_into_paragraphs  # type: ignore

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sys


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_instruction_models() -> list[str]:
    repo_root = get_repo_root()
    base_dir = repo_root / "dataset" / "output" / "amendments" / "instructions"
    base_dir.mkdir(parents=True, exist_ok=True)
    models: list[str] = []
    try:
        for p in base_dir.iterdir():
            if p.is_dir():
                models.append(p.name)
    except Exception:
        pass
    return sorted(models)


def get_instruction_files(model_name: str | None = None) -> list[Path]:
    repo_root = get_repo_root()
    instructions_dir = repo_root / "dataset" / "output" / "amendments" / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    if not model_name:
        return []
    target_dir = instructions_dir / model_name
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    return sorted(target_dir.glob("*.json"))


def load_json(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


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


def find_initial_contract_path(stem: str) -> Path | None:
    repo_root = get_repo_root()
    docs_dir = repo_root / "dataset" / "documents" / "amendments" / "initial"
    candidates = list(docs_dir.glob(f"{stem}.md"))
    return candidates[0] if candidates else None

def find_amendment_path(stem: str) -> Path | None:
    repo_root = get_repo_root()
    docs_dir = repo_root / "dataset" / "documents" / "amendments" / "amendment"
    candidates = list(docs_dir.glob(f"{stem}.md"))
    return candidates[0] if candidates else None

def get_restated_path(stem: str, model_name: str) -> Path:
    repo_root = get_repo_root()
    return repo_root / "dataset" / "output" / "amendments" / "restated" / model_name / f"{stem}.md"


def render_paragraphs(paragraphs: list[dict] | None) -> str:
    if not paragraphs:
        return "<em>None</em>"
    blocks: list[str] = []
    for p in paragraphs:
        idx = html.escape(str(p.get("index", "")))
        text = html.escape(str(p.get("text", "")))
        blocks.append(f"<div class='pitem'><span class='pidx'>[{idx}]</span> {text}</div>")
    return "".join(blocks)


def build_table(instructions: list[dict]) -> str:
    styles = """
    <style>
    :root { color-scheme: light dark; }
    html, body { margin: 0; padding: 0; }
    body { background: #ffffff; color: #111111; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }
    .rev-table { border-collapse: collapse; width: 100%; table-layout: fixed; color: inherit; font-family: inherit; }
    .rev-table th, .rev-table td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
    .rev-table th { background: #f6f6f6; text-align: left; }
    .cell-scroll { max-height: none; overflow: visible; }
    .pitem { margin-bottom: 6px; }
    .pidx { color: #666; margin-right: 6px; font-family: monospace; }
    .muted { color: #666; }

    @media (prefers-color-scheme: dark) {
      body { background: #0e1117; color: #e6e6e6; }
      .rev-table th { background: #1b1f2a; color: #e6e6e6; }
      .rev-table td, .rev-table th { border-color: #2b2f3a; }
      .pidx, .muted { color: #9aa4b2; }
    }
    </style>
    """

    header = (
        "<tr>"
        "<th style='width:20%'>Amendment</th>"
        "<th style='width:30%'>Target & explanation</th>"
        "<th style='width:25%'>Initial paragraphs</th>"
        "<th style='width:25%'>Revised paragraphs</th>"
        "<th style='width:20%'>Revision</th>"
        "</tr>"
    )

    rows: list[str] = []
    for ins in instructions:
        amendment = html.escape(str(ins.get("amendment_span_text", "")))
        target = html.escape(str(ins.get("target_section", "")))
        conf_target = ins.get("confidence_target")
        change_expl = html.escape(str(ins.get("change_explanation", "")))
        combined_target_expl = (
            (f"<div><strong>{target}</strong></div>" if target else "")
            + f"<div class='muted'>confidence_target: {conf_target}</div>"
            + f"<div>{change_expl}</div>"
        )

        initial_html = render_paragraphs(ins.get("initial_paragraphs") or [])
        revised_html = render_paragraphs(ins.get("revised_paragraphs") or [])

        conf_rev = ins.get("confidence_revision")
        rev_expl = html.escape(str(ins.get("revision_explanation", "")))
        revision_html = (
            f"<div class='muted'>confidence_revision: {conf_rev}</div>"
            f"<div>{rev_expl}</div>"
        )

        row = (
            "<tr>"
            f"<td><div class='cell-scroll'>{amendment}</div></td>"
            f"<td><div class='cell-scroll'>{combined_target_expl}</div></td>"
            f"<td><div class='cell-scroll'>{initial_html}</div></td>"
            f"<td><div class='cell-scroll'>{revised_html}</div></td>"
            f"<td><div class='cell-scroll'>{revision_html}</div></td>"
            "</tr>"
        )
        rows.append(row)


    table_html = styles + "<table class='rev-table'>" + header + "".join(rows) + "</table>"
    return table_html


def estimate_height(instructions: list[dict]) -> int:
    lines = 10
    for ins in instructions:
        amendment = str(ins.get("amendment_span_text", ""))
        lines += amendment.count("\n") + 1
        lines += len(ins.get("initial_paragraphs") or [])
        lines += len(ins.get("revised_paragraphs") or [])
        lines += 6
    px = lines * 22 + 200
    if px < 400:
        return 400
    if px > 8000:
        return 80000
    return int(px)

def main() -> None:
    st.set_page_config(page_title="Contract Reviser Viewer", layout="wide")
    st.title("Contract Reviser - Instructions Viewer")

    # Sidebar filters
    st.sidebar.header("Filters")
    models = get_instruction_models()
    if "model_name" not in st.session_state and models:
        st.session_state.model_name = models[0]
    if models:
        default_index = models.index(st.session_state.get("model_name", models[0])) if st.session_state.get("model_name") in models else 0
        selected_model = st.sidebar.selectbox("Model", models, index=default_index)
        if selected_model != st.session_state.get("model_name"):
            st.session_state.model_name = selected_model
            st.session_state.file_idx = 0
    else:
        st.sidebar.info("No model folders found under dataset/output/amendments/instructions/")

    files = get_instruction_files(st.session_state.get("model_name"))
    if "file_idx" not in st.session_state:
        st.session_state.file_idx = 0

    if not files:
        st.info("No instruction JSON files found for the selected model.")
        return

    col_a, col_b = st.columns([6, 1])
    with col_a:
        current_idx = int(st.session_state.file_idx) % len(files)
        current_file = files[current_idx]
        st.subheader(f"Model: {st.session_state.get('model_name', '')} — File {current_idx + 1} / {len(files)}: {current_file.name}")
    with col_b:
        if st.button("Next file ▶"):
            st.session_state.file_idx = (int(st.session_state.file_idx) + 1) % len(files)
            st.rerun()

    instructions = load_json(current_file)
    if not instructions:
        st.warning("Selected file contains no instructions or failed to parse JSON.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Instructions", "Amendment", "Initial document", "Revised document"])

    # -------- Tab 1: Instructions (existing)
    with tab1:
        # Build a native Streamlit table to avoid iframe height issues and browser zoom truncation
        def format_paragraphs_text(paragraphs: list[dict] | None) -> str:
            if not paragraphs:
                return ""
            lines: list[str] = []
            for p in paragraphs:
                idx = p.get("index", "")
                txt = str(p.get("text", ""))
                lines.append(f"[{idx}] {txt}")
            return "\n\n".join(lines)

        table_rows = []
        for ins in instructions:
            amendment = str(ins.get("amendment_span_text", ""))
            target = str(ins.get("target_section", ""))
            conf_target = ins.get("confidence_target")
            change_expl = str(ins.get("change_explanation", ""))
            combined_target_expl = (
                (f"{target}\n" if target else "")
                + f"confidence_target: {conf_target}\n"
                + f"{change_expl}"
            )
            initial_txt = format_paragraphs_text(ins.get("initial_paragraphs") or [])
            revised_txt = format_paragraphs_text(ins.get("revised_paragraphs") or [])
            conf_rev = ins.get("confidence_revision")
            rev_expl = str(ins.get("revision_explanation", ""))
            revision_txt = f"confidence_revision: {conf_rev}\n{rev_expl}"
            table_rows.append(
                {
                    "Amendment": amendment,
                    "Target & explanation": combined_target_expl,
                    "Initial paragraphs": initial_txt,
                    "Revised paragraphs": revised_txt,
                    "Revision": revision_txt,
                }
            )

        df = pd.DataFrame(table_rows, columns=[
            "Amendment", "Target & explanation", "Initial paragraphs", "Revised paragraphs", "Revision"
        ])
        # Style: top-align cells and set equal column widths; preserve line breaks
        styler = (
            df.style
            .set_table_styles([
                {"selector": "table", "props": [("table-layout", "fixed"), ("width", "100%")]},
                {"selector": "th, td", "props": [("vertical-align", "top"), ("text-align", "left")]},
            ])
        )
        for col in df.columns:
            styler = styler.set_properties(subset=[col], **{"width": "20%", "white-space": "pre-wrap", "vertical-align": "top"})
        st.table(styler)

    # -------- Tab 2: Amendment
    with tab2:
        stem = current_file.stem
        init_path = find_amendment_path(stem)
        if not init_path:
            st.info("Could not find amendment document markdown by stem.")
        else:
            # Load, split into paragraphs, display
            text = read_text_best_effort(init_path)
            paras = split_text_into_paragraphs(text)
            init_rows = [
                {"Index": p.index, "Text": p.text}
                for p in paras
            ]
            init_df = pd.DataFrame(init_rows, columns=["Text"])
            init_styler = (
                init_df.style
                .set_table_styles([
                    {"selector": "table", "props": [("table-layout", "fixed"), ("width", "100%")]},
                    {"selector": "th, td", "props": [("vertical-align", "top"), ("text-align", "left")]},
                ])
                .set_properties(subset=["Text"], **{"width": "90%", "white-space": "pre-wrap", "vertical-align": "top"})
            )
            st.table(init_styler)

    # -------- Tab 3: Initial document
    with tab3:
        stem = current_file.stem
        init_path = find_initial_contract_path(stem)
        if not init_path:
            st.info("Could not find initial document markdown by stem.")
        else:
            # Load, split into paragraphs, display
            text = read_text_best_effort(init_path)
            paras = split_text_into_paragraphs(text)
            init_rows = [
                {"Index": p.index, "Text": p.text}
                for p in paras
            ]
            init_df = pd.DataFrame(init_rows, columns=["Text"])
            init_styler = (
                init_df.style
                .set_table_styles([
                    {"selector": "table", "props": [("table-layout", "fixed"), ("width", "100%")]},
                    {"selector": "th, td", "props": [("vertical-align", "top"), ("text-align", "left")]},
                ])
                .set_properties(subset=["Text"], **{"width": "90%", "white-space": "pre-wrap", "vertical-align": "top"})
            )
            st.table(init_styler)

    # -------- Tab 4: Revised document
    with tab4:
        stem = current_file.stem
        revised_path = get_restated_path(stem, st.session_state.get("model_name", ""))
        if not revised_path.exists():
            st.info("Revised (restated) document not found.")
        else:
            text = read_text_best_effort(revised_path)
            # For revised, paragraphs were already produced; display as lines
            lines = [x for x in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n") if x.strip()]
            rev_rows = [
                {"Index": i, "Text": t.strip()} for i, t in enumerate(lines)
            ]
            rev_df = pd.DataFrame(rev_rows, columns=["Text"])
            rev_styler = (
                rev_df.style
                .set_table_styles([
                    {"selector": "table", "props": [("table-layout", "fixed"), ("width", "100%")]},
                    {"selector": "th, td", "props": [("vertical-align", "top"), ("text-align", "left")]},
                ])
                .set_properties(subset=["Text"], **{"width": "90%", "white-space": "pre-wrap", "vertical-align": "top"})
            )
            st.table(rev_styler)


if __name__ == "__main__":
    main()


