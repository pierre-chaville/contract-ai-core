from __future__ import annotations

import json
import html
from pathlib import Path
from typing import List, Optional

import streamlit as st


def list_output_files(output_dir: Path) -> List[Path]:
    return sorted(output_dir.glob("*.json"))


def load_classification(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_percent_string(confidence_percent: Optional[int], confidence_float: Optional[float]) -> str:
    if confidence_percent is not None:
        return f"{confidence_percent}%"
    if confidence_float is not None:
        try:
            return f"{round(confidence_float * 100)}%"
        except Exception:
            return ""
    return ""


def render_html_table(paragraphs: List[dict]) -> str:
    # Build a simple HTML table: columns = line, clause, confidence, text
    rows_html: List[str] = []
    for p in paragraphs:
        line_no = p.get("index")
        clause = p.get("clause_key") or ""
        conf_pct = p.get("confidence_percent")
        conf_float = p.get("confidence")
        confidence = to_percent_string(conf_pct, conf_float)
        text = p.get("text") or ""

        rows_html.append(
            "<tr>"
            f"<td class=\"line\">{html.escape(str(line_no) if line_no is not None else '')}</td>"
            f"<td class=\"clause\">{html.escape(str(clause))}</td>"
            f"<td class=\"confidence\">{html.escape(confidence)}</td>"
            f"<td class=\"text\">{html.escape(str(text))}</td>"
            "</tr>"
        )

    style = (
        "<style>"
        "table { width: 100%; border-collapse: collapse; }"
        "th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }"
        "th { background: #f4f4f4; color: #111; text-align: left; }"
        ".line { width: 8%; white-space: nowrap; }"
        ".clause { width: 15%; white-space: nowrap; }"
        ".confidence { width: 10%; white-space: nowrap; }"
        ".text { width: 75%; }"
        "@media (prefers-color-scheme: dark) {"
        "  th, td { border-color: #444; }"
        "  th { background: #222; color: #fff; }"
        "  td { color: #eaeaea; }"
        "}"
        "</style>"
    )
    head = "<thead><tr><th>Line</th><th>Clause</th><th>Confidence</th><th>Text</th></tr></thead>"
    body = "<tbody>" + "".join(rows_html) + "</tbody>"
    return style + "<table>" + head + body + "</table>"


def main() -> None:
    st.set_page_config(page_title="ISDA Classifier Viewer", layout="wide")
    st.title("ISDA Classifier Viewer")

    dataset_dir = Path(__file__).resolve().parent
    output_dir = dataset_dir / "output" / "clauses" / "ISDA"

    if "file_index" not in st.session_state:
        st.session_state.file_index = 0

    json_files = list_output_files(output_dir)
    if not json_files:
        st.warning(f"No JSON files found in {output_dir}")
        return

    # Ensure index stays within bounds
    st.session_state.file_index = max(0, min(st.session_state.file_index, len(json_files) - 1))

    current_file = json_files[st.session_state.file_index]
    st.caption(f"File {st.session_state.file_index + 1} / {len(json_files)}: {current_file.name}")

    col1, _ = st.columns([1, 9])
    with col1:
        if st.button("Next file ➡", use_container_width=True):
            st.session_state.file_index = (st.session_state.file_index + 1) % len(json_files)
            st.rerun()

    data = load_classification(current_file)
    paragraphs = data.get("paragraphs", [])

    html_table = render_html_table(paragraphs)
    st.markdown(html_table, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


