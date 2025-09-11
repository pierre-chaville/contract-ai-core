from __future__ import annotations

"""Streamlit multi-page app entrypoint.

Run:
  streamlit run dataset/app.py

Pages are discovered automatically from the dataset/pages/ directory.
"""

from pathlib import Path

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Contract AI - Tools", layout="wide")
    st.title("Contract AI - Dataset Tools")

    st.write("Use the left sidebar to navigate to a specific tool page.")

    repo_root = Path(__file__).resolve().parents[1]
    st.markdown(f"Repository root: `{repo_root}`\n\n" "This app provides:")
    st.markdown(
        "- Organizer Results Viewer (browse organizer results and document text)\n"
        "- Reviser Instructions Viewer (inspect amendment instructions and restated output)"
    )


if __name__ == "__main__":
    main()
