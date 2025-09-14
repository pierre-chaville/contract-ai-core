from __future__ import annotations

# Streamlit multi-page app entrypoint.
# Run:
#   streamlit run dataset/app.py
# Pages are discovered automatically from the dataset/pages/ directory.
from pathlib import Path

import streamlit as st
from streamlit_mermaid import st_mermaid


def main() -> None:
    st.set_page_config(page_title="Contract AI - Tools", layout="wide")
    st.title("Contract AI - Tools")

    st.write("Use the left sidebar to navigate to a specific tool page.")

    repo_root = Path(__file__).resolve().parents[1]
    st.markdown(f"Repository root: `{repo_root}`\n\n" "This app provides:")
    st.markdown(
        "- Organizer Results Viewer (browse organizer results and document text)\n"
        "- Reviser Instructions Viewer (inspect amendment instructions and restated output)"
    )
    diagram = """
        graph TD
            subgraph "Analytics"
            Y(Database loading) --> Z(User query)
        end
            subgraph "Processing"
            G([Contract]) --> P(Contract filtered)
            P(Contract filtered) --> H(Clauses) --> I(Datapoints)
            J([Amendment]) --> K(Restated) --> H(Clauses)
            P(Contract filtered) --> K(Restated)
            I(Datapoints) --> Y(Database loading)
        end
            subgraph "Migration"
                X(Organizer) --> G([Contract])
                X(Organizer) --> J([Amendment])
        end
            subgraph "Negociation"
            A(Templating) --> B(Guidelines Review)
        end
        """

    st_mermaid(diagram)


if __name__ == "__main__":
    main()
