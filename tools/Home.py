# Streamlit multi-page app entrypoint.
# Run:
#   streamlit run dataset/app.py
# Pages are discovered automatically from the dataset/pages/ directory.
from __future__ import annotations

import streamlit as st
from streamlit_mermaid import st_mermaid


def main() -> None:
    st.set_page_config(page_title="Contract AI", layout="wide")
    st.title("Contract AI")
    st.write("*NB: Use the left sidebar to navigate to a specific tool page.*")
    st.markdown("Macro process of **Contract AI**:")
    st.markdown(
        "1. **Negotiation** of a new contract\n"
        "2. **Migration** of misclassified documents\n"
        "3. **Processing** of a signed contract and amendments\n"
        "4. **Analytics**\n"
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
    I(Datapoints) --> I1(Template compare) --> I2(Ask a question)
  end
	subgraph "Migration"
		X(Organizer) --> G([Contract])
		X(Organizer) --> J([Amendment])
  end
	subgraph "Negociation"
	  A(Authoring) --> B(Guidelines Review) --> B1(Template compare) --> B2(Ask a question)
  end        """

    st_mermaid(diagram)


if __name__ == "__main__":
    main()
