# Contract Type Viewer (Streamlit page)
# Displays core template info and tables for clauses, datapoints, enums, and structures.
# Appears in the Streamlit sidebar when running: streamlit run dataset/app.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def get_repo_root() -> Path:
    # Find the 'dataset' directory in parents, then return its parent
    here = Path(__file__).resolve()
    dataset_dir = None
    for p in here.parents:
        if p.name == "dataset":
            dataset_dir = p
            break
    return dataset_dir.parent if dataset_dir else here.parents[2]


def list_templates() -> list[str]:
    repo_root = get_repo_root()
    ct_dir = repo_root / "dataset" / "contract_types"
    keys: list[str] = []
    try:
        for p in sorted(ct_dir.glob("*.json")):
            keys.append(p.stem)
    except Exception:
        pass
    return keys


def load_template_dict(template_key: str) -> dict[str, Any]:
    # Add dataset/ to sys.path so we can import utilities
    dataset_dir = get_repo_root() / "dataset"
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    from utilities import load_template  # type: ignore

    return load_template(template_key)


def main() -> None:
    st.set_page_config(page_title="Contract Type Viewer", layout="wide")

    # Sidebar: select template key
    st.sidebar.header("Selection")
    keys = list_templates()
    if not keys:
        st.warning("No template JSON files found under dataset/contract_types/")
        return

    if "template_key" not in st.session_state:
        st.session_state.template_key = keys[0]
    selected = st.sidebar.selectbox(
        "Contract type", keys, index=keys.index(st.session_state.get("template_key", keys[0]))
    )
    if selected != st.session_state.get("template_key"):
        st.session_state.template_key = selected

    # Load template model (dict)
    model = load_template_dict(selected)
    st.title("Contract types")
    st.subheader(f"{model.get('name') or selected}")

    tab_overview, tab_clauses, tab_datapoints, tab_guidelines, tab_enums, tab_structures = st.tabs(
        [
            "Overview",
            "Clauses",
            "Datapoints",
            "Guidelines",
            "Enums",
            "Structures",
        ]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            st.subheader("Main")
            main_rows = [
                {"label": "key", "value": model.get("key")},
                {"label": "name", "value": model.get("name")},
                {"label": "use case", "value": model.get("use_case")},
                {"label": "description", "value": model.get("description")},
                {"label": "prompt scope filter", "value": model.get("prompt_scope_filter")},
                {"label": "prompt scope amendment", "value": model.get("prompt_scope_amendment")},
            ]
            st.table(pd.DataFrame(main_rows))
        with right:
            st.subheader("Counts")
            st.write(
                {
                    "clauses": len(model.get("clauses", [])),
                    "datapoints": len(model.get("datapoints", [])),
                    "guidelines": len(model.get("guidelines", [])),
                    "enums": len(model.get("enums", [])),
                    "structures": len(model.get("structures", [])),
                }
            )

    with tab_clauses:
        clauses = model.get("clauses", []) or []
        df = pd.DataFrame(clauses)
        if not df.empty:
            # Reorder columns if present
            preferred = ["key", "title", "description", "parent_key", "sort_order"]
            cols = [c for c in preferred if c in df.columns] + [
                c for c in df.columns if c not in preferred
            ]
            # Use st.table to allow multi-line wrapping (st.dataframe often truncates)
            st.table(df[cols])
        else:
            st.info("No clauses defined.")

    with tab_datapoints:
        dps = model.get("datapoints", []) or []
        df = pd.DataFrame(dps)
        if not df.empty:
            preferred = [
                "key",
                "title",
                "data_type",
                "enum_key",
                "enum_multi_select",
                "scope",
                "clause_keys",
                "description",
                "sort_order",
            ]
            # Format clause_keys as comma-separated list for readability
            if "clause_keys" in df.columns:
                df = df.copy()
                df["clause_keys"] = df["clause_keys"].apply(
                    lambda v: ", ".join([str(x).strip() for x in v])
                    if isinstance(v, (list, tuple))
                    else ("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
                )
            cols = [c for c in preferred if c in df.columns] + [
                c for c in df.columns if c not in preferred
            ]
            # Use st.table to allow multi-line wrapping for description
            st.table(df[cols])
        else:
            st.info("No datapoints defined.")

    with tab_guidelines:
        guidelines = model.get("guidelines", []) or []
        if not guidelines:
            st.info("No guidelines defined.")
        else:
            df = pd.DataFrame(guidelines)
            # Choose key columns and keep others if present
            preferred = [
                "key",
                "guideline",
                "scope",
                "clause_keys",
                "fallback_from_key",
                "priority",
                "sort_order",
            ]
            # Format clause_keys as comma-separated list for readability
            if "clause_keys" in df.columns:
                df = df.copy()
                df["clause_keys"] = df["clause_keys"].apply(
                    lambda v: ", ".join([str(x).strip() for x in v])
                    if isinstance(v, (list, tuple))
                    else ("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
                )
            cols = [c for c in preferred if c in df.columns] + [
                c for c in df.columns if c not in preferred
            ]
            # Use static table to allow wrapping of long guideline text
            st.table(df[cols])

    with tab_enums:
        enums = model.get("enums", []) or []
        if not enums:
            st.info("No enums defined.")
        else:
            for e in enums:
                key = e.get("key")
                title = e.get("title") or ""
                st.markdown(f"### {key} — {title}")

                # Show enum key/title
                info_rows = [
                    {"field": "enum_key", "value": key},
                    {"field": "enum_title", "value": title},
                ]
                st.table(pd.DataFrame(info_rows))

                # Show options (codes & descriptions)
                opt_rows: list[dict[str, Any]] = []
                for opt in e.get("options", []) or []:
                    opt_rows.append(
                        {
                            "code": opt.get("code"),
                            "description": opt.get("description"),
                        }
                    )
                if opt_rows:
                    st.dataframe(pd.DataFrame(opt_rows), use_container_width=True)
                else:
                    st.info("No options for this enum.")
                st.divider()

    with tab_structures:
        structs = model.get("structures", []) or []
        if not structs:
            st.info("No structures defined.")
        else:
            for s in structs:
                skey = s.get("structure_key")
                title = s.get("title") or ""
                st.markdown(f"### {skey} — {title}")
                # Show structure core fields
                info_cols = [
                    {"field": "structure_key", "value": s.get("structure_key")},
                    {"field": "title", "value": s.get("title")},
                    {"field": "description", "value": s.get("description")},
                ]
                st.table(pd.DataFrame(info_cols))

                # Show elements for this structure
                el_rows: list[dict[str, Any]] = []
                for el in s.get("elements", []) or []:
                    el_rows.append(
                        {
                            "key": el.get("key"),
                            "title": el.get("title"),
                            "description": el.get("description"),
                            "data_type": el.get("data_type"),
                            "required": el.get("required"),
                            "enum_key": el.get("enum_key"),
                            "sort_order": el.get("sort_order"),
                        }
                    )
                if el_rows:
                    st.dataframe(pd.DataFrame(el_rows), use_container_width=True)
                else:
                    st.info("No elements for this structure.")
                st.divider()


if __name__ == "__main__":
    main()
