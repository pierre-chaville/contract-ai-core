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

    (
        tab_overview,
        tab_clauses,
        tab_datapoints,
        tab_guidelines,
        tab_enums,
        tab_structures,
        tab_export,
    ) = st.tabs(
        [
            "Overview",
            "Clauses",
            "Datapoints",
            "Guidelines",
            "Enums",
            "Structures",
            "Export/Import",
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
                {"label": "prompt scope amendment", "value": model.get("prompt_scope_amendment")},
            ]
            for row in main_rows:
                st.markdown(f"**{row['label']}:** {row['value']}")
            scopes = model.get("filtering_scopes") or []
            st.markdown("**filtering scopes:**")
            if scopes:
                for s in scopes:
                    name = s.get("name") or ""
                    desc = s.get("description") or ""
                    st.markdown(f"- {name}: {desc}")
            else:
                st.markdown("- none")
        with right:
            st.subheader("Counts")
            counts = [
                ("clauses", len(model.get("clauses", []))),
                ("datapoints", len(model.get("datapoints", []))),
                ("guidelines", len(model.get("guidelines", []))),
                ("enums", len(model.get("enums", []))),
                ("structures", len(model.get("structures", []))),
            ]
            for label, value in counts:
                st.markdown(f"**{label}:** {value}")

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
                "action",
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

    with tab_export:
        st.subheader("Export / Import")
        repo_root = get_repo_root()
        ct_dir = repo_root / "dataset" / "contract_types"
        part_options = [
            "main json",
            "clauses",
            "datapoints",
            "guidelines",
            "enums",
            "structures",
            "structure elements",
        ]
        if "export_part" not in st.session_state:
            st.session_state.export_part = part_options[0]
        part = st.selectbox(
            "Select part",
            part_options,
            index=part_options.index(st.session_state.get("export_part", part_options[0])),
            help="Choose which part of the contract type to export or import.",
        )
        if part != st.session_state.get("export_part"):
            st.session_state.export_part = part

        def df_for_part(model_dict: dict[str, Any], which: str) -> pd.DataFrame:
            which = which.lower().strip()
            if which == "clauses":
                rows = model_dict.get("clauses", []) or []
                preferred = ["key", "title", "description", "sort_order"]
                dfc = pd.DataFrame(rows)
                if dfc.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfc.columns] + [
                    c for c in dfc.columns if c not in preferred
                ]
                return dfc[cols]
            if which == "datapoints":
                rows = model_dict.get("datapoints", []) or []
                # Normalize clause_keys to comma-separated string for export
                normed = []
                for r in rows:
                    rr = dict(r)
                    cks = rr.get("clause_keys")
                    if isinstance(cks, (list, tuple)):
                        rr["clause_keys"] = ",".join(
                            [str(x).strip() for x in cks if str(x).strip()]
                        )
                    normed.append(rr)
                preferred = [
                    "key",
                    "title",
                    "description",
                    "required",
                    "data_type",
                    "enum_key",
                    "scope",
                    "clause_keys",
                    "sort_order",
                ]
                dfd = pd.DataFrame(normed)
                if dfd.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfd.columns] + [
                    c for c in dfd.columns if c not in preferred
                ]
                return dfd[cols]
            if which == "guidelines":
                rows = model_dict.get("guidelines", []) or []
                # Map model "key" back to CSV column "id"
                normed = []
                for r in rows:
                    rr = dict(r)
                    rr["id"] = rr.pop("key", None)
                    cks = rr.get("clause_keys")
                    if isinstance(cks, (list, tuple)):
                        rr["clause_keys"] = ",".join(
                            [str(x).strip() for x in cks if str(x).strip()]
                        )
                    normed.append(rr)
                preferred = [
                    "id",
                    "guideline",
                    "action",
                    "scope",
                    "clause_keys",
                    "fallback_from_key",
                    "priority",
                    "sort_order",
                ]
                dfg = pd.DataFrame(normed)
                if dfg.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfg.columns] + [
                    c for c in dfg.columns if c not in preferred
                ]
                return dfg[cols]
            if which == "enums":
                enums = model_dict.get("enums", []) or []
                flat_rows: list[dict[str, Any]] = []
                for e in enums:
                    ek = e.get("key")
                    et = e.get("title")
                    for opt in e.get("options") or []:
                        flat_rows.append(
                            {
                                "key": ek,
                                "title": et,
                                "code": opt.get("code"),
                                "description": opt.get("description"),
                            }
                        )
                preferred = ["key", "title", "code", "description"]
                dfe = pd.DataFrame(flat_rows)
                if dfe.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfe.columns] + [
                    c for c in dfe.columns if c not in preferred
                ]
                return dfe[cols]
            if which == "structures":
                structs = model_dict.get("structures", []) or []
                rows = [
                    {
                        "structure_key": s.get("structure_key"),
                        "title": s.get("title"),
                        "description": s.get("description"),
                    }
                    for s in structs
                ]
                preferred = ["structure_key", "title", "description"]
                dfs = pd.DataFrame(rows)
                if dfs.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfs.columns] + [
                    c for c in dfs.columns if c not in preferred
                ]
                return dfs[cols]
            if which == "structure elements":
                structs = model_dict.get("structures", []) or []
                rows = []
                for s in structs:
                    sk = s.get("structure_key")
                    for el in s.get("elements") or []:
                        rr = dict(el)
                        rr["structure_key"] = sk
                        rows.append(rr)
                preferred = [
                    "structure_key",
                    "key",
                    "title",
                    "description",
                    "data_type",
                    "required",
                    "enum_key",
                    "sort_order",
                ]
                dfelem = pd.DataFrame(rows)
                if dfelem.empty:
                    return pd.DataFrame(columns=preferred)
                cols = [c for c in preferred if c in dfelem.columns] + [
                    c for c in dfelem.columns if c not in preferred
                ]
                return dfelem[cols]
            return pd.DataFrame()

        if part == "main json":
            # Export main template JSON as-is from disk
            json_path = ct_dir / f"{selected}.json"
            try:
                json_text = json_path.read_text(encoding="utf-8")
            except Exception as e:
                st.error(f"Failed to read JSON: {e}")
                json_text = "{}"

            st.download_button(
                label="Download main JSON",
                data=json_text.encode("utf-8"),
                file_name=f"{selected}.json",
                mime="application/json",
            )

            # Import main JSON
            uploaded_json = st.file_uploader(
                "Upload main JSON to replace",
                type=["json"],
                key=f"uploader::{selected}::main_json",
                help="Uploading will overwrite the main JSON on disk.",
            )
            if uploaded_json is not None:
                try:
                    out_path = ct_dir / f"{selected}.json"
                    with out_path.open("wb") as f:
                        f.write(uploaded_json.getbuffer())
                except Exception as e:
                    st.error(f"Failed to save JSON: {e}")
                else:
                    st.success(f"Saved to {out_path.as_posix()}.")
                    if st.button("Reload template"):
                        st.rerun()
        else:
            df_current = df_for_part(model, part)

            # Export CSV for the selected part
            file_suffix = (
                "clauses.csv"
                if part == "clauses"
                else (
                    "datapoints.csv"
                    if part == "datapoints"
                    else (
                        "guidelines.csv"
                        if part == "guidelines"
                        else (
                            "enums.csv"
                            if part == "enums"
                            else (
                                "structures.csv"
                                if part == "structures"
                                else "structure_elements.csv"
                            )
                        )
                    )
                )
            )
            export_name = f"{selected}_{file_suffix}"
            st.download_button(
                label=f"Download {part} CSV",
                data=df_current.to_csv(index=False).encode("utf-8"),
                file_name=export_name,
                mime="text/csv",
            )

            # Import CSV
            uploaded = st.file_uploader(
                f"Upload {part} CSV to replace",
                type=["csv"],
                key=f"uploader::{selected}::{part}",
                help="Uploading will overwrite the corresponding CSV on disk.",
            )
            if uploaded is not None:
                try:
                    out_path = ct_dir / export_name
                    with out_path.open("wb") as f:
                        f.write(uploaded.getbuffer())
                except Exception as e:
                    st.error(f"Failed to save CSV: {e}")
                else:
                    st.success(f"Saved to {out_path.as_posix()}.")
                    if st.button("Reload template"):
                        st.rerun()

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
