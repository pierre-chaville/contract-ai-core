from __future__ import annotations

import sys

# Contracts Viewer (Streamlit page)
# Browse individual contracts under dataset/documents/contracts/<template>/ and view:
# - Paragraphs of the contract
# - Clauses (text + clause + confidence) from classification results
# - Full text
# - Datapoints extracted
# Use the sidebar to select a template and model, and navigate with "Next contract".
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st


def get_repo_root() -> Path:
    # Find the 'dataset' directory in parents, then return its parent
    here = Path(__file__).resolve()
    dataset_dir: Optional[Path] = None
    for p in here.parents:
        if p.name == "dataset":
            dataset_dir = p
            break
    return dataset_dir.parent if dataset_dir else here.parents[2]


def list_templates() -> list[str]:
    repo_root = get_repo_root()
    # Prefer keys defined in contract_types JSONs; fallback to folder names under documents/contracts
    ct_dir = repo_root / "dataset" / "contract_types"
    keys: set[str] = set()
    try:
        for p in sorted(ct_dir.glob("*.json")):
            keys.add(p.stem)
    except Exception:
        pass
    docs_dir = repo_root / "dataset" / "documents" / "contracts"
    if docs_dir.exists():
        for p in docs_dir.iterdir():
            if p.is_dir():
                keys.add(p.name)
    return sorted(keys)


def list_models_for_template(template_key: str) -> list[str]:
    repo_root = get_repo_root()
    models: set[str] = set()
    for category in ("clauses", "datapoints"):
        base = repo_root / "dataset" / "output" / category / template_key
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    models.add(d.name)
    return sorted(models)


def find_contract_files(template_key: str) -> list[Path]:
    repo_root = get_repo_root()
    base = repo_root / "dataset" / "documents" / "contracts" / template_key
    files = []
    if base.exists():
        files.extend(sorted(base.glob("*.txt")))
        files.extend(sorted(base.glob("*.md")))
    return files


def load_template_dict(template_key: str) -> dict[str, Any]:
    # Add dataset/ to sys.path so we can import utilities
    dataset_dir = get_repo_root() / "dataset"
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    from utilities import load_template  # type: ignore

    return load_template(template_key)


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


def load_classification_csv(
    template_key: str, model_name: str, stem: str
) -> Optional[pd.DataFrame]:
    repo_root = get_repo_root()
    path = repo_root / "dataset" / "output" / "clauses" / template_key / model_name / f"{stem}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        return None


def load_datapoints_csv(template_key: str, model_name: str, stem: str) -> Optional[pd.DataFrame]:
    repo_root = get_repo_root()
    path = (
        repo_root / "dataset" / "output" / "datapoints" / template_key / model_name / f"{stem}.csv"
    )
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        return None


def load_guidelines_csv(template_key: str, model_name: str, stem: str) -> Optional[pd.DataFrame]:
    repo_root = get_repo_root()
    path = (
        repo_root / "dataset" / "output" / "guidelines" / template_key / model_name / f"{stem}.csv"
    )
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path).fillna("")
        return df
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Contracts Viewer", layout="wide")
    st.title("Contracts")

    # Sidebar selectors
    st.sidebar.header("Selection")
    templates = list_templates()
    if not templates:
        st.info("No templates found.")
        return
    if "contracts_template" not in st.session_state:
        st.session_state.contracts_template = templates[0]
    template_key = st.sidebar.selectbox(
        "Template",
        templates,
        index=templates.index(st.session_state.get("contracts_template", templates[0])),
    )
    if template_key != st.session_state.get("contracts_template"):
        st.session_state.contracts_template = template_key
        st.session_state.contract_idx = 0

    models = list_models_for_template(template_key)
    if not models:
        st.sidebar.info("No models found under output for this template.")
        model_name = None
    else:
        if "contracts_model" not in st.session_state:
            st.session_state.contracts_model = models[0]
        model_name = st.sidebar.selectbox(
            "Model", models, index=models.index(st.session_state.get("contracts_model", models[0]))
        )
        if model_name != st.session_state.get("contracts_model"):
            st.session_state.contracts_model = model_name
            st.session_state.contract_idx = 0

    # Contracts list
    files = find_contract_files(template_key)
    if not files:
        st.warning(f"No contract files found in dataset/documents/contracts/{template_key}.")
        return

    if "contract_idx" not in st.session_state:
        st.session_state.contract_idx = 0
    idx = int(st.session_state.contract_idx) % len(files)
    current_path = files[idx]

    col_a, col_b = st.columns([6, 1])
    with col_a:
        st.subheader(
            f"Template: {template_key} — File {idx + 1} / {len(files)}: {current_path.name}"
        )
    with col_b:
        if st.button("Next contract ▶"):
            st.session_state.contract_idx = (int(st.session_state.contract_idx) + 1) % len(files)
            st.rerun()

    # Tabs
    tab_processing, tab_review, tab_datapoints, tab_clauses = st.tabs(
        ["Processing", "Review", "Datapoints", "Clauses"]
    )

    with tab_datapoints:
        if not model_name:
            st.info("Select a model to view datapoints.")
        else:
            df_dp_tab = load_datapoints_csv(template_key, model_name, current_path.stem)
            if df_dp_tab is None or df_dp_tab.empty:
                st.info("No datapoints results found for this contract.")
            else:
                # Render datapoints one by one using markdown
                dfv = df_dp_tab.copy()

                # Load template to detect structure types and element titles
                try:
                    tmpl = load_template_dict(template_key)
                except Exception:
                    tmpl = {}
                dp_defs = (tmpl.get("datapoints") or []) if isinstance(tmpl, dict) else []
                struct_defs = (tmpl.get("structures") or []) if isinstance(tmpl, dict) else []

                def parse_structure_type(data_type: Any | None) -> tuple[str, str | None]:
                    s = str(data_type or "").strip().lower()
                    if s.startswith("list[") and "object:" in s:
                        inside = s[s.find("[") + 1 : s.rfind("]")]
                        if inside.startswith("object:"):
                            key = inside.split(":", 1)[1].strip().strip("[]")
                            return "list_object", key
                    if s.startswith("object:"):
                        return "object", s.split(":", 1)[1].strip().strip("[]")
                    return "simple", None

                dp_key_to_struct: dict[str, tuple[str, str]] = {}
                struct_key_to_el_title: dict[str, dict[str, str]] = {}
                try:
                    for sd in struct_defs:
                        skey = str(sd.get("structure_key", "")).strip()
                        el_map: dict[str, str] = {}
                        for el in sd.get("elements") or []:
                            el_key = str(el.get("key", "")).strip()
                            el_title = str(el.get("title", el_key)).strip()
                            if el_key:
                                el_map[el_key] = el_title
                        if skey:
                            struct_key_to_el_title[skey] = el_map
                except Exception:
                    struct_key_to_el_title = {}
                try:
                    for dp in dp_defs:
                        key = str(dp.get("key", "")).strip()
                        kind, skey = parse_structure_type(dp.get("data_type"))
                        if key and skey and kind in ("object", "list_object"):
                            dp_key_to_struct[key] = (kind, skey)
                except Exception:
                    dp_key_to_struct = {}

                def try_parse_json(value: Any) -> Any:
                    s = str(value).strip()
                    if not s:
                        return None
                    if not (
                        s.startswith("{")
                        or s.startswith("[")
                        or s.startswith('"{')
                        or s.startswith('"[')
                    ):
                        return None
                    try:
                        import json

                        return json.loads(s)
                    except Exception:
                        try:
                            import ast

                            return ast.literal_eval(s)
                        except Exception:
                            return None

                def format_conf_percent(val: Any) -> str:
                    try:
                        if val is None or (isinstance(val, float) and pd.isna(val)):
                            return ""
                        num = float(val)
                        if 0.0 <= num <= 1.0:
                            num *= 100.0
                        if num < 0:
                            num = 0.0
                        if num > 100:
                            num = 100.0
                        return f"{int(round(num))}%"
                    except Exception:
                        s = str(val).strip()
                        return f"{s}%" if s and not s.endswith("%") else s

                for _, row in dfv.iterrows():
                    title = str(row.get("title", "")).strip() or str(row.get("key", "")).strip()
                    dp_key = str(row.get("key", "")).strip()
                    value_cell = row.get("value", "")
                    conf_str = format_conf_percent(row.get("confidence", ""))
                    explanation = str(row.get("explanation", "")).strip()

                    struct_info = dp_key_to_struct.get(dp_key)
                    if struct_info:
                        kind, skey = struct_info
                        # Heading for the datapoint
                        st.markdown(f"**{title}:**\n\n `{conf_str} {explanation}`")

                        el_titles = struct_key_to_el_title.get(skey, {})
                        parsed = try_parse_json(value_cell)

                        def render_one_object(
                            obj: Any, el_titles: dict[str, str] = el_titles
                        ) -> None:
                            if not isinstance(obj, dict):
                                return
                            keys_in_order = list(el_titles.keys()) or list(obj.keys())
                            for el_key in keys_in_order:
                                data = obj.get(el_key)
                                ttl = el_titles.get(el_key, el_key)
                                if isinstance(data, dict):
                                    val = data.get("value")
                                    c = format_conf_percent(data.get("confidence"))
                                    expl = str(data.get("explanation", "")).strip()
                                    st.markdown(f"- **{ttl}:** {val}\n\n`{c} {expl}`".strip())
                                else:
                                    st.markdown(f"- **{ttl}:** {data}")

                        if kind == "list_object" and isinstance(parsed, list):
                            for obj in parsed:
                                render_one_object(obj)
                                st.markdown("")
                        else:
                            render_one_object(parsed if isinstance(parsed, dict) else {})
                            st.markdown("")
                    else:
                        st.markdown(
                            f"**{title}**: {value_cell}\n\n`{conf_str} {explanation}`".strip()
                        )
                        # st.markdown("")

    with tab_processing:
        if not model_name:
            st.info("Select a model to view clause classification results.")
        else:
            df_cls = load_classification_csv(template_key, model_name, current_path.stem)
            if df_cls is None or df_cls.empty:
                st.info("No clause results found for this contract.")
            else:
                # Expect columns: index, clause_key, confidence (percent int), text
                text_series = df_cls.get("text", pd.Series(dtype=str))
                clause_series = df_cls.get("clause_key", pd.Series(dtype=str))
                # Map clause keys to titles from the template
                try:
                    tmpl = load_template_dict(template_key)
                    clauses = tmpl.get("clauses", []) or []
                    key_to_title = {
                        str(c.get("key")): c.get("title") or str(c.get("key")) for c in clauses
                    }
                except Exception:
                    key_to_title = {}
                clause_title_series = clause_series.map(lambda k: key_to_title.get(str(k), str(k)))
                conf_series = df_cls.get("confidence", pd.Series(dtype=object)).map(
                    lambda v: f"{int(v)}%" if pd.notna(v) and str(v).strip() != "" else ""
                )
                # Build clause cells: if same key as previous non-empty, show only (confidence) with left border
                keys_list = clause_series.tolist()
                titles_list = clause_title_series.tolist()
                conf_list = conf_series.tolist()
                repeat_flags: list[bool] = []
                prev_key: str | None = None
                for k in keys_list:
                    k_str = str(k).strip() if k is not None else ""
                    is_repeat = prev_key is not None and k_str != "" and k_str == prev_key
                    repeat_flags.append(bool(is_repeat))
                    if k_str != "":
                        prev_key = k_str

                clause_cells: list[str] = []
                text_cells: list[str] = []
                for is_rep, title, conf in zip(repeat_flags, titles_list, conf_list, strict=False):
                    conf_str = str(conf).strip()
                    if is_rep:
                        content = f"({conf_str})" if conf_str else ""
                        clause_cells.append(f'<div class="repeat-cell">{content}</div>')
                        # Mark text cell as repeated to draw right border
                        text_cells.append('<div class="text-repeat-cell">{}</div>')
                    else:
                        title_str = str(title).strip()
                        content = f"{title_str} ({conf_str})" if conf_str else title_str
                        clause_cells.append(f'<div class="normal-cell">{content}</div>')
                        text_cells.append('<div class="text-normal-cell">{}</div>')

                # Merge raw text into wrappers (avoid HTML breaking)
                safe_text = (
                    text_series.astype(str)
                    .str.replace("<", "&lt;")
                    .str.replace(">", "&gt;")
                    .str.replace("\t", "")
                )
                text_cells = [
                    tpl.format(txt)
                    for tpl, txt in zip(text_cells, safe_text.tolist(), strict=False)
                ]

                # Build datapoints column aligned by first evidence paragraph index
                n_rows = len(text_cells)
                dp_cells: list[str] = [""] * n_rows
                df_dp_for_clauses = load_datapoints_csv(template_key, model_name, current_path.stem)
                if df_dp_for_clauses is not None and not df_dp_for_clauses.empty:
                    try:
                        tmpl = load_template_dict(template_key)
                    except Exception:
                        tmpl = {}
                    dp_defs = (tmpl.get("datapoints") or []) if isinstance(tmpl, dict) else []
                    struct_defs = (tmpl.get("structures") or []) if isinstance(tmpl, dict) else []

                    def parse_structure_type(data_type: Any | None) -> tuple[str, str | None]:
                        s = str(data_type or "").strip().lower()
                        if s.startswith("list[") and "object:" in s:
                            inside = s[s.find("[") + 1 : s.rfind("]")]
                            if inside.startswith("object:"):
                                key = inside.split(":", 1)[1].strip().strip("[]")
                                return "list_object", key
                        if s.startswith("object:"):
                            return "object", s.split(":", 1)[1].strip().strip("[]")
                        return "simple", None

                    dp_key_to_struct: dict[str, tuple[str, str]] = {}
                    struct_key_to_el_title: dict[str, dict[str, str]] = {}
                    try:
                        for sd in struct_defs:
                            skey = str(sd.get("structure_key", "")).strip()
                            el_map: dict[str, str] = {}
                            for el in sd.get("elements") or []:
                                el_key = str(el.get("key", "")).strip()
                                el_title = str(el.get("title", el_key)).strip()
                                if el_key:
                                    el_map[el_key] = el_title
                            if skey:
                                struct_key_to_el_title[skey] = el_map
                    except Exception:
                        struct_key_to_el_title = {}
                    try:
                        for dp in dp_defs:
                            key = str(dp.get("key", "")).strip()
                            kind, skey = parse_structure_type(dp.get("data_type"))
                            if key and skey and kind in ("object", "list_object"):
                                dp_key_to_struct[key] = (kind, skey)
                    except Exception:
                        dp_key_to_struct = {}

                    def try_parse_json(value: Any) -> Any:
                        s = str(value).strip()
                        if not s:
                            return None
                        if not (
                            s.startswith("{")
                            or s.startswith("[")
                            or s.startswith('"{')
                            or s.startswith('"[')
                        ):
                            return None
                        try:
                            import json

                            return json.loads(s)
                        except Exception:
                            try:
                                import ast

                                return ast.literal_eval(s)
                            except Exception:
                                return None

                    def parse_evidence_field(value: Any) -> list[int]:
                        try:
                            import ast
                        except Exception:
                            ast = None  # type: ignore
                        if value is None or (isinstance(value, float) and pd.isna(value)):
                            return []
                        s = str(value).strip()
                        if s == "":
                            return []
                        try:
                            if "ast" in locals() and ast is not None:
                                parsed = ast.literal_eval(s)
                                if isinstance(parsed, (list, tuple)):
                                    out: list[int] = []
                                    for x in parsed:
                                        try:
                                            out.append(int(x))
                                        except Exception:
                                            continue
                                    return out
                                if isinstance(parsed, (int, float)):
                                    return [int(parsed)]
                        except Exception:
                            pass
                        tokens = s.replace("[", "").replace("]", "").split(",")
                        out: list[int] = []
                        for tok in tokens:
                            tok = tok.strip()
                            if not tok:
                                continue
                            try:
                                out.append(int(float(tok)))
                            except Exception:
                                continue
                        return out

                    for _, r in df_dp_for_clauses.iterrows():
                        dp_key = str(r.get("key", "")).strip()
                        title = str(r.get("title", "")).strip()
                        value_str = str(r.get("value", "")).strip()
                        conf_val = r.get("confidence", "")
                        conf_str = ""
                        try:
                            if pd.notna(conf_val) and str(conf_val).strip() != "":
                                conf_str = f"{int(conf_val)}%"
                        except Exception:
                            conf_str = str(conf_val).strip()
                        explanation = str(r.get("explanation", "")).strip()

                        struct_info = dp_key_to_struct.get(dp_key)
                        if struct_info:
                            kind, skey = struct_info
                            header_line = f"{title}: {conf_str} {explanation}".strip()
                            parts: list[str] = [header_line]
                            parsed = try_parse_json(r.get("value", ""))
                            el_titles = struct_key_to_el_title.get(skey, {})

                            def render_one_object(
                                obj: Any, el_titles: dict[str, str] = el_titles
                            ) -> list[str]:
                                lines: list[str] = []
                                if isinstance(obj, dict):
                                    keys_in_order = list(el_titles.keys()) or list(obj.keys())
                                    for el_key in keys_in_order:
                                        data = obj.get(el_key)
                                        if not isinstance(data, dict):
                                            val = str(data)
                                            ttl = el_titles.get(el_key, el_key)
                                            sub = f"{ttl}: {val}".strip()
                                            lines.append(sub)
                                            continue
                                        val = data.get("value")
                                        c = str(data.get("confidence", "")).strip()
                                        if c and not c.endswith("%"):
                                            try:
                                                c = f"{int(float(c)*100) if float(c) <= 1.0 else int(float(c))}%"
                                            except Exception:
                                                pass
                                        expl = data.get("explanation")
                                        ttl = el_titles.get(el_key, el_key)
                                        first = f"{ttl}: {val}".strip()
                                        second = f"{c} {expl}".strip()
                                        lines.append(first)
                                        if second:
                                            lines.append(second)
                                return lines

                            if kind == "list_object" and isinstance(parsed, list):
                                items_blocks: list[str] = []
                                for obj in parsed:
                                    block_lines = render_one_object(obj)
                                    if block_lines:
                                        items_blocks.append("<br/>".join(block_lines))
                                if items_blocks:
                                    parts.append("<br/><br/>".join(items_blocks))
                            else:
                                block_lines = render_one_object(
                                    parsed if isinstance(parsed, dict) else {}
                                )
                                if block_lines:
                                    parts.append("<br/>".join(block_lines))
                            content_html = "<br/>".join(parts)
                        else:
                            content_lines = [f"{title}: {value_str}"]
                            trail = f"{conf_str} {explanation}".strip()
                            if trail:
                                content_lines.append(trail)
                            content_html = "<br/>".join(content_lines)

                        ev = parse_evidence_field(r.get("evidence", ""))
                        idx = 0
                        if ev:
                            try:
                                idx = min(int(i) for i in ev)
                            except Exception:
                                idx = 0
                        if n_rows > 0:
                            idx = max(0, min(idx, n_rows - 1))
                        dp_cells[idx] = (
                            (dp_cells[idx] + f'<div class="dp-item">{content_html}</div>')
                            if dp_cells[idx]
                            else f'<div class="dp-item">{content_html}</div>'
                        )

                df_view = pd.DataFrame(
                    {
                        "text": pd.Series(text_cells),
                        "clause": clause_cells,
                        "datapoints": pd.Series(dp_cells),
                    }
                )
                # Render a single HTML table with fixed column widths and no borders; repeated rows show a right border on the text column
                html = df_view.to_html(index=False, escape=False)
                html = html.replace(
                    '<table border="1" class="dataframe">',
                    (
                        "<style>"
                        ".wrapped-table{width:100%;table-layout:fixed;border-collapse:collapse;}"
                        ".wrapped-table th,.wrapped-table td{border:none;padding:0.5rem;vertical-align:top;word-wrap:break-word;white-space:normal;}"
                        ".wrapped-table td .text-repeat-cell{border-right:4px solid #888;padding-right:0.5rem;}"
                        ".wrapped-table col.text-col{width:50%;}"
                        ".wrapped-table col.clause-col{width:25%;}"
                        ".wrapped-table col.dp-col{width:25%;}"
                        ".wrapped-table .dp-item{margin-bottom:0.5rem;padding-bottom:0.25rem;border-bottom:1px solid #eee;}"
                        "</style>"
                        '<table class="wrapped-table">'
                        "<colgroup>"
                        '<col class="text-col"/>'
                        '<col class="clause-col"/>'
                        '<col class="dp-col"/>'
                        "</colgroup>"
                    ),
                )
                st.markdown(html, unsafe_allow_html=True)

    with tab_review:
        if not model_name:
            st.info("Select a model to review guidelines.")
        else:
            # Load classification for clause labels
            df_cls = load_classification_csv(template_key, model_name, current_path.stem)
            # Load guidelines
            df_gl = load_guidelines_csv(template_key, model_name, current_path.stem)
            if df_cls is None or df_cls.empty:
                st.info("No clause results found for this contract.")
            elif df_gl is None or df_gl.empty:
                st.info("No guidelines results found for this contract.")
            else:
                # Prepare clause and text columns (same as Clauses tab)
                text_series = df_cls.get("text", pd.Series(dtype=str))
                clause_series = df_cls.get("clause_key", pd.Series(dtype=str))
                try:
                    tmpl = load_template_dict(template_key)
                    clauses = tmpl.get("clauses", []) or []
                    key_to_title = {
                        str(c.get("key")): c.get("title") or str(c.get("key")) for c in clauses
                    }
                except Exception:
                    key_to_title = {}
                clause_title_series = clause_series.map(lambda k: key_to_title.get(str(k), str(k)))
                conf_series = df_cls.get("confidence", pd.Series(dtype=object)).map(
                    lambda v: f"{int(v)}%" if pd.notna(v) and str(v).strip() != "" else ""
                )
                keys_list = clause_series.tolist()
                titles_list = clause_title_series.tolist()
                conf_list = conf_series.tolist()
                repeat_flags: list[bool] = []
                prev_key: str | None = None
                for k in keys_list:
                    k_str = str(k).strip() if k is not None else ""
                    is_repeat = prev_key is not None and k_str != "" and k_str == prev_key
                    repeat_flags.append(bool(is_repeat))
                    if k_str != "":
                        prev_key = k_str
                clause_cells: list[str] = []
                text_cells: list[str] = []
                for is_rep, title, conf in zip(repeat_flags, titles_list, conf_list, strict=False):
                    conf_str = str(conf).strip()
                    if is_rep:
                        content = f"({conf_str})" if conf_str else ""
                        clause_cells.append(f'<div class="repeat-cell">{content}</div>')
                        text_cells.append('<div class="text-repeat-cell">{}</div>')
                    else:
                        title_str = str(title).strip()
                        content = f"{title_str} ({conf_str})" if conf_str else title_str
                        clause_cells.append(f'<div class="normal-cell">{content}</div>')
                        text_cells.append('<div class="text-normal-cell">{}</div>')
                safe_text = (
                    text_series.astype(str)
                    .str.replace("<", "&lt;")
                    .str.replace(">", "&gt;")
                    .str.replace("\t", "")
                )
                text_cells = [
                    tpl.format(txt)
                    for tpl, txt in zip(text_cells, safe_text.tolist(), strict=False)
                ]

                # Build guidelines column aligned by first evidence paragraph index
                n_rows = len(text_cells)
                gl_cells: list[str] = [""] * n_rows

                def parse_evidence_field(value: Any) -> list[int]:
                    try:
                        import ast
                    except Exception:
                        ast = None  # type: ignore
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        return []
                    s = str(value).strip()
                    if s == "":
                        return []
                    try:
                        if "ast" in locals() and ast is not None:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, (list, tuple)):
                                out: list[int] = []
                                for x in parsed:
                                    try:
                                        out.append(int(x))
                                    except Exception:
                                        continue
                                return out
                            if isinstance(parsed, (int, float)):
                                return [int(parsed)]
                    except Exception:
                        pass
                    tokens = s.replace("[", "").replace("]", "").split(",")
                    out: list[int] = []
                    for tok in tokens:
                        tok = tok.strip()
                        if not tok:
                            continue
                        try:
                            out.append(int(float(tok)))
                        except Exception:
                            continue
                    return out

                for _, r in df_gl.iterrows():
                    guideline_text = str(r.get("guideline", "")).strip()
                    matched = str(r.get("guideline_matched", "")).strip()
                    conf_val = r.get("confidence", "")
                    try:
                        conf_str = (
                            f"{int(conf_val)}%"
                            if pd.notna(conf_val) and str(conf_val).strip() != ""
                            else ""
                        )
                    except Exception:
                        conf_str = str(conf_val).strip()
                    explanation = str(r.get("explanation", "")).strip()
                    header = guideline_text
                    status = f"Matched: {matched}" if matched != "" else ""
                    trail = " ".join([p for p in [status, conf_str, explanation] if p]).strip()
                    content_html = header if not trail else f"{header}<br/>{trail}"

                    ev = parse_evidence_field(r.get("evidence", ""))
                    idx = 0
                    if ev:
                        try:
                            idx = min(int(i) for i in ev)
                        except Exception:
                            idx = 0
                    if n_rows > 0:
                        idx = max(0, min(idx, n_rows - 1))
                    gl_cells[idx] = (
                        (gl_cells[idx] + f'<div class="dp-item">{content_html}</div>')
                        if gl_cells[idx]
                        else f'<div class="dp-item">{content_html}</div>'
                    )

                df_view = pd.DataFrame(
                    {
                        "text": pd.Series(text_cells),
                        "clause": clause_cells,
                        "guidelines": pd.Series(gl_cells),
                    }
                )
                html = df_view.to_html(index=False, escape=False)
                html = html.replace(
                    '<table border="1" class="dataframe">',
                    (
                        "<style>"
                        ".wrapped-table{width:100%;table-layout:fixed;border-collapse:collapse;}"
                        ".wrapped-table th,.wrapped-table td{border:none;padding:0.5rem;vertical-align:top;word-wrap:break-word;white-space:normal;}"
                        ".wrapped-table td .text-repeat-cell{border-right:4px solid #888;padding-right:0.5rem;}"
                        ".wrapped-table col.text-col{width:50%;}"
                        ".wrapped-table col.clause-col{width:25%;}"
                        ".wrapped-table col.dp-col{width:25%;}"
                        ".wrapped-table .dp-item{margin-bottom:0.5rem;padding-bottom:0.25rem;border-bottom:1px solid #eee;}"
                        "</style>"
                        '<table class="wrapped-table">'
                        "<colgroup>"
                        '<col class="text-col"/>'
                        '<col class="clause-col"/>'
                        '<col class="dp-col"/>'
                        "</colgroup>"
                    ),
                )
                st.markdown(html, unsafe_allow_html=True)

    with tab_clauses:
        if not model_name:
            st.info("Select a model to view clauses.")
        else:
            df_cls = load_classification_csv(template_key, model_name, current_path.stem)
            if df_cls is None or df_cls.empty:
                st.info("No clause results found for this contract.")
            else:
                # Map clause keys to titles
                try:
                    tmpl = load_template_dict(template_key)
                    clauses = tmpl.get("clauses", []) or []
                    key_to_title = {
                        str(c.get("key")): c.get("title") or str(c.get("key")) for c in clauses
                    }
                except Exception:
                    key_to_title = {}

                # Group paragraphs by clause_key in appearance order
                grouped: dict[str, list[str]] = {}
                for _, row in df_cls.iterrows():
                    k = str(row.get("clause_key", "")).strip()
                    if not k:
                        continue
                    txt = str(row.get("text", "")).strip()
                    if not txt:
                        continue
                    grouped.setdefault(k, []).append(txt)

                if not grouped:
                    st.info("No classified clauses found in this document.")
                else:
                    for ck, para_list in grouped.items():
                        title = key_to_title.get(ck, ck)
                        st.markdown(f"### {title}")
                        for p in para_list:
                            st.markdown(f"- {p}")


if __name__ == "__main__":
    main()
