from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from contract_ai_core.utilities import text_to_paragraphs


def get_repo_root() -> Path:
    # Find the 'dataset' directory in parents, then return its parent
    here = Path(__file__).resolve()
    dataset_dir: Optional[Path] = None
    for p in here.parents:
        if p.name == "dataset":
            dataset_dir = p
            break
    return dataset_dir.parent if dataset_dir else here.parents[2]


def list_contract_types() -> list[str]:
    repo_root = get_repo_root()
    base = repo_root / "dataset" / "output" / "filter"
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    except Exception:
        return []


def list_models(contract_type: str) -> list[str]:
    repo_root = get_repo_root()
    base = repo_root / "dataset" / "output" / "filter" / contract_type
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()]) if base.exists() else []
    except Exception:
        return []


def list_filter_files(contract_type: str, model_name: str) -> list[Path]:
    repo_root = get_repo_root()
    base = repo_root / "dataset" / "output" / "filter" / contract_type / model_name
    try:
        return sorted(base.glob("*.json")) if base.exists() else []
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


def main() -> None:
    st.set_page_config(page_title="Contract Filter", layout="wide")
    st.title("Contract Filter")

    # Sidebar: select contract type and model
    st.sidebar.header("Selection")
    contract_types = list_contract_types()
    if not contract_types:
        st.info("No filter outputs found under dataset/output/filter/")
        return
    if "filter_contract_type" not in st.session_state:
        st.session_state.filter_contract_type = contract_types[0]
    contract_type = st.sidebar.selectbox(
        "Contract type",
        contract_types,
        index=contract_types.index(st.session_state.get("filter_contract_type", contract_types[0])),
    )
    if contract_type != st.session_state.get("filter_contract_type"):
        st.session_state.filter_contract_type = contract_type
        st.session_state.filter_model = None
        st.session_state.filter_idx = 0

    models = list_models(contract_type)
    if not models:
        st.sidebar.info(f"No models found under output/filter/{contract_type}")
        return
    if "filter_model" not in st.session_state or st.session_state.filter_model not in models:
        st.session_state.filter_model = models[0]
    model_name = st.sidebar.selectbox(
        "Model", models, index=models.index(st.session_state.get("filter_model", models[0]))
    )
    if model_name != st.session_state.get("filter_model"):
        st.session_state.filter_model = model_name
        st.session_state.filter_idx = 0

    # Files from output/filter/<model_name>
    files = list_filter_files(contract_type, model_name)
    if not files:
        st.warning(f"No outputs found in dataset/output/filter/{contract_type}/{model_name}/")
        return

    if "filter_idx" not in st.session_state:
        st.session_state.filter_idx = 0
    idx = int(st.session_state.filter_idx) % len(files)
    current_json = files[idx]

    # Load current result JSON
    try:
        data = json.loads(current_json.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read {current_json.name}: {e}")
        return

    filename = str(data.get("filename") or Path(current_json.stem).name)
    stem = Path(filename).stem

    col_title, col_prev, col_next = st.columns([8, 1, 1])
    with col_title:
        st.subheader(
            f"Type: {contract_type} — Model: {model_name} — File {idx + 1} / {len(files)}: {filename}"
        )
    with col_prev:
        if st.button("◀ Previous"):
            st.session_state.filter_idx = (int(st.session_state.filter_idx) - 1) % len(files)
            st.rerun()
    with col_next:
        if st.button("Next ▶"):
            st.session_state.filter_idx = (int(st.session_state.filter_idx) + 1) % len(files)
            st.rerun()

    tab_contract, tab_scopes, tab_filtered, tab_text = st.tabs(
        ["Contract", "Scopes", "Filtered", "Full text"]
    )

    # Contract tab: mirror 20_Contracts.py Contract tab (PNG + gold organizer metadata)
    with tab_contract:
        repo_root = get_repo_root()
        pngs_dir = repo_root / "dataset" / "documents" / "organizer" / "pngs"
        gold_dir = repo_root / "dataset" / "gold" / "organizer"

        col_left, col_right = st.columns([1, 1])
        with col_left:
            png_candidates = [pngs_dir / f"{stem}.png", pngs_dir / f"{filename}.png"]
            shown = False
            for pc in png_candidates:
                if pc.exists():
                    st.image(str(pc))
                    shown = True
                    break
            if not shown:
                st.caption("No PNG preview found for this document.")

        with col_right:
            st.subheader("Metadata")
            json_path = gold_dir / f"{stem}.json"
            if not json_path.exists():
                st.info("Gold organizer JSON not found for this document.")
            else:
                try:
                    meta = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception as e:
                    st.error(f"Failed to read gold JSON: {e}")
                    meta = {}

                def _fmt_pct(val: float | None) -> str:
                    try:
                        return f"{int(round(float(val) * 100))}%" if val is not None else ""
                    except Exception:
                        return ""

                def _show_field(title: str, key: str) -> None:
                    node = meta.get(key) or {}
                    node = node if isinstance(node, dict) else {}
                    value = node.get("value") or ""
                    conf = node.get("confidence")
                    expl = node.get("explanation") or ""
                    st.markdown(f"**{title}:** {value}")
                    meta_pct = _fmt_pct(conf)
                    st.caption(
                        ((f"confidence: {meta_pct}") if meta_pct else "")
                        + (f" — {expl}" if expl else "")
                    )

                _show_field("Contract type", "contract_type")
                _show_field("Contract type version", "contract_type_version")
                _show_field("Contract date", "contract_date")
                _show_field("Amendment date", "amendment_date")
                _show_field("Amendment number", "amendment_number")
                _show_field("Version type", "version_type")
                _show_field("Status", "status")
                _show_field("Party name 1", "party_name_1")
                _show_field("Party role 1", "party_role_1")
                _show_field("Party name 2", "party_name_2")
                _show_field("Party role 2", "party_role_2")
                _show_field("Document quality", "document_quality")

    # Scopes tab: show scope results from filter output
    with tab_scopes:
        scopes = data.get("scopes") or []
        if not isinstance(scopes, list) or not scopes:
            st.info("No scopes found in filter output.")
        else:
            st.subheader("Filtering scopes")
            for sc in scopes:
                name = str((sc or {}).get("name", "")).strip()
                start = (sc or {}).get("start_line")
                end = (sc or {}).get("end_line")
                conf = (sc or {}).get("confidence")
                expl = str((sc or {}).get("explanation") or "").strip()
                try:
                    conf_str = f"{int(round(float(conf) * 100))}%" if conf is not None else ""
                except Exception:
                    conf_str = ""
                meta = " ".join([p for p in [conf_str, expl] if p]).strip()
                st.markdown(f"- **{name}**: {start} — {end}" + (f"\n\n`{meta}`" if meta else ""))

    # Text tab: show original document text
    with tab_text:
        repo_root = get_repo_root()
        txt_dir = repo_root / "dataset" / "documents" / "organizer" / "files"
        txt_path = txt_dir / filename
        if not txt_path.exists():
            st.info("Source text not found under dataset/documents/organizer/files/.")
        else:
            raw = read_text_best_effort(txt_path)
            # Try to use core paragraph splitter; fallback to blank-line split
            paras: list[tuple[int, str]] = []
            try:
                repo_root = get_repo_root()
                src_dir = repo_root / "src"
                if str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))

                items = text_to_paragraphs(raw)
                paras = [(p.index, p.text) for p in items]
            except Exception:
                # Fallback simple split on double newlines
                blocks = [b.strip() for b in raw.split("\n\n")]
                paras = [(i, t) for i, t in enumerate([b for b in blocks if b])]

            numbered = "\n".join([f"{i}: {t}" for i, t in paras])
            st.text_area("Document text", value=numbered, height=700)

    # Filtered tab: show paragraphs for each scope with meta
    with tab_filtered:
        scopes = data.get("scopes") or []
        if not isinstance(scopes, list) or not scopes:
            st.info("No scopes found in filter output.")
        else:
            repo_root = get_repo_root()
            txt_dir = repo_root / "dataset" / "documents" / "organizer" / "files"
            txt_path = txt_dir / filename
            if not txt_path.exists():
                st.info("Source text not found under dataset/documents/organizer/files/.")
            else:
                raw = read_text_best_effort(txt_path)
                # Build paragraphs with indices
                paras_text: list[str] = []
                try:
                    src_dir = repo_root / "src"
                    if str(src_dir) not in sys.path:
                        sys.path.insert(0, str(src_dir))

                    items = text_to_paragraphs(raw)
                    max_idx = -1
                    if items:
                        max_idx = max(p.index for p in items)
                    paras_text = []
                    # Create dense index array to allow direct access by index
                    dense: dict[int, str] = {p.index: p.text for p in items}
                    for i in range(0, max_idx + 1):
                        paras_text.append(dense.get(i, ""))
                except Exception:
                    blocks = [b.strip() for b in raw.split("\n\n")]
                    paras_text = [b for b in blocks if b]

                for sc in scopes:
                    name = str((sc or {}).get("name", "")).strip()
                    start = (sc or {}).get("start_line")
                    end = (sc or {}).get("end_line")
                    conf = (sc or {}).get("confidence")
                    expl = str((sc or {}).get("explanation") or "").strip()
                    try:
                        conf_str = f"{int(round(float(conf) * 100))}%" if conf is not None else ""
                    except Exception:
                        conf_str = ""

                    st.markdown(f"### {name}")
                    meta = " ".join([p for p in [conf_str, expl] if p]).strip()
                    if meta:
                        st.caption(meta)

                    # Clamp and render paragraphs same as Text tab (numbered lines)
                    try:
                        s = int(start) if start is not None else -1
                    except Exception:
                        s = -1
                    try:
                        e = int(end) if end is not None else -1
                    except Exception:
                        e = -1
                    if s < 0 or e < s:
                        st.info("No paragraph span identified for this scope.")
                        continue
                    s = max(0, s)
                    e = min(e, len(paras_text) - 1)
                    numbered = "\n".join(
                        [
                            f"{i}: {paras_text[i]}"
                            for i in range(s, e + 1)
                            if 0 <= i < len(paras_text) and paras_text[i]
                        ]
                    )
                    st.text_area(f"{name} span", value=numbered, height=400)

    # (Removed JSON tab)


if __name__ == "__main__":
    main()
