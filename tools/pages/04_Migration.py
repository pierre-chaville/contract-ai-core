from __future__ import annotations

import json

# Streamlit viewer for organizer results (moved under pages/).
# Browse results.csv by model, inspect original text and extracted metadata.
# Appears in the Streamlit sidebar when running: streamlit run dataset/app.py
from pathlib import Path
from typing import Optional

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


def get_organizer_models() -> list[str]:
    repo_root = get_repo_root()
    base_dir = repo_root / "dataset" / "output" / "organizer"
    base_dir.mkdir(parents=True, exist_ok=True)
    models: list[str] = []
    try:
        for p in base_dir.iterdir():
            if p.is_dir():
                has_json = any(child.suffix.lower() == ".json" for child in p.iterdir())
                if has_json:
                    models.append(p.name)
    except Exception:
        pass
    return sorted(models)


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


def find_source_file(filename: str) -> Optional[Path]:
    repo_root = get_repo_root()
    # Search under dataset/documents/organizer recursively; support .txt and .md
    base_dir = repo_root / "dataset" / "documents" / "organizer"
    if not base_dir.exists():
        return None
    # Try exact match in any nested folder
    for p in base_dir.rglob(filename):
        if p.is_file():
            return p
    # Fallback: try stem match across common extensions
    stem = Path(filename).stem
    for ext in (".txt", ".md"):
        for p in base_dir.rglob(stem + ext):
            if p.is_file():
                return p
    return None


def format_pct(conf: float | None) -> str:
    try:
        return f"{int(round(float(conf) * 100))}%" if conf is not None else ""
    except Exception:
        return ""


def main() -> None:
    st.set_page_config(page_title="Contract Organizer", layout="wide")
    st.title("Contract Organizer")

    # Sidebar: model selector
    st.sidebar.header("Selection")
    models = get_organizer_models()
    if "model_name" not in st.session_state and models:
        st.session_state.model_name = models[0]
    if models:
        default_index = (
            models.index(st.session_state.get("model_name", models[0]))
            if st.session_state.get("model_name") in models
            else 0
        )
        selected_model = st.sidebar.selectbox("Model", models, index=default_index)
        if selected_model != st.session_state.get("model_name"):
            st.session_state.model_name = selected_model
            st.session_state.file_idx = 0
    else:
        st.sidebar.info("No model folders found under dataset/output/organizer/")

    model_name = st.session_state.get("model_name")
    if not model_name:
        st.info("Select a model to begin.")
        return

    repo_root = get_repo_root()
    model_dir = repo_root / "dataset" / "output" / "organizer" / model_name
    json_files = sorted(model_dir.glob("*.json"))
    if not json_files:
        st.info("No JSON outputs found for this model.")
        return

    # Navigation
    if "file_idx" not in st.session_state:
        st.session_state.file_idx = 0

    num_files = len(json_files)
    current_idx = int(st.session_state.file_idx) % num_files
    current_json = json_files[current_idx]
    try:
        with current_json.open("r", encoding="utf-8") as jf:
            data = json.load(jf)
    except Exception as exc:
        st.error(f"Failed to read {current_json.name}: {exc}")
        return
    filename = str(data.get("filename") or current_json.stem)

    col_title, col_prev, col_next = st.columns([8, 1, 1])
    with col_title:
        st.subheader(f"Model: {model_name} — File {current_idx + 1} / {num_files}: {filename}")
    with col_prev:
        if st.button("◀ Previous"):
            st.session_state.file_idx = (int(st.session_state.file_idx) - 1) % num_files
            st.rerun()
    with col_next:
        if st.button("Next ▶"):
            st.session_state.file_idx = (int(st.session_state.file_idx) + 1) % num_files
            st.rerun()

    # Two columns: left text, right extracted fields
    left, right = st.columns(2)

    with left:
        # Optional image preview before text
        pngs_dir = repo_root / "dataset" / "documents" / "organizer" / "pngs"
        png_candidates = [
            pngs_dir / f"{Path(filename).stem}.png",
            pngs_dir / f"{filename}.png",
        ]
        for pc in png_candidates:
            if pc.exists():
                st.image(str(pc))
                break
        src_path = find_source_file(filename)
        if not src_path:
            st.warning("Original file not found under dataset/documents/organizer.")
        else:
            text = read_text_best_effort(src_path)
            st.text_area("Document text", value=text, height=700)

    with right:
        st.subheader("Extracted fields")

        def show_field(title: str, field_key: str) -> None:
            node = data.get(field_key) or {}
            if not isinstance(node, dict):
                node = {}
            value = node.get("value") or ""
            conf = node.get("confidence")
            conf_pct = format_pct(conf)
            expl = node.get("explanation") or ""
            st.markdown(f"**{title}:** {value}")
            st.caption(
                (f"confidence: {conf_pct}" if conf_pct else "") + (f" — {expl}" if expl else "")
            )

        show_field("Version type", "version_type")
        show_field("Contract type", "contract_type")
        show_field("Contract type version", "contract_type_version")
        show_field("Contract date", "contract_date")
        if data["version_type"]["value"] == "AMENDMENT":
            show_field("Amendment date", "amendment_date")
            show_field("Amendment number", "amendment_number")
        show_field("Status", "status")
        show_field("Party name 1", "party_name_1")
        show_field("Party role 1", "party_role_1")
        show_field("Party name 2", "party_name_2")
        show_field("Party role 2", "party_role_2")
        show_field("Document quality", "document_quality")


if __name__ == "__main__":
    main()
