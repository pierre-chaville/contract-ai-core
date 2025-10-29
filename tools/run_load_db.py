from __future__ import annotations

"""
Load CONTRACTS table from dataset files.

Sources:
- Metadata: dataset/gold/organizer/*.json
- Full text: dataset/documents/organizer/files/<filename>.txt
  - Optional filter: dataset/gold/filter/<contract_type>/<filename>.json
    Keep only paragraphs specified in "scopes" (start_line to end_line, inclusive)
- Clauses: dataset/gold/clauses/<contract_type>/<filename>.csv
- Datapoints: dataset/gold/datapoints/<contract_type>/<filename>.csv

Usage:
  python tools/run_load_db.py [--dry-run] [--echo]
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# Mirror the CONTRACTS table definition from run_create_db.py to avoid cross-imports.
class Base(DeclarativeBase):
    pass


class ContractRecord(Base):
    __tablename__ = "CONTRACTS"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contract_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    contract_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_type_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_amendment_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    number_amendments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    party_name_1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_role_1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_name_2: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_role_2: Mapped[str | None] = mapped_column(String(255), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    business_purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    clauses_text: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    list_clauses: Mapped[list] = mapped_column(JSON, nullable=True, default=list)
    datapoints: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    guidelines: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


def _sqlite_url(db_path: str) -> str:
    abs_path = os.path.abspath(db_path)
    return f"sqlite:///{abs_path}"


def _ensure_parent_directory(db_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_filter(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_scopes_to_text(text: str, scopes: List[dict]) -> str:
    # Interpret text as lines; select the ranges defined by scopes
    # Each scope has start_line and end_line (1-based or 0-based?)
    # We will treat them as 1-based inclusive for typical human-generated files.
    lines = text.splitlines()
    kept: List[str] = []
    for s in scopes:
        start_line = int(s.get("start_line", 1))
        end_line = int(s.get("end_line", len(lines)))
        # Convert 1-based to 0-based indices and clamp
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines) - 1, end_line - 1)
        if start_idx <= end_idx:
            kept.extend(lines[start_idx : end_idx + 1])
    return "\n".join(kept)


def _read_clauses_csv(path: Path) -> Dict[str, str]:
    # Input rows represent paragraphs: index, clause_key, (optional) clause_title, confidence, text
    # We aggregate paragraph texts per clause TITLE when available; fallback to key if title missing
    print(path)
    if not path.exists():
        return {}
    aggregated: Dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        columns = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            columns.get("clause_key")
            or columns.get("clause_id")
            or columns.get("key")
            or columns.get("clause")
        )
        # Prefer an explicit human-readable title
        title_col = (
            columns.get("title")
            or columns.get("clause_title")
            or columns.get("name")
            or columns.get("clause_name")
            or columns.get("label")
        )
        text_col = columns.get("text") or columns.get("clause_text") or columns.get("value")
        for row in reader:
            clause_key = (row.get(key_col or "") or "").strip()
            title_val = (row.get(title_col or "") or "").strip()
            paragraph_text = (row.get(text_col or "") or "").strip()
            if not paragraph_text:
                continue
            # Skip rows with neither key nor title
            if not clause_key and not title_val:
                continue
            agg_key = title_val or clause_key
            aggregated.setdefault(agg_key, []).append(paragraph_text)
    # Join paragraphs per clause with newlines, keyed by TITLE
    return {k: "\n".join(v) for k, v in aggregated.items()}


def _read_clause_titles(path: Path) -> List[str]:
    """Return the list of clause TITLES present in the clauses CSV (fallback to keys)."""
    if not path.exists():
        return []
    titles: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        columns = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            columns.get("clause_key")
            or columns.get("clause_id")
            or columns.get("key")
            or columns.get("clause")
        )
        title_col = (
            columns.get("title")
            or columns.get("clause_title")
            or columns.get("name")
            or columns.get("clause_name")
            or columns.get("label")
        )
        seen = set()
        for row in reader:
            title_val = (row.get(title_col or "") or "").strip()
            clause_key = (row.get(key_col or "") or "").strip()
            value = title_val or clause_key
            if value and value not in seen:
                titles.append(value)
                seen.add(value)
    return titles


def _read_clause_keys(path: Path) -> List[str]:
    """Return the list of clause KEYS present in the clauses CSV."""
    if not path.exists():
        return []
    keys: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        columns = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            columns.get("clause_key")
            or columns.get("clause_id")
            or columns.get("key")
            or columns.get("clause")
        )
        for row in reader:
            clause_key = (row.get(key_col or "") or "").strip()
            if clause_key and clause_key not in seen:
                keys.append(clause_key)
                seen.add(clause_key)
    return keys


def _load_clause_title_map(dataset_root: Path, contract_type: Optional[str]) -> Dict[str, str]:
    """Load a mapping of clause_key -> title from dataset/contract_types/<contract_type>_clauses.csv.

    Returns an empty dict if no mapping file is available.
    """
    if not contract_type:
        return {}
    map_path = dataset_root / "contract_types" / f"{contract_type}_clauses.csv"
    if not map_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        cols = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            cols.get("clause_key")
            or cols.get("clause_id")
            or cols.get("key")
            or cols.get("id")
            or cols.get("clause")
        )
        title_col = (
            cols.get("title") or cols.get("clause_title") or cols.get("name") or cols.get("label")
        )
        for row in reader:
            key_val = (row.get(key_col or "") or "").strip()
            title_val = (row.get(title_col or "") or "").strip()
            if key_val and title_val:
                mapping[key_val] = title_val
    return mapping


def _read_clauses_csv_with_map(path: Path, title_map: Dict[str, str]) -> Dict[str, str]:
    """Aggregate clause texts by TITLE using a provided title map; fallback to row title, then key."""
    if not path.exists():
        return {}
    aggregated: Dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        columns = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            columns.get("clause_key")
            or columns.get("clause_id")
            or columns.get("key")
            or columns.get("clause")
        )
        title_col = (
            columns.get("title")
            or columns.get("clause_title")
            or columns.get("name")
            or columns.get("clause_name")
            or columns.get("label")
        )
        text_col = columns.get("text") or columns.get("clause_text") or columns.get("value")
        for row in reader:
            clause_key = (row.get(key_col or "") or "").strip()
            row_title = (row.get(title_col or "") or "").strip()
            paragraph_text = (row.get(text_col or "") or "").strip()
            if not paragraph_text:
                continue
            if not clause_key and not row_title:
                continue
            mapped_title = title_map.get(clause_key, "") if clause_key else ""
            agg_key = mapped_title or row_title or clause_key
            aggregated.setdefault(agg_key, []).append(paragraph_text)
    return {k: "\n".join(v) for k, v in aggregated.items()}


def _read_clause_titles_with_map(path: Path, title_map: Dict[str, str]) -> List[str]:
    if not path.exists():
        return []
    titles: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        columns = {name.lower(): name for name in reader.fieldnames}
        key_col = (
            columns.get("clause_key")
            or columns.get("clause_id")
            or columns.get("key")
            or columns.get("clause")
        )
        title_col = (
            columns.get("title")
            or columns.get("clause_title")
            or columns.get("name")
            or columns.get("clause_name")
            or columns.get("label")
        )
        for row in reader:
            clause_key = (row.get(key_col or "") or "").strip()
            row_title = (row.get(title_col or "") or "").strip()
            title = title_map.get(clause_key) if clause_key else None
            value = title or row_title or clause_key
            if value and value not in seen:
                titles.append(value)
                seen.add(value)
    return titles


def _read_datapoints_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    result: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect columns like: datapoint_id, value
        fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
        id_col = fieldnames.get("datapoint_id") or fieldnames.get("key") or fieldnames.get("id")
        val_col = fieldnames.get("value") or fieldnames.get("val") or fieldnames.get("text")
        for row in reader:
            did = (row.get(id_col or "") or "").strip()
            val = row.get(val_col or "")
            if did:
                result[did] = val
    return result


def _discover_organizer_jsons(dataset_root: Path) -> List[Path]:
    return sorted((dataset_root / "gold" / "organizer").glob("*.json"))


def _normalize_contract_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("key", "code", "name", "type"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        # last resort: any first str value
        for v in value.values():
            if isinstance(v, str) and v:
                return v
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return _normalize_contract_type(first)
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _load_metadata_from_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect a top-level metadata-like structure; be permissive: flatten known fields
    # Example keys used by DB columns
    mapping = {
        "contract_id": None,
        "contract_number": None,
        "contract_type": None,
        "contract_type_version": None,
        "contract_date": None,
        "last_amendment_date": None,
        "number_amendments": None,
        "status": None,
        "party_name_1": None,
        "party_role_1": None,
        "party_name_2": None,
        "party_role_2": None,
        "department": None,
        "contract_owner": None,
        "business_purpose": None,
        "filename": None,
    }
    for key in list(mapping.keys()):
        if key in data:
            mapping[key] = data[key]
        elif "metadata" in data and isinstance(data["metadata"], dict) and key in data["metadata"]:
            mapping[key] = data["metadata"][key]

    # filename may be nested under document
    if not mapping.get("filename") and isinstance(data.get("document"), dict):
        doc = data["document"]
        for k in ("filename", "file", "name"):
            if isinstance(doc.get(k), str):
                mapping["filename"] = doc[k]
                break

    # normalize specific fields
    mapping["contract_type"] = _normalize_contract_type(mapping.get("contract_type"))
    mapping["number_amendments"] = _to_int(mapping.get("number_amendments"))
    return mapping


def _resolve_paths(dataset_root: Path, contract_type: Optional[str], filename: str) -> dict:
    doc_txt = dataset_root / "documents" / "organizer" / "files" / f"{filename}.txt"
    ct_dir = (contract_type or "").strip()
    filt_json = dataset_root / "gold" / "filter" / ct_dir / f"{filename}.json"
    clauses_csv = dataset_root / "gold" / "clauses" / ct_dir / f"{filename}.csv"
    datapoints_csv = dataset_root / "gold" / "datapoints" / ct_dir / f"{filename}.csv"
    return {
        "doc_txt": doc_txt,
        "filt_json": filt_json,
        "clauses_csv": clauses_csv,
        "datapoints_csv": datapoints_csv,
    }


def upsert_contract(session: Session, payload: dict, dry_run: bool = False) -> None:
    # Try to find by contract_id if available, else by filename fallback
    contract_id = payload.get("contract_id") or payload.get("filename")
    if contract_id:
        existing = session.scalar(
            select(ContractRecord).where(ContractRecord.contract_id == str(contract_id))
        )
    else:
        existing = None

    if existing:
        for key, value in payload.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
    else:
        rec = ContractRecord(**payload)
        session.add(rec)

    if not dry_run:
        session.commit()


def load_all(dry_run: bool = False, echo: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "dataset"
    db_path = dataset_root / "contracts.sqlite"

    engine = create_engine(_sqlite_url(str(db_path)), echo=echo, future=True)
    Base.metadata.create_all(bind=engine)

    organizer_jsons = _discover_organizer_jsons(dataset_root)

    with Session(engine) as session:
        for org_path in organizer_jsons:
            metadata = _load_metadata_from_json(org_path)
            filename = metadata.get("filename") or org_path.stem
            filename = filename.replace(".pdf", "").replace(".PDF", "")
            contract_type = _normalize_contract_type(metadata.get("contract_type"))
            paths = _resolve_paths(dataset_root, contract_type, filename.replace(".txt", ""))
            print("--------------------------------")
            print("paths", paths)
            full_text = ""
            if paths["doc_txt"].exists():
                text = _read_text_file(paths["doc_txt"])
                filter_obj = (
                    _load_filter(paths["filt_json"]) if paths["filt_json"].exists() else None
                )
                if filter_obj and isinstance(filter_obj, dict):
                    scopes = filter_obj.get("scopes") or []
                    if scopes:
                        text = _apply_scopes_to_text(text, scopes)
                full_text = text

            # Load mapping for clause keys -> titles by contract_type
            title_map = _load_clause_title_map(dataset_root, contract_type)
            clauses_text = (
                _read_clauses_csv_with_map(paths["clauses_csv"], title_map)
                if paths["clauses_csv"].exists()
                else {}
            )
            list_clauses = (
                _read_clause_keys(paths["clauses_csv"]) if paths["clauses_csv"].exists() else []
            )
            datapoints = (
                _read_datapoints_csv(paths["datapoints_csv"])
                if paths["datapoints_csv"].exists()
                else {}
            )

            def _unwrap_value(v: Any) -> Any:
                # Common pattern {'value': X, ...}
                if isinstance(v, dict):
                    if "value" in v:
                        return v["value"]
                    # fallback to first scalar string
                    for k in ("text", "name", "label"):
                        if isinstance(v.get(k), str):
                            return v[k]
                return v

            def _to_str(v: Any) -> Optional[str]:
                if v is None:
                    return None
                v = _unwrap_value(v)
                if v is None:
                    return None
                return str(v)

            payload = {
                # metadata fields mapped directly
                "contract_id": _to_str(metadata.get("contract_id") or filename),
                "contract_number": _to_str(metadata.get("contract_number")),
                "contract_type": _to_str(contract_type),
                "contract_type_version": _to_str(metadata.get("contract_type_version")),
                "contract_date": _to_str(metadata.get("contract_date")),
                "last_amendment_date": _to_str(metadata.get("last_amendment_date")),
                "number_amendments": _to_int(_unwrap_value(metadata.get("number_amendments"))),
                "status": _to_str(metadata.get("status")),
                "party_name_1": _to_str(metadata.get("party_name_1")),
                "party_role_1": _to_str(metadata.get("party_role_1")),
                "party_name_2": _to_str(metadata.get("party_name_2")),
                "party_role_2": _to_str(metadata.get("party_role_2")),
                "department": _to_str(metadata.get("department")),
                "contract_owner": _to_str(metadata.get("contract_owner")),
                "business_purpose": _to_str(metadata.get("business_purpose")),
                # content
                "full_text": full_text or "",
                "clauses_text": clauses_text,
                "list_clauses": list_clauses,
                "datapoints": datapoints,
                # guidelines left empty for now; no source provided in request
                "guidelines": {},
            }

            upsert_contract(session, payload, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CONTRACTS table from dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB.")
    parser.add_argument("--echo", action="store_true", help="Enable SQL echo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_all(dry_run=args.dry_run, echo=args.echo)


if __name__ == "__main__":
    main()
