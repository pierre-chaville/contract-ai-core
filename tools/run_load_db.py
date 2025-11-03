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
import ast
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    list_datapoints: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    list_guidelines: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


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


# === Datapoints enrichment (titles, enums, objects) ===


def _load_enums(dataset_root: Path) -> Dict[str, Dict[str, str]]:
    """Load enum sets from contract_types/ISDA_enums.csv.

    Returns mapping: enum_set_name -> { key -> title }
    """
    path = dataset_root / "contract_types" / "ISDA_enums.csv"
    enums: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        print("no enums file", path)
        return enums
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("no fieldnames", path)
            return enums
        cols = {str(n).strip().lstrip("\ufeff").lower(): n for n in reader.fieldnames}
        # ISDA_enums.csv uses: key (enum set), title (set title), code (enum key), description (enum item title)
        set_col = (
            cols.get("enum")
            or cols.get("set")
            or cols.get("group")
            or cols.get("set_name")
            or cols.get("key")
        )
        code_col = cols.get("code") or cols.get("enum_code") or cols.get("id")
        title_col = cols.get("description") or cols.get("title") or cols.get("label")
        for row in reader:
            set_name = (row.get(set_col or "") or "").strip()
            key = (row.get(code_col or "") or "").strip()
            title = (row.get(title_col or "") or "").strip()
            if not set_name or not key:
                continue
            if set_name not in enums:
                enums[set_name] = {}
            enums[set_name][key] = title or key
    return enums


def _load_datapoint_definitions(
    dataset_root: Path, contract_type: Optional[str]
) -> Dict[str, Tuple[str, str, str]]:
    """Load datapoint definitions: id -> (title, data_type, enum_key).

    Attempts contract_types/<contract_type>_datapoints.csv, else ISDA_datapoints.csv.
    """
    candidates: List[Path] = []
    if contract_type:
        ct = str(contract_type).strip()
        candidates.append(dataset_root / "contract_types" / f"{ct}_datapoints.csv")
    candidates.append(dataset_root / "contract_types" / "ISDA_datapoints.csv")
    defs: Dict[str, Tuple[str, str, str]] = {}
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return defs
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return defs
        cols = {n.lower(): n for n in reader.fieldnames}
        id_col = cols.get("key") or cols.get("id") or cols.get("datapoint_id")
        title_col = cols.get("title") or cols.get("label") or cols.get("name")
        type_col = cols.get("data_type") or cols.get("type")
        enum_col = cols.get("enum_key") or cols.get("enum") or cols.get("enum_set")
        for row in reader:
            did = (row.get(id_col or "") or "").strip()
            title = (row.get(title_col or "") or did).strip()
            dtype = (row.get(type_col or "") or "").strip()
            enum_key = (row.get(enum_col or "") or "").strip()
            if did:
                defs[did] = (title, dtype, enum_key)
    return defs


def _load_structure_definitions(dataset_root: Path) -> Dict[str, Dict[str, Tuple[str, str, str]]]:
    """Load structures and their elements: struct -> field_key -> (title, data_type, enum_key)."""
    elems_path = dataset_root / "contract_types" / "ISDA_structure_elements.csv"
    # We mostly need field title and data_type per structure key
    # The structures file may define structures; elements file links structure->field
    field_defs: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    if not elems_path.exists():
        return field_defs
    with elems_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return field_defs
        cols = {str(n).strip().lstrip("\ufeff").lower(): n for n in reader.fieldnames}
        struct_col = cols.get("structure") or cols.get("structure_key") or cols.get("key")
        field_key_col = (
            cols.get("field") or cols.get("field_key") or cols.get("key_field") or cols.get("key")
        )
        title_col = cols.get("title") or cols.get("label") or cols.get("name")
        dtype_col = cols.get("data_type") or cols.get("type")
        enum_col = cols.get("enum_key") or cols.get("enum") or cols.get("enum_set")
        for row in reader:
            sname = (row.get(struct_col or "") or "").strip()
            fkey = (row.get(field_key_col or "") or "").strip()
            ftitle = (row.get(title_col or "") or fkey).strip()
            fdtype = (row.get(dtype_col or "") or "").strip()
            fenum = (row.get(enum_col or "") or "").strip()
            if not sname or not fkey:
                continue
            field_defs.setdefault(sname, {})[fkey] = (ftitle, fdtype, fenum)
    return field_defs


def _parse_type_spec(type_spec: str) -> Dict[str, str]:
    s = (type_spec or "").strip()
    if not s:
        return {"kind": "string"}
    low = s.lower().replace(" ", "")
    # list[...] pattern
    if low.startswith("list[") and low.endswith("]"):
        inner = low[5:-1]
        if inner.startswith("enum"):
            # list[enum:SET] or list[enum/SET]
            return {"kind": "list_enum"}
        if inner.startswith("object"):
            struct = (
                inner.split(":", 1)[1]
                if ":" in inner
                else inner.split("/", 1)[1]
                if "/" in inner
                else ""
            )
            return {"kind": "list_object", "struct": struct}
        return {"kind": "list_string"}
    # enum
    if low.startswith("enum"):
        return {"kind": "enum"}
    # object
    if low.startswith("object"):
        struct = low.split(":", 1)[1] if ":" in low else low.split("/", 1)[1] if "/" in low else ""
        return {"kind": "object", "struct": struct}
    return {"kind": "string"}


def _safe_json_load(val: Any) -> Any:
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            # Fallback: handle Python-literal-like strings with single quotes
            try:
                lit = ast.literal_eval(val)
                if isinstance(lit, (dict, list)):
                    return lit
            except Exception:
                pass
            return val
    return val


def _unwrap_leaf_value(val: Any) -> Any:
    """Recursively unwrap common wrapper objects to their 'value' payload.

    - Dicts with a 'value' key return that value (recursively unwrapped)
    - Lists are unwrapped element-wise
    - Other types returned as-is
    """
    if isinstance(val, dict):
        if "value" in val:
            return _unwrap_leaf_value(val.get("value"))
        # Not a typical wrapper; return as-is
        return val
    if isinstance(val, list):
        return [_unwrap_leaf_value(v) for v in val]
    return val


def _transform_enum_value(enum_sets: Dict[str, Dict[str, str]], enum_name: str, value: Any) -> Any:
    mapping = enum_sets.get(enum_name) or {}
    base = _unwrap_leaf_value(_safe_json_load(value))
    if isinstance(base, list):
        return [mapping.get(str(v), str(v)) for v in base]
    return mapping.get(str(base), str(base))


def _transform_object_value(
    struct_defs: Dict[str, Dict[str, Tuple[str, str, str]]],
    enum_sets: Dict[str, Dict[str, str]],
    struct_name: str,
    value: Any,
) -> Any:
    print("struct_name", struct_name)
    print("value", value)
    print("data", _safe_json_load(value))
    data = _safe_json_load(value)
    fields = struct_defs.get(struct_name) or {}
    print("type(data)", type(data))
    if isinstance(data, dict):
        print("------> is dict")
        out: Dict[str, Any] = {}
        for fkey, fval in data.items():
            ftitle, fdtype, fenum = fields.get(fkey, (fkey, "", ""))
            spec = _parse_type_spec(fdtype)
            print("ftitle", ftitle, "fdtype", fdtype, "fenum", fenum, "------> spec", spec)
            print("------> fval", fval)
            if spec["kind"] == "enum":
                print("------> is enum")
                out[ftitle] = _transform_enum_value(enum_sets, fenum, fval)
            elif spec["kind"] == "list_enum":
                out[ftitle] = _transform_enum_value(enum_sets, fenum, fval)
            elif spec["kind"] == "object":
                out[ftitle] = _transform_object_value(
                    struct_defs, enum_sets, spec.get("struct", ""), fval
                )
            elif spec["kind"] == "list_object":
                inner = _safe_json_load(fval)
                if isinstance(inner, list):
                    out[ftitle] = [
                        _transform_object_value(struct_defs, enum_sets, spec.get("struct", ""), it)
                        for it in inner
                    ]
                else:
                    out[ftitle] = inner
            else:
                out[ftitle] = _unwrap_leaf_value(fval)
        return out
    # If list of dicts
    if isinstance(data, list):
        return [
            _transform_object_value(struct_defs, enum_sets, struct_name, it)
            if isinstance(it, (dict, list))
            else it
            for it in data
        ]
    return data


def _transform_datapoints(
    dataset_root: Path,
    contract_type: Optional[str],
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    dp_defs = _load_datapoint_definitions(dataset_root, contract_type)
    enum_sets = _load_enums(dataset_root)
    struct_defs = _load_structure_definitions(dataset_root)
    out: Dict[str, Any] = {}
    for did, raw_val in raw.items():
        title, dtype, denum = dp_defs.get(did, (did, "", ""))
        spec = _parse_type_spec(dtype)
        print(title, "spec", spec)
        try:
            if spec["kind"] == "enum":
                out[title] = _transform_enum_value(enum_sets, denum, raw_val)
            elif spec["kind"] == "list_enum":
                out[title] = _transform_enum_value(enum_sets, denum, raw_val)
            elif spec["kind"] == "object":
                out[title] = _transform_object_value(
                    struct_defs, enum_sets, spec.get("struct", ""), raw_val
                )
                print(title, "->", out[title])
            elif spec["kind"] == "list_object":
                inner = _safe_json_load(raw_val)
                if isinstance(inner, list):
                    out[title] = [
                        _transform_object_value(struct_defs, enum_sets, spec.get("struct", ""), it)
                        for it in inner
                    ]
                else:
                    out[title] = inner
            else:
                out[title] = _unwrap_leaf_value(raw_val)
        except Exception:
            out[title] = _unwrap_leaf_value(raw_val)

    return out


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
                _read_clause_titles_with_map(paths["clauses_csv"], title_map)
                if paths["clauses_csv"].exists()
                else []
            )
            datapoints_raw = (
                _read_datapoints_csv(paths["datapoints_csv"])
                if paths["datapoints_csv"].exists()
                else {}
            )
            list_datapoints = _transform_datapoints(dataset_root, contract_type, datapoints_raw)

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
                "list_datapoints": list_datapoints,
                # guidelines left empty for now; no source provided in request
                "list_guidelines": {},
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
