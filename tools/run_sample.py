from __future__ import annotations

"""Build sample text files per contract type/version from gold filter scopes.

For each gold file in dataset/gold/filter/<contract_type>/<filename>.json:
- Open dataset/gold/organizer/<filename>.json to read version_type and contract version
  (field name 'contract_type_version' preferred; fallback to 'contract_version' if present)
- If version_type is not empty and not AMENDMENT, append the scoped paragraphs from
  dataset/documents/organizer/files/<filename>.txt to:
    dataset/samples/<contract_type>/<contract_version>.txt

Each appended block is prefixed with a header line identifying the source file.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def text_to_paragraphs_safe(text: str) -> list[tuple[int, str]]:
    # Try to use core utility; fallback to simple blank-line split
    try:
        import sys as _sys

        src_dir = get_repo_root() / "src"
        if str(src_dir) not in _sys.path:
            _sys.path.insert(0, str(src_dir))
        from contract_ai_core.utilities import text_to_paragraphs as _ttp  # type: ignore

        items = _ttp(text)
        return [(p.index, p.text) for p in items]
    except Exception:
        blocks = [b.strip() for b in text.split("\n\n")]
        return [(i, b) for i, b in enumerate([b for b in blocks if b])]


def parse_value(node: Any, key: str) -> str | None:
    if not isinstance(node, dict):
        return None
    v = node.get(key)
    if isinstance(v, dict):
        val = v.get("value")
        return str(val) if val not in (None, "") else None
    return str(v) if v not in (None, "") else None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Create samples from gold filter scopes")
    parser.add_argument("--type", dest="contract_type", default=None, help="Contract type key")
    args = parser.parse_args()

    root = get_repo_root()
    gold_base = root / "dataset" / "gold" / "filter"
    org_base = root / "dataset" / "gold" / "organizer"
    txt_base = root / "dataset" / "documents" / "organizer" / "files"
    out_base = root / "dataset" / "samples"

    types = (
        [args.contract_type]
        if args.contract_type
        else ([p.name for p in gold_base.iterdir() if p.is_dir()] if gold_base.exists() else [])
    )
    if not types:
        logging.info("No contract types found under %s", gold_base)
        return

    for ct in sorted(types):
        ct_dir = gold_base / ct
        pred_files = sorted(ct_dir.glob("*.json"))
        if not pred_files:
            continue
        for pf in pred_files:
            try:
                pred = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                continue
            filename = str(pred.get("filename") or pf.stem)
            # Organizer gold JSON by same filename.stem
            org_path = org_base / f"{Path(filename).stem}.json"
            if not org_path.exists():
                logging.info("Missing organizer gold for %s", filename)
                continue
            try:
                org = json.loads(org_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            version_type = parse_value(org, "version_type") or ""
            # Prefer contract_type_version, fallback to contract_version if present
            contract_version = (
                parse_value(org, "contract_type_version")
                or parse_value(org, "contract_version")
                or ""
            )
            if not version_type or version_type.strip().upper() == "AMENDMENT":
                continue
            if not contract_version:
                continue

            src_txt = txt_base / filename
            if not src_txt.exists():
                logging.info("Source text not found: %s", src_txt)
                continue
            raw = read_text_best_effort(src_txt)
            paras = text_to_paragraphs_safe(raw)
            if not paras:
                continue

            # Collect unique selected paragraph indices from scopes
            selected: set[int] = set()
            for sc in pred.get("scopes") or []:
                try:
                    s = int((sc or {}).get("start_line", -1))
                    e = int((sc or {}).get("end_line", -1))
                except Exception:
                    s, e = -1, -1
                if s < 0 or e < s:
                    continue
                for i in range(s, e + 1):
                    selected.add(i)

            if not selected:
                continue

            out_dir = out_base / ct / contract_version
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(filename).stem}.txt"

            # Write selected paragraphs in order
            ordered_idx = sorted([i for i in selected if 0 <= i < len(paras)])
            if not ordered_idx:
                continue
            with out_path.open("w", encoding="utf-8") as f:
                for i in ordered_idx:
                    _, text = paras[i]
                    f.write(f"{text}\n")
            logging.info("created %s", out_path.relative_to(root))


if __name__ == "__main__":
    main()
