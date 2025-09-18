from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_relaxed(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().casefold()
    # Replace common punctuation with space
    for ch in ["\u2019", "'", '"', ",", ";", ":"]:
        v = v.replace(ch, " ")
    # Collapse whitespace
    v = " ".join(v.split())
    return v


MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


def normalize_date(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    if v == "":
        return ""
    # Already ISO?
    import re as _re

    m = _re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", v)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}".casefold()

    vv = v.replace(",", " ").casefold()
    parts = [p for p in vv.split() if p]
    day = mon = year = None
    for p in parts:
        if p.isdigit() and len(p) == 4 and year is None:
            year = p
        elif p.isdigit() and len(p) <= 2 and day is None:
            day = p.zfill(2)
        else:
            m = MONTHS.get(p.strip(".,"))
            if m and mon is None:
                mon = m
    if year and mon and day:
        return f"{year}-{mon}-{day}"
    return normalize_relaxed(v)


def relaxed_equal(field: str, gold: str, pred: str) -> bool:
    f = field.strip().lower()
    if f in ("contract_date", "amendment_date"):
        return normalize_date(gold) == normalize_date(pred)
    # default relaxed compare
    return normalize_relaxed(gold) == normalize_relaxed(pred)


def load_json_map(dir_path: Path) -> dict[str, dict[str, str]]:
    """Load per-document JSON files into a flat map keyed by original filename.

    Returns a dict where values have simple string fields for comparison, and
    also include "<field>_explanation" entries if present in the JSON.
    """
    data: dict[str, dict[str, str]] = {}
    if not dir_path.exists():
        return data
    for p in dir_path.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as jf:
                obj = json.load(jf)
        except Exception:
            continue
        filename = str(obj.get("filename") or p.stem)
        rec: dict[str, str] = {"filename": filename}
        for field in (
            "contract_type",
            "contract_type_version",
            "contract_date",
            "amendment_date",
            "amendment_number",
            "version_type",
            "status",
            "party_name_1",
            "party_role_1",
            "party_name_2",
            "party_role_2",
            "document_quality",
        ):
            node = obj.get(field) or {}
            if isinstance(node, dict):
                val = node.get("value")
                expl = node.get("explanation")
            else:
                val = None
                expl = None
            rec[field] = "" if val is None else str(val)
            rec[f"{field}_explanation"] = "" if expl is None else str(expl)
        data[filename] = rec
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute organizer metadata metrics")
    parser.add_argument("--model", required=True, help="Model name (folder under output/organizer)")
    args = parser.parse_args()

    model_name = args.model
    repo_root = Path(__file__).resolve().parents[1]

    pred_dir = repo_root / "dataset" / "output" / "organizer" / model_name
    gold_dir = repo_root / "dataset" / "gold" / "organizer"
    out_dir = repo_root / "dataset" / "metrics" / "organizer" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.csv"

    if not pred_dir.exists():
        print(f"Predictions not found: {pred_dir}")
        return
    if not gold_dir.exists():
        print(f"Gold not found: {gold_dir}")
        return

    pred_map = load_json_map(pred_dir)
    gold_map = load_json_map(gold_dir)

    common = sorted(set(pred_map.keys()) & set(gold_map.keys()))
    if not common:
        print("No overlapping filenames between predictions and gold.")
        return

    fields = [
        "contract_type",
        "contract_type_version",
        "contract_date",
        "amendment_date",
        "amendment_number",
        "version_type",
        "status",
        "party_name_1",
        "party_role_1",
        "party_name_2",
        "party_role_2",
        "document_quality",
    ]

    # Aggregates per field
    per_field: dict[str, dict[str, int]] = {
        f: {"tp": 0, "fp": 0, "fn": 0, "present": 0, "correct": 0} for f in fields
    }
    mismatches: list[
        tuple[str, str, str, str, str]
    ] = []  # (filename, field, gold_value, pred_value, gold_explanation)

    for fn in common:
        gold_row = gold_map[fn]
        pred_row = pred_map[fn]
        for field in fields:
            gold_val = (gold_row.get(field) or "").strip()
            pred_val = (pred_row.get(field) or "").strip()

            if gold_val != "":
                per_field[field]["present"] += 1
                if relaxed_equal(field, gold_val, pred_val):
                    per_field[field]["correct"] += 1
                    if pred_val != "":
                        per_field[field]["tp"] += 1
                    else:
                        # pred empty but equal happens only if both empty; guarded by gold_val != ''
                        pass
                else:
                    if pred_val != "":
                        per_field[field]["fp"] += 1
                    per_field[field]["fn"] += 1
                    gold_expl = (gold_row.get(f"{field}_explanation") or "").strip()
                    mismatches.append((fn, field, gold_val, pred_val, gold_expl))
            else:
                # Gold empty: count FP if prediction non-empty
                if pred_val != "":
                    per_field[field]["fp"] += 1
                    gold_expl = (gold_row.get(f"{field}_explanation") or "").strip()
                    mismatches.append((fn, field, gold_val, pred_val, gold_expl))

    # Write metrics CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "accuracy", "support", "precision", "recall", "f1"])
        for field in fields:
            agg = per_field[field]
            present = agg["present"]
            correct = agg["correct"]
            tp = agg["tp"]
            fp = agg["fp"]
            fn = agg["fn"]
            accuracy = (correct / present) if present else 0.0
            precision = (tp / (tp + fp)) if (tp + fp) else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            def to_pct(val: float) -> str:
                try:
                    return f"{int(round(float(val) * 100))}%"
                except Exception:
                    return "0%"

            writer.writerow(
                [
                    field,
                    to_pct(accuracy),
                    present,
                    to_pct(precision),
                    to_pct(recall),
                    to_pct(f1),
                ]
            )
            print(
                f"  {field}: {to_pct(accuracy)} {present} {to_pct(precision)} {to_pct(recall)} {to_pct(f1)}"
            )

    print(f"Wrote metrics to {out_path}")

    # Write mismatches CSV
    mismatch_path = out_dir / "mismatch.csv"
    with mismatch_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "field", "gold_value", "pred_value", "gold_explanation"])
        for row in mismatches:
            writer.writerow(list(row))
    print(f"Wrote mismatches to {mismatch_path}")


if __name__ == "__main__":
    main()
