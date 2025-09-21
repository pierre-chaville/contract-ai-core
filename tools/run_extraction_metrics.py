from __future__ import annotations

"""CLI: Compute extraction metrics (relaxed accuracy, per-key accuracy/F1, completion).

Usage:
  python dataset/run_extraction_metrics.py --template ISDA --model gpt-4.1-mini
"""

import argparse
import ast as _ast
import csv
import json as _json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from utilities import load_template


def load_pred_datapoints(path: Path) -> dict[str, tuple[str, float, str]]:
    """Load predicted datapoints from CSV to a map: key -> (value, confidence_percent).

    Expects columns: key, title, confidence, value
    """
    data: dict[str, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            value = (row.get("value") or "").strip()
            conf_raw = (row.get("confidence") or "").strip()
            explanation = (row.get("explanation") or "").strip()
            try:
                conf = float(conf_raw)
            except Exception:
                conf = float("nan")
            data[key] = (value, conf, explanation)
    return data


def load_gold_datapoints(path: Path) -> dict[str, str]:
    """
    Load gold datapoints from CSV to a map: key -> value.
    Expected columns: key, value (other columns ignored)
    """
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("key") or "").strip()
            if not key:
                continue
            value = (row.get("value") or "").strip()
            data[key] = value
    return data


def normalize_relaxed(value: str) -> str:
    """
    Relaxed normalization for general strings: casefold, trim, collapse spaces,
    strip quotes/punct. Also removes commas in numbers.
    """
    if value is None:
        return ""
    v = value.strip().casefold()
    # Replace common punctuation with space
    for ch in ["\u2019", "'", '"', ",", ";", ":"]:
        v = v.replace(ch, " ")
    # Collapse whitespace
    v = " ".join(v.split())
    return v


def normalize_money(value: str) -> str:
    if value is None:
        return ""
    v = value.strip().casefold()
    # Remove spaces, commas, typical currency symbols; keep currency codes and digits
    for ch in [",", " ", "\u00a0", "$", "€", "£", "¥"]:
        v = v.replace(ch, "")
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
    v = value.strip().casefold()
    # Replace commas
    v = v.replace(",", " ")
    parts = [p for p in v.split() if p]
    # Try formats like 1 Jan 2023, Jan 1, 2023, 2023-01-01
    day = mon = year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = p
        elif p.isdigit() and len(p) <= 2 and day is None:
            day = p.zfill(2)
        else:
            m = MONTHS.get(p.strip(".,"), None)
            if m and mon is None:
                mon = m
    # Already in ISO?
    if re_match := __import__("re").match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", value.strip()):
        return f"{re_match.group(1)}-{re_match.group(2)}-{re_match.group(3)}"
    if year and mon and day:
        return f"{year}-{mon}-{day}"
    # print("value", value, "normalized", normalize_relaxed(value))
    return normalize_relaxed(value)


def normalize_bool(value: str) -> str:
    if value is None:
        return "false"
    v = value.strip().casefold()
    if v in ("true", "yes", "1"):
        return "true"
    if v in ("false", "no", "0"):
        return "false"
    return "false"


def normalize_float(value: str) -> str:
    if value is None:
        return ""
    v = value.strip().casefold()
    # Remove thousands separators and percent sign
    v = v.replace(",", "").replace("%", "")
    return v


def normalize_int(value: str) -> str:
    if value is None:
        return ""
    v = value.strip().casefold()
    v = v.replace(",", "").replace("%", "")
    return v


def normalize_enum(value: str) -> str:
    if value is None:
        return ""
    v = value.strip()
    # Keep only alphanumerics, compare case-insensitively
    v = "".join(ch for ch in v if ch.isalnum())
    return v.casefold()


def relaxed_equal(gold: str, pred: str, dtype: str) -> bool:
    if gold is None:
        gold = ""
    if pred is None:
        pred = ""

    dt = (dtype or "").strip().lower()
    # Floats (allow tolerance); support synonyms
    if dt in ("float", "number", "double", "percent"):
        gs = normalize_float(gold)
        ps = normalize_float(pred)
        try:
            gv = float(gs) if gs != "" else float("nan")
            pv = float(ps) if ps != "" else float("nan")
            if np.isnan(gv) or np.isnan(pv):
                return gs == ps
            tol = max(1e-6, 1e-3 * max(abs(gv), abs(pv)))
            return abs(gv - pv) <= tol
        except Exception:
            return gs == ps

    # Integers
    if dt in ("int", "integer"):
        gs = normalize_int(gold)
        ps = normalize_int(pred)
        try:
            gv = int(float(gs)) if gs not in ("", "nan") else -1
            pv = int(float(ps)) if ps not in ("", "nan") else -1
            return gv == pv
        except Exception:
            return gs == ps

    # Booleans
    if dt in ("bool", "boolean"):
        return normalize_bool(gold) == normalize_bool(pred)

    # Dates
    if dt in ("date", "datetime"):
        return normalize_date(gold) == normalize_date(pred)

    # Enums
    if dt in ("enum",):
        return normalize_enum(gold) == normalize_enum(pred)

    # Money/amount (if provided)
    if dt in ("money", "amount"):
        return normalize_money(gold) == normalize_money(pred)

    # Default string compare
    return normalize_relaxed(gold) == normalize_relaxed(pred)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Compute datapoint extraction metrics")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument(
        "--model", required=True, help="Model name used for predictions (folder name)"
    )
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]

    gold_dir = repo_root / "dataset" / "gold" / "datapoints" / template_key
    pred_dir = repo_root / "dataset" / "output" / "datapoints" / template_key / model_name
    out_dir = repo_root / "dataset" / "metrics" / "extraction" / template_key / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load template for types and required flags
    template = load_template(template_key)
    key_to_type = {
        dp["key"]: dp.get("data_type", "string") for dp in template.get("datapoints", [])
    }

    # Build structure element types for flattening object/list[object] datapoints
    def _parse_structure_type(dt: str) -> tuple[str, str | None]:
        s = str(dt or "").strip().lower()
        if s.startswith("list[") and "object:" in s:
            inside = s[s.find("[") + 1 : s.rfind("]")]
            if inside.startswith("object:"):
                key = inside.split(":", 1)[1].strip().strip("[]")
                return ("list_object", key)
        if s.startswith("object:"):
            return ("object", s.split(":", 1)[1].strip().strip("[]"))
        return ("simple", None)

    struct_el_types: dict[str, dict[str, str]] = {}
    for sdef in template.get("structures", []) or []:
        skey = str(sdef.get("structure_key", "")).strip()
        if not skey:
            continue
        elmap: dict[str, str] = {}
        for el in sdef.get("elements", []) or []:
            ekey = str(el.get("key", "")).strip()
            etype = str(el.get("data_type", "string")).strip() or "string"
            if ekey:
                elmap[ekey] = etype
        struct_el_types[skey] = elmap

    dp_struct_kind: dict[str, tuple[str, str]] = {}
    for dp in template.get("datapoints", []) or []:
        key = str(dp.get("key", "")).strip()
        dt = str(dp.get("data_type", "")).strip()
        kind, skey = _parse_structure_type(dt)
        if key and kind in ("object", "list_object") and skey:
            dp_struct_kind[key] = (kind, skey)

    def _try_parse_json(text: str) -> Any:
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None
        try:
            return _json.loads(s)
        except Exception:
            try:
                return _ast.literal_eval(s)
            except Exception:
                return None

    def _extract_leaf_val(v: Any) -> str:
        # If v is a dict with a 'value' field, take it; else str()
        try:
            if isinstance(v, dict) and "value" in v:
                val = v.get("value")
                return "" if val is None else str(val)
            return "" if v is None else str(v)
        except Exception:
            return ""

    def flatten_pred_map(
        pred_map: dict[str, tuple[str, float, str]],
    ) -> tuple[dict[str, tuple[str, float, str]], dict[str, str]]:
        out: dict[str, tuple[str, float, str]] = {}
        flat_types: dict[str, str] = {}
        for key, (val, conf, expl) in pred_map.items():
            if key in dp_struct_kind:
                kind, skey = dp_struct_kind[key]
                parsed = _try_parse_json(val)
                elem_types = struct_el_types.get(skey, {})
                if kind == "object" and isinstance(parsed, dict):
                    for el_key, el_val in parsed.items():
                        flat_key = f"{key}.{el_key}"
                        out[flat_key] = (_extract_leaf_val(el_val), conf, expl)
                        flat_types[flat_key] = elem_types.get(el_key, "string")
                elif kind == "list_object" and isinstance(parsed, list):
                    for i, obj in enumerate(parsed):
                        if not isinstance(obj, dict):
                            continue
                        for el_key, el_val in obj.items():
                            flat_key = f"{key}[{i}].{el_key}"
                            out[flat_key] = (_extract_leaf_val(el_val), conf, expl)
                            flat_types[flat_key] = elem_types.get(el_key, "string")
                else:
                    # Fallback: keep original
                    out[key] = (val, conf, expl)
            else:
                out[key] = (val, conf, expl)
        return out, flat_types

    def flatten_gold_map(gold_map: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        out: dict[str, str] = {}
        flat_types: dict[str, str] = {}
        for key, val in gold_map.items():
            if key in dp_struct_kind:
                kind, skey = dp_struct_kind[key]
                parsed = _try_parse_json(val)
                elem_types = struct_el_types.get(skey, {})
                if kind == "object" and isinstance(parsed, dict):
                    for el_key, el_val in parsed.items():
                        flat_key = f"{key}.{el_key}"
                        out[flat_key] = _extract_leaf_val(el_val)
                        flat_types[flat_key] = elem_types.get(el_key, "string")
                elif kind == "list_object" and isinstance(parsed, list):
                    for i, obj in enumerate(parsed):
                        if not isinstance(obj, dict):
                            continue
                        for el_key, el_val in obj.items():
                            flat_key = f"{key}[{i}].{el_key}"
                            out[flat_key] = _extract_leaf_val(el_val)
                            flat_types[flat_key] = elem_types.get(el_key, "string")
                else:
                    out[key] = val
            else:
                out[key] = val
        return out, flat_types

    required_keys = {
        dp["key"] for dp in template.get("datapoints", []) if dp.get("required", False)
    }

    gold_files = {p.stem: p for p in gold_dir.glob("*.csv")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.csv")}
    common_docs = sorted(set(gold_files.keys()) & set(pred_files.keys()))
    if not common_docs:
        logging.warning("No overlapping files between gold=%s and preds=%s", gold_dir, pred_dir)
        return

    # Aggregates
    total_gold_present = 0
    total_relaxed_correct = 0

    # Per-key counts for F1
    per_key_tp: dict[str, int] = {}
    per_key_fp: dict[str, int] = {}
    per_key_fn: dict[str, int] = {}
    # Per-key accuracy components (considering only docs where gold has a value)
    per_key_present: dict[str, int] = {}
    per_key_correct: dict[str, int] = {}

    # Type-specific
    type_counts: dict[str, tuple[int, int]] = {
        "date": (0, 0),
        "money": (0, 0),
    }  # (gold_present, correct)

    # Completion
    completion_fractions: list[float] = []

    # Collect mismatches for a consolidated CSV: file, key, gold_value, pred_value
    mismatches: list[tuple[str, str, str, str, str]] = []

    # Selective extraction at 90% confidence
    hc_threshold = 90.0
    hc_selected = 0
    hc_correct = 0
    hc_total_gold_present = 0

    for stem in common_docs:
        gold_map_raw = load_gold_datapoints(gold_files[stem])
        pred_map_raw = load_pred_datapoints(pred_files[stem])

        # Flatten object and list[object] datapoints to element-level keys
        gold_map, gold_flat_types = flatten_gold_map(gold_map_raw)
        pred_map, pred_flat_types = flatten_pred_map(pred_map_raw)

        # Relaxed accuracy and type-specific
        for key, gold_val in gold_map.items():
            # Prefer flattened element type if available; fallback to dp-level type
            dtype = (
                pred_flat_types.get(key)
                or gold_flat_types.get(key)
                or key_to_type.get(key, "string")
            )
            pred_val, pred_conf, explanation = pred_map.get(key, ("", float("nan"), ""))
            if gold_val.strip() != "":
                total_gold_present += 1
                # High-confidence tracking denominator
                hc_total_gold_present += 1
                # Per-key accuracy tracking
                per_key_present[key] = per_key_present.get(key, 0) + 1
                if relaxed_equal(gold_val, pred_val, dtype):
                    total_relaxed_correct += 1
                    per_key_correct[key] = per_key_correct.get(key, 0) + 1
                    # High-confidence accuracy counting
                    if not np.isnan(pred_conf) and pred_conf >= hc_threshold:
                        hc_selected += 1
                        hc_correct += 1
                else:
                    mismatches.append((stem, key, gold_val, pred_val, explanation))
                    if not np.isnan(pred_conf) and pred_conf >= hc_threshold:
                        hc_selected += 1

                # Type-specific for certain dtypes
                if dtype.lower() == "date":
                    gp, cc = type_counts["date"]
                    type_counts["date"] = (
                        gp + 1,
                        cc + (1 if relaxed_equal(gold_val, pred_val, dtype) else 0),
                    )
                if dtype.lower() in ("int", "float"):
                    gp, cc = type_counts["money"]
                    type_counts["money"] = (
                        gp + 1,
                        cc + (1 if relaxed_equal(gold_val, pred_val, dtype) else 0),
                    )

            # Field-level F1 accounting (single value per field per doc)
            if gold_val.strip() == "":
                # Gold negative
                if pred_val.strip() != "":
                    per_key_fp[key] = per_key_fp.get(key, 0) + 1
            else:
                if pred_val.strip() == "":
                    per_key_fn[key] = per_key_fn.get(key, 0) + 1
                else:
                    if relaxed_equal(gold_val, pred_val, dtype):
                        per_key_tp[key] = per_key_tp.get(key, 0) + 1
                    else:
                        per_key_fp[key] = per_key_fp.get(key, 0) + 1
                        per_key_fn[key] = per_key_fn.get(key, 0) + 1

        # Document completion score
        # Primary: fraction of required keys with any predicted value
        # Fallback: if no required keys are defined in the template,
        #           use the set of keys that have a gold value in this document.
        if required_keys:
            denom_keys = required_keys
        else:
            denom_keys = {k for k, v in gold_map.items() if (v or "").strip() != ""}
        if denom_keys:
            num_present = sum(
                1
                for k in denom_keys
                if (pred_map.get(k, ("", float("nan"), ""))[0] or "").strip() != ""
            )
            completion_fractions.append(num_present / len(denom_keys))

    # Compute overall metrics
    relaxed_accuracy = (total_relaxed_correct / total_gold_present) if total_gold_present else 0.0
    completion_score = (
        (sum(completion_fractions) / len(completion_fractions)) if completion_fractions else 0.0
    )
    hc_accuracy = (hc_correct / hc_selected) if hc_selected else float("nan")
    hc_coverage = (hc_selected / hc_total_gold_present) if hc_total_gold_present else 0.0

    # Per-key metrics aggregated across documents
    per_key_list: list[dict[str, float | int | str]] = []
    all_keys = set(
        list(per_key_tp.keys())
        + list(per_key_fp.keys())
        + list(per_key_fn.keys())
        + list(per_key_present.keys())
    )
    for key in sorted(all_keys):
        tp = per_key_tp.get(key, 0)
        fp = per_key_fp.get(key, 0)
        fn = per_key_fn.get(key, 0)
        present = per_key_present.get(key, 0)
        correct = per_key_correct.get(key, 0)
        precision = (float(tp) / float(tp + fp)) if (tp + fp) else 0.0
        recall = (float(tp) / float(tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy_k = (float(correct) / float(present)) if present else float("nan")
        per_key_list.append(
            {
                "key": key,
                "accuracy": accuracy_k,
                "count": int(present),
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    summary = {
        "Relaxed match accuracy": relaxed_accuracy,
        "Document completion score": completion_score,
        "Selective 90% accuracy": hc_accuracy,
        "Selective 90% coverage": hc_coverage,
        "Field level F1 macro": float(np.mean([it["f1"] for it in per_key_list]))
        if per_key_list
        else 0.0,
        "per_key": per_key_list,
    }

    def _to_pct(val: float) -> str:
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "nan"
            return f"{int(round(float(val) * 100))}%"
        except Exception:
            return str(val)

    def _map_to_pct(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _map_to_pct(v) for k, v in obj.items()}
        if isinstance(obj, int | float):
            return _to_pct(float(obj))
        return obj

    # Map summary to percentages where applicable, keeping counts
    def _map_list_pct(lst: list[dict[str, object]]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for it in lst:
            out.append(
                {
                    "key": it.get("key", ""),
                    "accuracy": _to_pct(float(it.get("accuracy", 0.0) or 0.0)),
                    "count": it.get("count", 0),
                    "f1": _to_pct(float(it.get("f1", 0.0) or 0.0)),
                    "precision": _to_pct(float(it.get("precision", 0.0) or 0.0)),
                    "recall": _to_pct(float(it.get("recall", 0.0) or 0.0)),
                }
            )
        return out

    summary_pct = {k: v for k, v in summary.items() if k != "per_key"}
    summary_pct = _map_to_pct(summary_pct)
    summary_pct["per_key"] = _map_list_pct(per_key_list)

    out_dir = repo_root / "dataset" / "metrics" / "extraction" / template_key / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    # No separate per-key CSVs; per-key metrics are included in the summary

    logging.info("\n%s", yaml.safe_dump(summary_pct, sort_keys=False, allow_unicode=True))
    if mismatches:
        with (out_dir / "mismatches.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "key", "gold_value", "pred_value", "explanation"])
            for row in sorted(mismatches):
                writer.writerow(list(row))


if __name__ == "__main__":
    main()
