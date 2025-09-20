from __future__ import annotations

"""Compute filtering scope metrics vs gold.

For each contract type and model, compare predicted scope spans against gold:
- Accuracy per scope instance (1.0 if start_line and end_line both match exactly, else 0.0)
- IoU per scope instance, treating spans as inclusive integer ranges

Aggregates:
- Overall average accuracy and IoU across all scope instances
- IoU histogram in ranges: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, and exact 1.0
- Per-scope averages (avg accuracy, avg IoU, count)

Outputs: dataset/metrics/<contract_type>/<model>/result.json

Usage:
  python tools/run_filter_metrics.py            # compute metrics for all types/models found
  python tools/run_filter_metrics.py --type ISDA --model gpt-4.1-mini
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def list_contract_types() -> list[str]:
    root = get_repo_root()
    base = root / "dataset" / "output" / "filter"
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    except Exception:
        return []


def list_models_for_type(contract_type: str) -> list[str]:
    root = get_repo_root()
    base = root / "dataset" / "output" / "filter" / contract_type
    try:
        return sorted([p.name for p in base.iterdir() if p.is_dir()])
    except Exception:
        return []


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def span_iou(pred: Tuple[int, int], gold: Tuple[int, int]) -> float:
    """Compute IoU for inclusive integer spans (start, end). Returns 0 for invalid spans."""
    ps, pe = int(pred[0]), int(pred[1])
    gs, ge = int(gold[0]), int(gold[1])
    if ps < 0 or pe < ps or gs < 0 or ge < gs:
        if ps == gs and pe == ge:
            return 1.0
        else:
            return 0.0
    inter_start = max(ps, gs)
    inter_end = min(pe, ge)
    inter = max(0, inter_end - inter_start + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def bucket_iou(iou: float) -> str:
    if iou == 1.0:
        return "1.0"
    if iou < 0.2:
        return "0-0.2"
    if iou < 0.4:
        return "0.2-0.4"
    if iou < 0.6:
        return "0.4-0.6"
    if iou < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def compute_metrics_for(contract_type: str, model: str) -> dict[str, Any] | None:
    root = get_repo_root()
    pred_dir = root / "dataset" / "output" / "filter" / contract_type / model
    gold_dir = root / "dataset" / "gold" / "filter" / contract_type
    if not pred_dir.exists() or not gold_dir.exists():
        logging.info("Skipping %s/%s: missing directories", contract_type, model)
        return None

    # Iterate predicted files; match to gold by stem
    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        logging.info("No predicted files for %s/%s", contract_type, model)
        return None

    total_pairs = 0
    sum_acc = 0.0
    sum_iou = 0.0
    hist_counts: Dict[str, int] = defaultdict(int)
    per_scope_sum_acc: Dict[str, float] = defaultdict(float)
    per_scope_sum_iou: Dict[str, float] = defaultdict(float)
    per_scope_count: Dict[str, int] = defaultdict(int)
    files_compared = 0

    for pf in pred_files:
        pred = read_json(pf) or {}
        stem = pf.stem
        gold_path = gold_dir / f"{stem}.json"
        gold = read_json(gold_path) or {}
        if not gold:
            continue
        files_compared += 1

        # Build scope maps by name
        pred_scopes = {
            str((s or {}).get("name", "")).strip(): s for s in (pred.get("scopes") or [])
        }
        gold_scopes = {
            str((s or {}).get("name", "")).strip(): s for s in (gold.get("scopes") or [])
        }

        # Compare only scopes present in gold
        for name, gs in gold_scopes.items():
            if not name:
                continue
            ps = pred_scopes.get(name)
            g_start = int(gs.get("start_line", -1))
            g_end = int(gs.get("end_line", -1))
            if ps is None:
                acc = 0.0
                iou = 0.0
            else:
                p_start = int(ps.get("start_line", -1))
                p_end = int(ps.get("end_line", -1))
                acc = 1.0 if (p_start == g_start and p_end == g_end) else 0.0
                iou = span_iou((p_start, p_end), (g_start, g_end))

            if iou != 1.0:
                print(f"name: {name}, acc: {acc}, iou: {iou}")
            total_pairs += 1
            sum_acc += acc
            sum_iou += iou
            hist_counts[bucket_iou(iou)] += 1
            per_scope_sum_acc[name] += acc
            per_scope_sum_iou[name] += iou
            per_scope_count[name] += 1

    if total_pairs == 0:
        logging.info("No scope pairs to compare for %s/%s", contract_type, model)
        return None

    overall = {
        "avg_accuracy": sum_acc / total_pairs,
        "avg_iou": sum_iou / total_pairs,
        "histogram_iou": {
            # Ensure all buckets present
            k: int(hist_counts.get(k, 0))
            for k in ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0"]
        },
    }

    per_scope = {
        name: {
            "count": per_scope_count[name],
            "avg_accuracy": per_scope_sum_acc[name] / per_scope_count[name]
            if per_scope_count[name]
            else 0.0,
            "avg_iou": per_scope_sum_iou[name] / per_scope_count[name]
            if per_scope_count[name]
            else 0.0,
        }
        for name in sorted(per_scope_count.keys())
    }

    return {
        "contract_type": contract_type,
        "model": model,
        "counts": {"files": files_compared, "pairs": total_pairs},
        "overall": overall,
        "per_scope": per_scope,
    }


def write_result(contract_type: str, model: str, result: dict[str, Any]) -> None:
    root = get_repo_root()
    out_dir = root / "dataset" / "metrics" / "filter" / contract_type / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info("wrote %s", out_path.relative_to(root))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Compute metrics for filter scopes vs gold")
    parser.add_argument("--type", dest="contract_type", default=None, help="Contract type key")
    parser.add_argument("--model", dest="model", default=None, help="Model name")
    args = parser.parse_args()

    if args.contract_type and args.model:
        res = compute_metrics_for(args.contract_type, args.model)
        if res:
            write_result(args.contract_type, args.model, res)
        return

    # Otherwise, enumerate all types/models under output/filter and compute
    types = [args.contract_type] if args.contract_type else list_contract_types()
    for ct in types:
        models = [args.model] if args.model else list_models_for_type(ct)
        for m in models:
            res = compute_metrics_for(ct, m)
            if res:
                write_result(ct, m, res)


if __name__ == "__main__":
    main()
