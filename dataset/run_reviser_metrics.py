import math
import pprint
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Span = Tuple[
    int, int
]  # inclusive [start, end], 1-indexed or 0-indexed (consistent across pred/gold)


# ---------- utilities ----------
def normalize(
    spans: List[Span], clamp_to: Optional[Tuple[int, int]] = None, merge_adjacent: bool = True
) -> List[Span]:
    """Sort and (optionally) merge overlapping/adjacent spans; optionally clamp to [lo, hi]."""
    s = []
    if clamp_to:
        lo, hi = clamp_to
        for a, b in spans:
            a = max(lo, a)
            b = min(hi, b)
            if a <= b:
                s.append((a, b))
    else:
        s = spans[:]
    s.sort()
    if not s:
        return s
    out = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb + (1 if merge_adjacent else 0):  # overlap or touch
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def length(span: Span) -> int:
    a, b = span
    return max(0, b - a + 1)


def inter_len(p: Span, g: Span) -> int:
    return max(0, min(p[1], g[1]) - max(p[0], g[0]) + 1)


def iou(p: Span, g: Span) -> float:
    inter = inter_len(p, g)
    if inter == 0:
        return 0.0
    union = length(p) + length(g) - inter
    return inter / union


def to_lines(spans: List[Span]) -> set:
    s = set()
    for a, b in spans:
        s.update(range(a, b + 1))
    return s


# ---------- matching (Hungarian if available, else greedy) ----------
def match_spans(pred: List[Span], gold: List[Span]) -> List[Tuple[int, int, float]]:
    """
    Returns list of (pi, gi, iou) one-to-one matches maximizing total IoU.
    """
    if not pred or not gold:
        return []
    M = np.zeros((len(pred), len(gold)), dtype=float)
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            M[i, j] = iou(p, g)
    try:
        # maximize IoU -> minimize cost = 1 - IoU
        from scipy.optimize import linear_sum_assignment

        # pad to square (Hungarian expects rectangular, but we want one-to-one up to min(n,m))
        cost = 1.0 - M
        ri, cj = linear_sum_assignment(cost)
        pairs = [
            (int(i), int(j), float(M[i, j])) for i, j in zip(ri, cj, strict=False) if M[i, j] > 0
        ]
        # Keep only up to min(len(pred), len(gold)) naturally
        return pairs
    except Exception:
        # Greedy fallback
        used_p, used_g, pairs = set(), set(), []
        flat = sorted(
            ((float(M[i, j]), i, j) for i in range(len(pred)) for j in range(len(gold))),
            reverse=True,
        )
        for val, i, j in flat:
            if val <= 0:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            pairs.append((i, j, val))
        return pairs


# ---------- metrics ----------
@dataclass
class SpanPRF:
    precision: float
    recall: float
    f1: float


def prf(tp: int, fp: int, fn: int) -> SpanPRF:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return SpanPRF(P, R, F1)


def span_detection_scores(
    pred: List[Span], gold: List[Span], iou_thresholds=(0.3, 0.5, 0.7)
) -> Dict[str, SpanPRF]:
    matches = match_spans(pred, gold)
    # Map best IoU by pred and by gold
    iou_by_pred = {pi: 0.0 for pi in range(len(pred))}
    iou_by_gold = {gi: 0.0 for gi in range(len(gold))}
    for pi, gi, ov in matches:
        iou_by_pred[pi] = ov
        iou_by_gold[gi] = ov
    scores = {}
    for t in iou_thresholds:
        tp = sum(1 for v in iou_by_pred.values() if v >= t)
        fp = len(pred) - tp
        fn = sum(1 for v in iou_by_gold.values() if v < t)
        scores[f"F1@{t:.1f}"] = prf(tp, fp, fn)
    return scores


def soft_prf(pred: List[Span], gold: List[Span]) -> SpanPRF:
    matches = match_spans(pred, gold)
    tp_mass = sum(inter_len(pred[pi], gold[gi]) for pi, gi, _ in matches)
    pred_mass = sum(length(p) for p in pred)
    gold_mass = sum(length(g) for g in gold)
    P = tp_mass / pred_mass if pred_mass else 0.0
    R = tp_mass / gold_mass if gold_mass else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return SpanPRF(P, R, F1)


def boundary_errors(pred: List[Span], gold: List[Span]) -> Dict[str, Dict[str, float]]:
    matches = match_spans(pred, gold)
    start_errs = [abs(pred[pi][0] - gold[gi][0]) for pi, gi, _ in matches]
    end_errs = [abs(pred[pi][1] - gold[gi][1]) for pi, gi, _ in matches]

    def agg(xs):
        if not xs:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {
            "mean": float(sum(xs) / len(xs)),
            "median": float(statistics.median(xs)),
            "p90": float(sorted(xs)[int(math.ceil(0.9 * len(xs)) - 1)]),
        }

    return {"start": agg(start_errs), "end": agg(end_errs)}


def split_merge_rates(pred: List[Span], gold: List[Span]) -> Dict[str, float]:
    matches = match_spans(pred, gold)
    g_to_p = defaultdict(set)
    p_to_g = defaultdict(set)
    for pi, gi, _ in matches:
        g_to_p[gi].add(pi)
        p_to_g[pi].add(gi)
    split_rate = sum(1 for gi in range(len(gold)) if len(g_to_p.get(gi, set())) >= 2) / (
        len(gold) or 1
    )
    merge_rate = sum(1 for pi in range(len(pred)) if len(p_to_g.get(pi, set())) >= 2) / (
        len(pred) or 1
    )
    return {"split_rate": split_rate, "merge_rate": merge_rate}


def line_level(pred: List[Span], gold: List[Span]) -> Dict[str, float]:
    P = to_lines(pred)
    G = to_lines(gold)
    inter = len(P & G)
    union = len(P | G) or 1
    jacc = inter / union
    coverage = inter / (len(G) or 1)  # recall at line level
    leakage = (len(P - G) / (len(P) or 1)) if P else 0.0  # 1 - precision at line level
    return {"jaccard": jacc, "coverage": coverage, "leakage": leakage}


# ---------- master entry ----------
def evaluate_spans(
    pred: List[Span],
    gold: List[Span],
    clamp_to: Optional[Tuple[int, int]] = None,
    merge_adjacent: bool = True,
    iou_thresholds=(0.3, 0.5, 0.7),
) -> Dict:
    pred_n = normalize(pred, clamp_to=clamp_to, merge_adjacent=merge_adjacent)
    gold_n = normalize(gold, clamp_to=clamp_to, merge_adjacent=merge_adjacent)
    results = {
        "counts": {"pred": len(pred_n), "gold": len(gold_n)},
        "span_prf": {
            k: vars(v) for k, v in span_detection_scores(pred_n, gold_n, iou_thresholds).items()
        },
        "soft_prf": vars(soft_prf(pred_n, gold_n)),
        "boundary_errors": boundary_errors(pred_n, gold_n),
        "split_merge": split_merge_rates(pred_n, gold_n),
        "line_level": line_level(pred_n, gold_n),
        "normalized": {"pred": pred_n, "gold": gold_n},
    }
    return results


# ---------- Corpus aggregation ----------
def _span_detection_counts(
    pred: List[Span], gold: List[Span], thresholds
) -> Dict[float, Tuple[int, int, int]]:
    """Return {tau: (tp, fp, fn)} for one doc."""
    matches = match_spans(pred, gold)
    best_pred = {pi: 0.0 for pi in range(len(pred))}
    best_gold = {gi: 0.0 for gi in range(len(gold))}
    for pi, gi, ov in matches:
        best_pred[pi] = ov
        best_gold[gi] = ov
    out = {}
    for t in thresholds:
        tp = sum(1 for v in best_pred.values() if v >= t)
        fp = len(pred) - tp
        fn = sum(1 for v in best_gold.values() if v < t)
        out[t] = (tp, fp, fn)
    return out


def _soft_masses(pred: List[Span], gold: List[Span]) -> Tuple[int, int, int]:
    matches = match_spans(pred, gold)
    tp_mass = sum(inter_len(pred[pi], gold[gi]) for pi, gi, _ in matches)
    pred_mass = sum(length(p) for p in pred)
    gold_mass = sum(length(g) for g in gold)
    return tp_mass, pred_mass, gold_mass


def _boundary_err_arrays(pred: List[Span], gold: List[Span]):
    matches = match_spans(pred, gold)
    start_errs = [abs(pred[pi][0] - gold[gi][0]) for pi, gi, _ in matches]
    end_errs = [abs(pred[pi][1] - gold[gi][1]) for pi, gi, _ in matches]
    return start_errs, end_errs


def _line_level_counts(pred: List[Span], gold: List[Span]):
    P, G = to_lines(pred), to_lines(gold)
    inter = len(P & G)
    union = len(P | G)
    return inter, union, len(G), len(P - G), len(P)


def aggregate_corpus(
    docs: Iterable[Tuple[List[Span], List[Span]]],
    clamp_to=None,
    merge_adjacent=True,
    iou_thresholds=(0.3, 0.5, 0.7),
) -> Dict:
    """
    docs: iterable of (pred_spans, gold_spans) for each document.
    Returns dict with 'micro' and 'macro' views + per-doc metrics if you want them.
    """
    per_doc = []
    # accumulators for micro
    micro_tp = defaultdict(int)
    micro_fp = defaultdict(int)
    micro_fn = defaultdict(int)
    tp_mass_sum = pred_mass_sum = gold_mass_sum = 0
    all_start_errs, all_end_errs = [], []
    inter_sum = union_sum = gold_size_sum = leakage_num_sum = pred_size_sum = 0
    pred_count_sum = gold_count_sum = 0

    for pred, gold in docs:
        pred_n = normalize(pred, clamp_to=clamp_to, merge_adjacent=merge_adjacent)
        gold_n = normalize(gold, clamp_to=clamp_to, merge_adjacent=merge_adjacent)

        # per-doc full report (same as before)
        doc_metrics = evaluate_spans(
            pred_n, gold_n, clamp_to=None, merge_adjacent=True, iou_thresholds=iou_thresholds
        )
        per_doc.append(doc_metrics)

        # micro: span detection
        counts = _span_detection_counts(pred_n, gold_n, iou_thresholds)
        for t, (tp, fp, fn) in counts.items():
            micro_tp[t] += tp
            micro_fp[t] += fp
            micro_fn[t] += fn

        # micro: soft masses
        tp_m, pm, gm = _soft_masses(pred_n, gold_n)
        tp_mass_sum += tp_m
        pred_mass_sum += pm
        gold_mass_sum += gm

        # micro: boundary pooling
        s_errs, e_errs = _boundary_err_arrays(pred_n, gold_n)
        all_start_errs.extend(s_errs)
        all_end_errs.extend(e_errs)

        # micro: line-level
        inter, union, gold_sz, leak_num, pred_sz = _line_level_counts(pred_n, gold_n)
        inter_sum += inter
        union_sum += union or 0
        gold_size_sum += gold_sz
        leakage_num_sum += leak_num
        pred_size_sum += pred_sz

        # counts
        pred_count_sum += len(pred_n)
        gold_count_sum += len(gold_n)

    # build micro view
    micro_span_prf = {}
    for t in iou_thresholds:
        tp, fp, fn = micro_tp[t], micro_fp[t], micro_fn[t]
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        micro_span_prf[f"F1@{t:.1f}"] = {"precision": P, "recall": R, "f1": F}

    micro_soft = {
        "precision": (tp_mass_sum / pred_mass_sum) if pred_mass_sum else 0.0,
        "recall": (tp_mass_sum / gold_mass_sum) if gold_mass_sum else 0.0,
    }
    micro_soft["f1"] = (
        (
            2
            * micro_soft["precision"]
            * micro_soft["recall"]
            / (micro_soft["precision"] + micro_soft["recall"])
        )
        if (micro_soft["precision"] + micro_soft["recall"])
        else 0.0
    )

    def _agg_err(xs):
        if not xs:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {
            "mean": float(sum(xs) / len(xs)),
            "median": float(statistics.median(xs)),
            "p90": float(sorted(xs)[int(math.ceil(0.9 * len(xs)) - 1)]),
        }

    micro_boundary = {"start": _agg_err(all_start_errs), "end": _agg_err(all_end_errs)}

    micro_line = {
        "jaccard": (inter_sum / union_sum) if union_sum else 0.0,
        "coverage": (inter_sum / (gold_size_sum or 1)),
        "leakage": (leakage_num_sum / (pred_size_sum or 1)) if pred_size_sum else 0.0,
    }

    # macro = unweighted average of per-doc metrics
    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    macro = {
        "span_prf": (
            {
                k: {
                    "precision": _mean([d["span_prf"][k]["precision"] for d in per_doc]),
                    "recall": _mean([d["span_prf"][k]["recall"] for d in per_doc]),
                    "f1": _mean([d["span_prf"][k]["f1"] for d in per_doc]),
                }
                for k in per_doc[0]["span_prf"].keys()
            }
            if per_doc
            else {}
        ),
        "soft_prf": (
            {
                "precision": _mean([d["soft_prf"]["precision"] for d in per_doc]),
                "recall": _mean([d["soft_prf"]["recall"] for d in per_doc]),
                "f1": _mean([d["soft_prf"]["f1"] for d in per_doc]),
            }
            if per_doc
            else {}
        ),
        "boundary_errors": (
            {
                "start": {
                    "mean": _mean([d["boundary_errors"]["start"]["mean"] for d in per_doc]),
                    "median": _mean([d["boundary_errors"]["start"]["median"] for d in per_doc]),
                    "p90": _mean([d["boundary_errors"]["start"]["p90"] for d in per_doc]),
                },
                "end": {
                    "mean": _mean([d["boundary_errors"]["end"]["mean"] for d in per_doc]),
                    "median": _mean([d["boundary_errors"]["end"]["median"] for d in per_doc]),
                    "p90": _mean([d["boundary_errors"]["end"]["p90"] for d in per_doc]),
                },
            }
            if per_doc
            else {}
        ),
        "line_level": (
            {
                "jaccard": _mean([d["line_level"]["jaccard"] for d in per_doc]),
                "coverage": _mean([d["line_level"]["coverage"] for d in per_doc]),
                "leakage": _mean([d["line_level"]["leakage"] for d in per_doc]),
            }
            if per_doc
            else {}
        ),
        "counts": {
            "pred_avg": _mean([d["counts"]["pred"] for d in per_doc]),
            "gold_avg": _mean([d["counts"]["gold"] for d in per_doc]),
        },
    }

    return {
        "micro": {
            "span_prf": micro_span_prf,
            "soft_prf": micro_soft,
            "boundary_errors": micro_boundary,
            "line_level": micro_line,
            "counts": {"pred_total": pred_count_sum, "gold_total": gold_count_sum},
        },
        "macro": macro,
        "per_doc": per_doc,
    }


# ---------- Example ----------
if __name__ == "__main__":
    corpus = [
        ([(12, 14), (16, 20), (33, 45)], [(12, 15), (18, 22), (30, 40)]),
        ([(5, 8)], [(3, 7), (10, 12)]),
        ([], [(1, 2)]),
    ]
    report = aggregate_corpus(corpus, iou_thresholds=(0.3, 0.5, 0.7))
    pprint.pprint(report["micro"], width=100)
    pprint.pprint(report["macro"], width=100)
