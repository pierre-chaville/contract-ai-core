from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def load_labels_from_csv(path: Path) -> tuple[dict[int, str], dict[int, float]]:
    """Load index -> clause_key and index -> confidence_percent from a classification CSV.

    Expects columns: index, clause_key, confidence, text
    Returns: (index_to_label, index_to_confidence_percent)
    """
    index_to_label: dict[int, str] = {}
    index_to_conf: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int((row.get("index") or "").strip())
            except Exception:
                continue
            label = (row.get("clause_key") or "").strip() or "NONE"
            conf_raw = (row.get("confidence") or "").strip()
            conf_val: float
            try:
                conf_val = float(conf_raw)
            except Exception:
                conf_val = float("nan")
            index_to_label[idx] = label
            index_to_conf[idx] = conf_val
    return index_to_label, index_to_conf


def align_gold_pred(gold: dict[int, str], pred: dict[int, str]) -> tuple[list[str], list[str]]:
    """Align gold and predicted labels by index (intersection only)."""
    common = sorted(set(gold.keys()) & set(pred.keys()))
    y_true = [gold[i] for i in common]
    y_pred = [pred[i] for i in common]
    return y_true, y_pred


def write_confusion_matrix_csv(out_csv: Path, labels: list[str], cm: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))


def write_confusion_matrix_png(out_png: Path, labels: list[str], cm: np.ndarray) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    # Ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate cells
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clause classification metrics")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument(
        "--model", required=True, help="Model name used for predictions (folder name)"
    )
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]

    gold_dir = repo_root / "dataset" / "gold" / "clauses" / template_key
    pred_dir = repo_root / "dataset" / "output" / "clauses" / template_key / model_name
    out_dir = repo_root / "dataset" / "metrics" / "classification" / template_key / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_files = {p.stem: p for p in gold_dir.glob("*.csv")}
    pred_files = {p.stem: p for p in pred_dir.glob("*.csv")}

    common_docs = sorted(set(gold_files.keys()) & set(pred_files.keys()))
    if not common_docs:
        print("No overlapping files between gold and predictions.")
        return

    all_true: list[str] = []
    all_pred: list[str] = []
    all_pred_conf: list[float] = []

    for stem in common_docs:
        gold_labels, _ = load_labels_from_csv(gold_files[stem])
        pred_labels, pred_conf = load_labels_from_csv(pred_files[stem])
        y_true, y_pred = align_gold_pred(gold_labels, pred_labels)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        # align confidences to the same order
        indices = sorted(set(gold_labels.keys()) & set(pred_labels.keys()))
        all_pred_conf.extend([pred_conf.get(i, float("nan")) for i in indices])

    # Label set across both gold and predictions
    labels = sorted(set(all_true) | set(all_pred))

    # Accuracy
    accuracy = float(accuracy_score(all_true, all_pred)) if all_true else 0.0

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=labels)

    # High-confidence accuracy (threshold in percent)
    threshold = 80.0
    hc_mask = [not np.isnan(c) and c >= threshold for c in all_pred_conf]
    hc_true = [t for t, m in zip(all_true, hc_mask, strict=False) if m]
    hc_pred = [p for p, m in zip(all_pred, hc_mask, strict=False) if m]
    if hc_true:
        high_conf_accuracy = float(accuracy_score(hc_true, hc_pred))
        coverage = float(len(hc_true)) / float(len(all_true))
    else:
        high_conf_accuracy = float("nan")
        coverage = 0.0

    # Cohen's Kappa
    kappa = float(cohen_kappa_score(all_true, all_pred, labels=labels)) if all_true else 0.0

    # Per-category detailed metrics
    macro_f1 = float(f1_score(all_true, all_pred, labels=labels, average="macro", zero_division=0))
    # Compute TP, FP, FN, TN per label from confusion matrix
    total = int(cm.sum()) if cm.size else 0
    per_category_list: list[dict[str, float | int | str]] = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i]) if cm.size else 0
        fn = int(cm[i, :].sum() - tp) if cm.size else 0
        fp = int(cm[:, i].sum() - tp) if cm.size else 0
        tn = int(total - tp - fp - fn) if cm.size else 0
        support = tp + fn
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy_cat = float((tp + tn) / total) if total > 0 else 0.0
        per_category_list.append(
            {
                "category": label,
                "accuracy": accuracy_cat,
                "count": support,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    # Write outputs
    summary = {
        "Number examples": len(all_true),
        "Accuracy": accuracy,
        "Cohen's Kappa": kappa,
        "High confidence threshold percent": threshold,
        "High confidence accuracy": high_conf_accuracy,
        "High confidence coverage": coverage,
        "Macro F1": macro_f1,
        "per_category": per_category_list,
    }
    with (out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    write_confusion_matrix_csv(out_dir / "confusion_matrix.csv", labels, cm)
    write_confusion_matrix_png(out_dir / "confusion_matrix.png", labels, cm)

    # Pretty print key metrics as percentages
    def _to_pct(val: float) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "nan"
        return f"{int(round(float(val) * 100))}%"

    summary_pct = {
        "Num examples": len(all_true),
        "Accuracy": _to_pct(accuracy),
        "cohen_kappa": _to_pct(kappa),
        "High confidence threshold percent": threshold,
        "High confidence accuracy": _to_pct(high_conf_accuracy),
        "High confidence coverage": _to_pct(coverage),
        "Macro F1": _to_pct(macro_f1),
        "per_category": [
            {
                "category": it["category"],
                "accuracy": _to_pct(it["accuracy"]),
                "count": it["count"],
                "f1": _to_pct(it["f1"]),
                "precision": _to_pct(it["precision"]),
                "recall": _to_pct(it["recall"]),
            }
            for it in per_category_list
        ],
    }
    print(yaml.safe_dump(summary_pct, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
