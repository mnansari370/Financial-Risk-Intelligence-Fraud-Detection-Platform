import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(labels: np.ndarray, scores: np.ndarray,
                    threshold: float = 0.5,
                    target_recall: float = 0.80) -> dict:
    """
    PR-AUC is the primary metric for imbalanced fraud detection.
    Also computes ROC-AUC, F1 at threshold, FPR, and Precision@target_recall.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    metrics = {}
    metrics["pr_auc"] = float(average_precision_score(labels, scores))

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")

    preds = (scores >= threshold).astype(int)
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    metrics["threshold_used"] = threshold

    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    idx = np.searchsorted(recall_curve[::-1], target_recall)
    idx = len(recall_curve) - 1 - idx
    idx = max(0, min(idx, len(precision_curve) - 1))
    metrics[f"precision_at_recall_{int(target_recall * 100)}"] = float(precision_curve[idx])

    return metrics


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray,
                         target_recall: float = 0.80) -> float:
    """Find the classification threshold that achieves target_recall on a given set."""
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    for i, recall in enumerate(recall_curve[:-1]):
        if recall <= target_recall:
            return float(thresholds[i])
    return float(thresholds[-1])


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(f"{prefix}PR-AUC:   {metrics.get('pr_auc', float('nan')):.4f}")
    print(f"{prefix}ROC-AUC:  {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"{prefix}F1:       {metrics.get('f1', float('nan')):.4f}")
    print(f"{prefix}FPR:      {metrics.get('fpr', float('nan')):.4f}")
    for k, v in metrics.items():
        if k.startswith("precision_at_recall"):
            recall_pct = k.split("_")[-1]
            print(f"{prefix}Prec@{recall_pct}%R: {v:.4f}")
