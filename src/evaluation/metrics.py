"""
Evaluation metrics for fraud detection.
NEVER use accuracy — always PR-AUC, F1, ROC-AUC, Precision@80%Recall, FPR.
"""

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
    Compute all fraud detection metrics.

    Args:
        labels: Ground truth binary labels (0/1)
        scores: Predicted fraud probabilities [0, 1]
        threshold: Classification threshold for F1/FPR
        target_recall: Recall level for Precision@Recall metric

    Returns:
        dict with keys: pr_auc, roc_auc, f1, precision_at_recall_X, fpr, threshold_used
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    metrics = {}

    # PR-AUC (primary metric)
    metrics["pr_auc"] = float(average_precision_score(labels, scores))

    # ROC-AUC
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")

    # F1 at threshold
    preds = (scores >= threshold).astype(int)
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
    metrics["threshold_used"] = threshold

    # False Positive Rate at threshold
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Precision@target_recall (e.g., Precision@80%Recall)
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    # Find the threshold where recall is closest to target_recall
    idx = np.searchsorted(recall_curve[::-1], target_recall)
    idx = len(recall_curve) - 1 - idx
    idx = max(0, min(idx, len(precision_curve) - 1))
    metrics[f"precision_at_recall_{int(target_recall * 100)}"] = float(precision_curve[idx])

    return metrics


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray,
                         target_recall: float = 0.80) -> float:
    """
    Find the classification threshold that achieves `target_recall`.
    Used after training, on the validation set.
    """
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    # recall_curve is in descending order; thresholds has len = len(recall_curve) - 1
    for i, recall in enumerate(recall_curve[:-1]):
        if recall <= target_recall:
            return float(thresholds[i])
    return float(thresholds[-1])


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty-print all metrics."""
    print(f"{prefix}PR-AUC:   {metrics.get('pr_auc', float('nan')):.4f}")
    print(f"{prefix}ROC-AUC:  {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"{prefix}F1:       {metrics.get('f1', float('nan')):.4f}")
    print(f"{prefix}FPR:      {metrics.get('fpr', float('nan')):.4f}")
    for k, v in metrics.items():
        if k.startswith("precision_at_recall"):
            recall_pct = k.split("_")[-1]
            print(f"{prefix}Prec@{recall_pct}%R: {v:.4f}")
