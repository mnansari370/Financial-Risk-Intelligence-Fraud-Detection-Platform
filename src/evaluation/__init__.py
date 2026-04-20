"""
Evaluation utilities.

  metrics.py           — compute_metrics, calibrate_threshold, print_metrics
  evaluate.py          — per-model test-set evaluation (GAT, XGBoost, anomaly)
  complementarity.py   — ensemble, Jaccard overlap, unique fraud catches

PR-AUC is the primary metric throughout; accuracy and ROC-AUC are reported
for completeness but should not be used for model selection on this dataset
(0.13% fraud rate means all-legitimate achieves 99.87% accuracy).
"""

from src.evaluation.metrics import compute_metrics, calibrate_threshold, print_metrics

__all__ = ["compute_metrics", "calibrate_threshold", "print_metrics"]
