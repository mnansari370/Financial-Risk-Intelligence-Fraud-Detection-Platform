"""
Data augmentation utilities used during training.

  apply_smote             — SMOTE oversampling for XGBoost (train split only)
  compute_scale_pos_weight — XGBoost class-weight helper
"""

from src.augmentation.augmentation import apply_smote, compute_scale_pos_weight

__all__ = ["apply_smote", "compute_scale_pos_weight"]
