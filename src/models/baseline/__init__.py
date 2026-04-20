"""
XGBoost baseline pipeline.

Tabular fraud classification with:
  - strict temporal 80/10/10 train/val/test split
  - SMOTE oversampling on the training split only
  - threshold calibration at 80% recall on the validation split
  - scale_pos_weight derived from actual class imbalance

Run via:  python -m src.models.baseline.xgboost_pipeline
Or SLURM: sbatch slurm/train_xgboost.sh
"""
