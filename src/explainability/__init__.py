"""
Model explainability.

  shap_explainer.py — SHAP TreeExplainer for XGBoost;
                      gradient × input saliency for FraudEdgeGAT edge features

Both produce a per-transaction feature attribution table (parquet) and a
global mean-|SHAP| feature importance CSV, consumed by the dashboard's
Alert Detail tab.
"""
