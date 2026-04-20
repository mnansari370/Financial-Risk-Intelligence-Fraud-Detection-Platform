"""
Financial Fraud Detection Platform — source root.

Sub-packages:
  graph          — PaySim graph construction and temporal splitting
  models         — FraudEdgeGAT, XGBoost baseline, anomaly detectors
  evaluation     — metrics, per-model evaluation, complementarity analysis
  explainability — SHAP and gradient saliency for XGBoost and GAT
  augmentation   — SMOTE utilities and focal-loss helpers
  llm            — GPT-4o-mini SAR report generator
  dashboard      — Streamlit monitoring dashboard
"""
