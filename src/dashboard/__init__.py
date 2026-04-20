"""
Streamlit monitoring dashboard.

Five tabs:
  1. Alert Queue      — model performance table and live alert feed
  2. Alert Detail     — per-transaction SHAP waterfall chart
  3. Graph View       — 1-hop PyVis subgraph around a selected transaction
  4. SAR Generator    — GPT-4o-mini EU AMLD6-compliant report generation
  5. Scenario Simulator — XGBoost what-if feature sensitivity analysis

Launch: streamlit run src/dashboard/app.py
"""
