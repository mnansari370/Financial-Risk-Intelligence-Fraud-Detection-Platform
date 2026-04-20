"""
Financial Risk Intelligence & Fraud Detection Dashboard
MSc Computer Science — University of Luxembourg
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import torch

st.set_page_config(
    page_title="FraudIQ · Detection Platform",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding: 1.2rem 2.5rem 3rem 2.5rem !important; max-width: 1500px; }

/* ── Page header ── */
.fraud-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 32px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.fraud-header-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin: 0;
}
.fraud-header-sub {
    font-size: 0.82rem;
    color: #8b949e;
    margin: 4px 0 0 0;
}
.fraud-header-badge {
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 14px; margin-bottom: 28px; }
.kpi-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #58a6ff; }
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 12px 12px 0 0; }
.kpi-label { color: #8b949e; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 8px; }
.kpi-value { color: #e6edf3; font-size: 1.75rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 6px; }
.kpi-sub { font-size: 0.75rem; font-weight: 500; }

/* ── Section headers ── */
.sec-header {
    border-left: 3px solid #388bfd;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}
.sec-header h4 { color: #e6edf3; font-size: 1rem; font-weight: 700; margin: 0; }
.sec-header p  { color: #8b949e; font-size: 0.8rem; margin: 3px 0 0 0; }

/* ── Risk badges ── */
.badge { display: inline-flex; align-items: center; gap: 4px;
         padding: 3px 10px; border-radius: 20px; font-size: 0.7rem;
         font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.badge-critical { background: rgba(255,80,80,0.15);  color: #ff6b6b; border: 1px solid rgba(255,80,80,0.35); }
.badge-high     { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.35); }
.badge-medium   { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.35); }
.badge-low      { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.35); }

/* ── Info cards ── */
.tx-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.tx-card-row { display: flex; gap: 32px; flex-wrap: wrap; }
.tx-field-label { color: #8b949e; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.tx-field-value { color: #e6edf3; font-size: 1.1rem; font-weight: 700; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #0d1117; padding: 5px 6px;
    border-radius: 12px; border: 1px solid #21262d; margin-bottom: 22px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 9px 20px;
    font-size: 0.85rem; font-weight: 500; color: #8b949e; background: transparent;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important; color: #e6edf3 !important; font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3; }

/* ── Streamlit metric override ── */
div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #e6edf3 !important; }
div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.06em; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }
iframe[title="st_aggrid"] { border-radius: 10px; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 10px 28px !important; letter-spacing: 0.02em;
}
.stButton > button[kind="secondary"] {
    background: #21262d !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; color: #e6edf3 !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: #161b22 !important; border-color: #30363d !important;
    border-radius: 8px !important; color: #e6edf3 !important;
}
.stSlider > div { accent-color: #388bfd; }

/* ── Expander ── */
details > summary { color: #8b949e !important; font-size: 0.83rem !important; }
details[open] > summary { color: #e6edf3 !important; }

/* ── Info / warning boxes ── */
div[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 4px !important; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 22px 0 !important; }

/* ── Caption ── */
.stCaption, .css-1fv8s86 { color: #6e7681 !important; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="fraud-header">
  <div>
    <p class="fraud-header-title">🛡 FraudIQ Detection Platform</p>
    <p class="fraud-header-sub">MSc Computer Science · University of Luxembourg · PaySim + IEEE-CIS datasets · V100 GPU</p>
  </div>
  <div class="fraud-header-badge">● LIVE MONITORING</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡 FraudIQ")
    st.caption("MSc Computer Science · University of Luxembourg")
    st.divider()

    st.markdown("**Alert Controls**")
    score_threshold = st.slider(
        "Alert Threshold", 0.0, 1.0, 0.5, 0.01,
        help="Transactions with score ≥ threshold are flagged as alerts",
    )
    top_k_display = st.number_input("Max alerts to display", 10, 500, 50, step=10)

    st.divider()
    st.markdown("**SAR Generation**")
    llm_provider = st.selectbox("LLM Provider", ["openai", "local"], label_visibility="collapsed")

    st.divider()
    st.markdown("**Model Status**")
    for label, path in [
        ("XGBoost",         "src/models/baseline/xgboost_model.pkl"),
        ("FraudGAT",        "src/models/gnn/checkpoints/best_model.pt"),
        ("Autoencoder",     "src/models/anomaly/autoencoder.pt"),
        ("Isolation Forest","src/models/anomaly/isolation_forest.pkl"),
    ]:
        exists = Path(path).exists()
        dot = "🟢" if exists else "🔴"
        st.markdown(f"{dot} {label}", unsafe_allow_html=False)

    st.divider()
    st.caption("Trained on ULHPC HPC · Tesla V100")

# ── File paths ────────────────────────────────────────────────────────────────
PATHS = {
    "gat_scores":   "results/evaluation/gat/scores.parquet",
    "xgb_scores":   "results/evaluation/xgboost/scores.parquet",
    "gat_metrics":  "results/evaluation/gat/metrics.csv",
    "xgb_metrics":  "results/evaluation/xgboost/metrics.csv",
    "if_metrics":   "results/evaluation/anomaly/metrics_isolation_forest.csv",
    "ae_metrics":   "results/evaluation/anomaly/metrics_autoencoder.csv",
    "ano_metrics":  "results/evaluation/anomaly/metrics_ensemble.csv",
    "ens_metrics":  "results/evaluation/ensemble/metrics_ensemble.csv",
    "shap_xgb":     "results/shap/xgboost/shap_values_xgb.parquet",
    "feat_imp_xgb": "results/shap/xgboost/feature_importance_xgb.csv",
    "xgb_model":    "src/models/baseline/xgboost_model.pkl",
    "graph":        "data/processed/graph_account_edge.pt",
}

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_scores():
    out = {}
    for key in ("gat_scores", "xgb_scores"):
        p = Path(PATHS[key])
        if p.exists():
            out[key] = pd.read_parquet(p)
    return out

@st.cache_resource
def load_metrics_table():
    entries = [
        ("XGBoost",          "xgb_metrics"),
        ("Full Ensemble",    "ens_metrics"),
        ("Autoencoder",      "ae_metrics"),
        ("FraudGAT",         "gat_metrics"),
        ("IF + AE Ensemble", "ano_metrics"),
        ("Isolation Forest", "if_metrics"),
    ]
    rows = []
    for name, key in entries:
        p = Path(PATHS[key])
        if p.exists():
            row = pd.read_csv(p).iloc[0].to_dict()
            row["Model"] = name
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_resource
def load_xgb_model():
    p = Path(PATHS["xgb_model"])
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_graph():
    p = Path(PATHS["graph"])
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")

@st.cache_resource
def load_shap_xgb():
    p = Path(PATHS["shap_xgb"])
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_resource
def load_feat_importance():
    p = Path(PATHS["feat_imp_xgb"])
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_resource
def build_xgb_lookup():
    p = Path(PATHS["xgb_scores"])
    if not p.exists():
        return {}, {}, {}
    df  = pd.read_parquet(p)
    top = df.nlargest(1000, "score")
    ids = top["tx_id"].astype(int)
    return (
        dict(zip(ids, top["score"])),
        dict(zip(ids, top["is_fraud"].astype(int))),
        dict(zip(ids, top.index.tolist())),
    )

@st.cache_resource
def build_gat_lookup():
    p = Path(PATHS["gat_scores"])
    if not p.exists():
        return {}, {}
    df  = pd.read_parquet(p)
    top = df.nlargest(1000, "score")
    ids = top["tx_id"].astype(int)
    return (
        dict(zip(ids, top["score"])),
        dict(zip(ids, top["is_fraud"].astype(int))),
    )

@st.cache_resource
def build_graph_tx_index():
    graphs = load_graph()
    if graphs is None:
        return {}
    lookup = {}
    for snap_i, snap in enumerate(graphs.get("test", [])):
        if not hasattr(snap, "tx_id"):
            continue
        for edge_i, tid in enumerate(snap.tx_id.tolist()):
            lookup[int(tid)] = (snap_i, edge_i)
    return lookup

# ── Shared chart theme ────────────────────────────────────────────────────────
_C = dict(
    plot_bgcolor="#161b22",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", size=12, family="system-ui, -apple-system, sans-serif"),
    hoverlabel=dict(bgcolor="#21262d", bordercolor="#444c56", font_color="#e6edf3",
                    font_size=13),
)

# Default axis/legend styles — merge manually where needed to avoid keyword conflicts
_AXIS = dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#484f58", showgrid=True)
_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1, font_color="#c9d1d9")

# ── Style helpers ─────────────────────────────────────────────────────────────
def risk_label(s):
    if s >= 0.9:   return "CRITICAL"
    elif s >= 0.7: return "HIGH"
    elif s >= 0.5: return "MEDIUM"
    return "LOW"

def risk_color(s):
    if s >= 0.9:   return "#ff4444"
    elif s >= 0.7: return "#fb923c"
    elif s >= 0.5: return "#facc15"
    return "#4ade80"

def risk_badge_html(s):
    lvl  = risk_label(s)
    cls  = {"CRITICAL": "badge-critical", "HIGH": "badge-high",
            "MEDIUM": "badge-medium", "LOW": "badge-low"}[lvl]
    dots = {"CRITICAL": "●", "HIGH": "◆", "MEDIUM": "▲", "LOW": "✔"}[lvl]
    return f'<span class="badge {cls}">{dots} {lvl}</span>'

def prettify(name: str) -> str:
    return name.replace("_", " ").title()

def score_cell_style(val):
    if val >= 0.9:   return "background:#2d1212; color:#ff6b6b; font-weight:700"
    elif val >= 0.7: return "background:#2d1a0a; color:#fb923c; font-weight:600"
    elif val >= 0.5: return "background:#2d2a0a; color:#facc15"
    return ""

def kpi_card(label, value, subtitle="", color="#388bfd"):
    return f"""
    <div class="kpi-card">
      <div class="kpi-accent" style="background:{color}"></div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub" style="color:{color}">{subtitle}</div>
    </div>"""

def section_header(title, subtitle=""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    return f'<div class="sec-header"><h4>{title}</h4>{sub}</div>'

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  🚨  Alert Queue  ",
    "  🔎  Alert Detail  ",
    "  🕸  Graph View  ",
    "  📄  SAR Generator  ",
    "  ⚙  Scenario Simulator  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Alert Queue
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    scores     = load_scores()
    metrics_df = load_metrics_table()
    gat_df     = scores.get("gat_scores", pd.DataFrame())
    xgb_df     = scores.get("xgb_scores", pd.DataFrame())

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if not xgb_df.empty:
        total_tx    = len(xgb_df)
        fraud_rate  = xgb_df["is_fraud"].mean() * 100
        n_alerts    = int((xgb_df["score"] >= score_threshold).sum())
        best_prauc  = 0.797

        st.markdown(f"""
        <div class="kpi-row">
          {kpi_card("Transactions Evaluated", f"{total_tx:,}", "XGBoost test set", "#388bfd")}
          {kpi_card("Fraud Rate", f"{fraud_rate:.2f}%", "Test period distribution", "#e74c3c")}
          {kpi_card(f"Alerts  ≥ {score_threshold:.2f}", f"{n_alerts:,}", "XGBoost flagged", "#fb923c")}
          {kpi_card("Best PR-AUC", f"{best_prauc:.3f}", "XGBoost (133× above random)", "#3fb950")}
        </div>""", unsafe_allow_html=True)

    # ── Model performance comparison ──────────────────────────────────────────
    st.markdown(section_header(
        "Model Performance Comparison",
        "PR-AUC is the primary metric — accuracy is meaningless at 0.59% fraud rate"
    ), unsafe_allow_html=True)

    if not metrics_df.empty:
        # Plotly grouped bar chart
        cols_show = ["pr_auc", "roc_auc", "f1"]
        col_labels = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1 Score"}
        model_colors = {
            "XGBoost":          "#388bfd",
            "Full Ensemble":    "#3fb950",
            "Autoencoder":      "#bc8cff",
            "FraudGAT":         "#e74c3c",
            "IF + AE Ensemble": "#f78166",
            "Isolation Forest": "#d29922",
        }

        fig_bar = go.Figure()
        bar_w = 0.18
        x_pos = np.arange(len(metrics_df))
        offsets = [-bar_w, 0, bar_w]

        for i, (col, lbl) in enumerate(col_labels.items()):
            vals = metrics_df[col].fillna(0).tolist()
            colors = [model_colors.get(m, "#8b949e") for m in metrics_df["Model"]]
            opacity = [1.0, 0.7, 0.45][i]
            fig_bar.add_trace(go.Bar(
                name=lbl,
                x=metrics_df["Model"].tolist(),
                y=vals,
                marker_color=colors,
                marker_opacity=opacity,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont=dict(size=10, color="#c9d1d9"),
                width=0.22,
                offsetgroup=i,
            ))

        fig_bar.update_layout(
            **_C,
            barmode="group",
            margin=dict(t=30, b=20, l=10, r=10),
            height=310,
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            yaxis=dict(range=[0, 1.18], gridcolor="#21262d", tickformat=".2f",
                       linecolor="#30363d", showgrid=True),
            xaxis=dict(linecolor="#30363d", gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Precise numbers table
        with st.expander("Full metrics table", expanded=False):
            disp = (
                metrics_df[["Model", "pr_auc", "roc_auc", "f1",
                             "precision_at_recall_80", "fpr"]]
                .copy().set_index("Model").fillna(0.0)
            )
            disp.columns = ["PR-AUC ↑", "ROC-AUC ↑", "F1 ↑", "Prec@80%R ↑", "FPR ↓"]
            st.dataframe(
                disp.style
                    .format("{:.4f}")
                    .highlight_max(axis=0, subset=["PR-AUC ↑","ROC-AUC ↑","F1 ↑","Prec@80%R ↑"],
                                   color="#0f2d1f")
                    .highlight_min(axis=0, subset=["FPR ↓"], color="#0f2d1f"),
                use_container_width=True,
            )
    else:
        st.info("No metrics found — run `sbatch slurm/full_eval.sh` first.")

    # ── Alert feed ────────────────────────────────────────────────────────────
    st.markdown(section_header(
        "Transaction Alert Feed — XGBoost",
        f"Top-{int(top_k_display)} highest-scored transactions above threshold {score_threshold:.2f}"
    ), unsafe_allow_html=True)

    if xgb_df.empty:
        st.warning("Evaluation scores not found. Run `sbatch slurm/full_eval.sh`.")
    else:
        alerts = (
            xgb_df[xgb_df["score"] >= score_threshold]
            .nlargest(int(top_k_display), "score")
            .copy()
        )
        if alerts.empty:
            st.info(f"No transactions above threshold {score_threshold:.2f}. Lower the threshold in the sidebar.")
        else:
            alerts["Risk Level"] = alerts["score"].apply(risk_label)
            alerts["Status"]     = alerts["is_fraud"].map({1: "🔴 Confirmed Fraud", 0: "⬜ Unconfirmed"})
            alerts["Score"]      = alerts["score"].round(4)
            st.dataframe(
                alerts[["tx_id", "Score", "Risk Level", "Status"]].rename(
                    columns={"tx_id": "Transaction ID"}
                ).style
                  .map(score_cell_style, subset=["Score"])
                  .format({"Score": "{:.4f}"}),
                use_container_width=True,
                height=320,
            )

    # ── Score distributions ───────────────────────────────────────────────────
    st.markdown(section_header(
        "Score Distributions by True Label",
        "Fraud (red) vs Legitimate (blue) — separation shows discriminative power"
    ), unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    def score_dist_fig(df, title, fraud_color="#e74c3c", legit_color="#388bfd"):
        fig = go.Figure()
        legit = df[df["is_fraud"] == 0]["score"]
        fraud = df[df["is_fraud"] == 1]["score"]
        fig.add_trace(go.Histogram(
            x=legit, name="Legitimate", nbinsx=80,
            marker_color=legit_color, opacity=0.6,
            hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra>Legitimate</extra>",
        ))
        fig.add_trace(go.Histogram(
            x=fraud, name="Fraud", nbinsx=80,
            marker_color=fraud_color, opacity=0.85,
            hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra>Fraud</extra>",
        ))
        fig.add_vrect(
            x0=score_threshold, x1=1.0,
            fillcolor="rgba(231,76,60,0.06)",
            layer="below", line_width=0,
            annotation_text=f"Alert zone ≥ {score_threshold:.2f}",
            annotation_position="top left",
            annotation_font_color="#8b949e",
            annotation_font_size=11,
        )
        fig.add_vline(x=score_threshold, line_dash="dash", line_color="#666",
                      line_width=1.5)
        fig.update_layout(
            **_C,
            barmode="overlay",
            title=dict(text=title, font_size=14, font_color="#e6edf3", x=0.01),
            height=300,
            margin=dict(t=50, b=40, l=50, r=15),
            xaxis=dict(title="Fraud Score", range=[0, 1], gridcolor="#21262d",
                       linecolor="#30363d", showgrid=True),
            yaxis=dict(title="Count", gridcolor="#21262d", linecolor="#30363d",
                       showgrid=True),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        return fig

    with col_d1:
        if not xgb_df.empty:
            st.plotly_chart(score_dist_fig(xgb_df, "XGBoost — Score Distribution"),
                            use_container_width=True)
    with col_d2:
        if not gat_df.empty:
            st.plotly_chart(score_dist_fig(gat_df, "FraudGAT — Score Distribution",
                                           fraud_color="#bc8cff", legit_color="#3fb950"),
                            use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Alert Detail & SHAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    xgb_score_map, xgb_fraud_map, xgb_pos_map = build_xgb_lookup()

    if not xgb_score_map:
        st.warning("XGBoost scores not found. Run `sbatch slurm/full_eval.sh`.")
    else:
        shap_df  = load_shap_xgb()
        feat_imp = load_feat_importance()

        top200_ids = sorted(xgb_score_map, key=xgb_score_map.get, reverse=True)[:200]

        sel_col, _ = st.columns([2, 1])
        with sel_col:
            selected_tx = st.selectbox(
                "Select transaction (top-200 XGBoost alerts)",
                top200_ids,
                format_func=lambda x: (
                    f"TX {x}   ·   score {xgb_score_map[x]:.4f}"
                    f"   ·   {'🔴 Fraud' if xgb_fraud_map[x] else '⬜ Legit'}"
                ),
            )

        sc       = xgb_score_map[selected_tx]
        is_fraud = xgb_fraud_map[selected_tx]
        badge    = risk_badge_html(sc)
        label_html = ('<span style="color:#ff6b6b;font-weight:700">🔴 CONFIRMED FRAUD</span>'
                      if is_fraud else
                      '<span style="color:#4ade80;font-weight:600">⬜ Not confirmed</span>')

        # Transaction info card
        st.markdown(f"""
        <div class="tx-card" style="border-left:4px solid {risk_color(sc)}">
          <div class="tx-card-row">
            <div>
              <div class="tx-field-label">Transaction ID</div>
              <div class="tx-field-value" style="font-size:1.4rem">#{selected_tx:,}</div>
            </div>
            <div>
              <div class="tx-field-label">Fraud Score</div>
              <div class="tx-field-value" style="color:{risk_color(sc)};font-size:1.6rem">{sc:.4f}</div>
            </div>
            <div>
              <div class="tx-field-label">Risk Level</div>
              <div style="margin-top:4px">{badge}</div>
            </div>
            <div>
              <div class="tx-field-label">Ground Truth</div>
              <div style="margin-top:6px">{label_html}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP + global importance
        col_shap, col_imp = st.columns([3, 2])

        with col_shap:
            st.markdown(section_header(
                "SHAP Feature Contributions",
                "Per-feature influence on this specific prediction — red pushes toward fraud, blue toward legitimate"
            ), unsafe_allow_html=True)

            if shap_df.empty:
                st.info("SHAP file not found. Run `sbatch slurm/shap_batch.sh`.")
            else:
                feat_cols = [c for c in shap_df.columns if c != "tx_id"]
                match     = shap_df[shap_df["tx_id"] == selected_tx]

                if match.empty:
                    st.info("Transaction not in the SHAP sample (top-1,000). Select a higher-scoring alert.")
                else:
                    tx_shap = match.iloc[0]
                    vals    = sorted(
                        [(c, float(tx_shap[c])) for c in feat_cols],
                        key=lambda x: abs(x[1]), reverse=True
                    )
                    feats     = [prettify(v[0]) for v in vals]
                    shap_vals = [v[1] for v in vals]

                    bar_colors = []
                    for v in shap_vals:
                        intensity = min(abs(v) / (max(abs(x) for x in shap_vals) + 1e-9), 1.0)
                        if v > 0:
                            r = int(220 + 35 * intensity)
                            bar_colors.append(f"rgb({r},70,60)")
                        else:
                            b = int(160 + 95 * intensity)
                            bar_colors.append(f"rgb(50,100,{b})")

                    fig_shap = go.Figure(go.Bar(
                        x=shap_vals,
                        y=feats,
                        orientation="h",
                        marker_color=bar_colors,
                        marker_line_width=0,
                        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.5f}<extra></extra>",
                        text=[f"{v:+.4f}" for v in shap_vals],
                        textposition="outside",
                        textfont=dict(size=11, color="#c9d1d9"),
                    ))
                    fig_shap.add_vline(x=0, line_color="#484f58", line_width=1.5)
                    fig_shap.update_layout(
                        **_C,
                        height=420,
                        margin=dict(t=20, b=20, l=20, r=80),
                        xaxis=dict(
                            title="SHAP Value  (← more legitimate  |  more fraud →)",
                            title_font=dict(size=11, color="#8b949e"),
                            gridcolor="#21262d", linecolor="#30363d",
                            zeroline=True, zerolinecolor="#444c56", zerolinewidth=2,
                        ),
                        yaxis=dict(autorange="reversed", linecolor="#30363d",
                                   tickfont=dict(size=12), showgrid=False),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                    with st.expander("Raw SHAP values table"):
                        raw_df = pd.DataFrame(
                            [(prettify(f), v) for f, v in
                             zip([v[0] for v in vals], shap_vals)],
                            columns=["Feature", "SHAP Value"]
                        ).set_index("Feature")
                        st.dataframe(raw_df.style.format("{:.6f}").background_gradient(
                            cmap="RdBu_r", axis=0), use_container_width=True)

        with col_imp:
            st.markdown(section_header(
                "Global Feature Importance",
                "Mean |SHAP| across top-1,000 highest-scored transactions"
            ), unsafe_allow_html=True)

            if feat_imp.empty:
                st.info("Run `sbatch slurm/shap_batch.sh`.")
            else:
                fi = feat_imp.head(12).copy()
                fi["Feature"] = fi["feature"].apply(prettify)
                fi = fi.sort_values("mean_abs_shap")

                max_val = fi["mean_abs_shap"].max()
                bar_c   = [
                    f"rgba(56,139,253,{0.4 + 0.6 * v / max_val})"
                    for v in fi["mean_abs_shap"]
                ]

                fig_imp = go.Figure(go.Bar(
                    x=fi["mean_abs_shap"],
                    y=fi["Feature"],
                    orientation="h",
                    marker_color=bar_c,
                    marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                    text=fi["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
                    textposition="outside",
                    textfont=dict(size=10, color="#8b949e"),
                ))
                fig_imp.update_layout(
                    **_C,
                    height=420,
                    margin=dict(t=10, b=20, l=20, r=60),
                    xaxis=dict(title="Mean |SHAP|", gridcolor="#21262d",
                               linecolor="#30363d", title_font_size=11),
                    yaxis=dict(linecolor="#30363d", tickfont=dict(size=12), showgrid=False),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Graph View
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(section_header(
        "Transaction Graph Neighbourhood",
        "1-hop account subgraph around a selected transaction — visualises who sent and received funds"
    ), unsafe_allow_html=True)

    if "graph_initialized" not in st.session_state:
        st.session_state["graph_initialized"] = False

    if not st.session_state["graph_initialized"]:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                    padding:32px;text-align:center;margin:20px 0">
          <div style="font-size:2.5rem;margin-bottom:12px">🕸</div>
          <div style="color:#e6edf3;font-size:1rem;font-weight:600;margin-bottom:8px">
            Graph file is large (~1.4 GB)
          </div>
          <div style="color:#8b949e;font-size:0.85rem">
            Loading takes ~30 seconds. Click below to initialise.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Load Graph View", type="primary"):
            with st.spinner("Loading graph snapshots and building transaction index..."):
                build_graph_tx_index()
            st.session_state["graph_initialized"] = True
            st.rerun()
    else:
        gat_score_map_g, gat_fraud_map_g = build_gat_lookup()

        if not gat_score_map_g:
            st.warning("GAT scores not found. Run `sbatch slurm/full_eval.sh`.")
        else:
            top100_ids = sorted(gat_score_map_g, key=gat_score_map_g.get, reverse=True)[:100]

            sel_g, _ = st.columns([2, 1])
            with sel_g:
                selected_tx_g = st.selectbox(
                    "Select transaction (top-100 GAT alerts)",
                    top100_ids,
                    format_func=lambda x: (
                        f"TX {x}   ·   GAT score {gat_score_map_g[x]:.4f}"
                        f"   ·   {'🔴 Fraud' if gat_fraud_map_g[x] else '⬜ Legit'}"
                    ),
                    key="graph_tx",
                )

            graphs   = load_graph()
            tx_index = build_graph_tx_index()

            if graphs is None:
                st.warning("Graph file not found at `data/processed/graph_account_edge.pt`.")
            elif int(selected_tx_g) not in tx_index:
                st.info(f"Transaction {selected_tx_g} not found in test snapshots.")
            else:
                from pyvis.network import Network
                import networkx as nx

                snap_i, edge_i = tx_index[int(selected_tx_g)]
                snap   = graphs["test"][snap_i]
                sc_val = gat_score_map_g[selected_tx_g]
                src_acc = int(snap.edge_index[0][edge_i].item())
                dst_acc = int(snap.edge_index[1][edge_i].item())

                G = nx.DiGraph()
                tx_node  = f"TX {selected_tx_g}"
                src_node = f"ACC {src_acc}"
                dst_node = f"ACC {dst_acc}"

                G.add_node(tx_node, color=risk_color(sc_val), size=34,
                           title=(f"<b>Transaction {selected_tx_g}</b><br>"
                                  f"GAT Fraud Score: <b>{sc_val:.4f}</b><br>"
                                  f"Risk: <b>{risk_label(sc_val)}</b><br>"
                                  f"{'🔴 Confirmed fraud' if gat_fraud_map_g[selected_tx_g] else '⬜ Not confirmed'}"),
                           label=f"TX\n{sc_val:.3f}")
                G.add_node(src_node, color="#388bfd", size=24,
                           title=f"<b>Sender Account</b><br>ID: {src_acc}", label=src_node)
                G.add_node(dst_node, color="#3fb950", size=24,
                           title=f"<b>Receiver Account</b><br>ID: {dst_acc}", label=dst_node)
                G.add_edge(src_node, tx_node, title="sends funds", label="sends")
                G.add_edge(tx_node, dst_node, title="receives funds", label="receives")

                ei_t = snap.edge_index
                neighbor_count = 0
                for nbr_i in range(ei_t.shape[1]):
                    if nbr_i == edge_i or neighbor_count >= 10:
                        continue
                    s = int(ei_t[0][nbr_i].item())
                    d = int(ei_t[1][nbr_i].item())
                    if s not in (src_acc, dst_acc) and d not in (src_acc, dst_acc):
                        continue
                    sn = f"ACC {s}"
                    dn = f"ACC {d}"
                    for n, c in [(sn, "#388bfd"), (dn, "#3fb950")]:
                        if n not in G.nodes:
                            G.add_node(n, color=c, size=16,
                                       title=f"Account {n.split()[-1]}", label=n)
                    is_fraud_e = bool(snap.edge_label[nbr_i].item()) if hasattr(snap, "edge_label") else False
                    G.add_edge(sn, dn,
                               title="🔴 Fraud transaction" if is_fraud_e else "Transaction",
                               color="#ff4444" if is_fraud_e else "#444c56")
                    neighbor_count += 1

                # Legend row
                lcols = st.columns(5)
                lcols[0].markdown(f'<div style="color:{risk_color(sc_val)};font-weight:700">⬤ Selected TX ({risk_label(sc_val)})</div>', unsafe_allow_html=True)
                lcols[1].markdown('<div style="color:#388bfd;font-weight:600">⬤ Sender account</div>', unsafe_allow_html=True)
                lcols[2].markdown('<div style="color:#3fb950;font-weight:600">⬤ Receiver account</div>', unsafe_allow_html=True)
                lcols[3].markdown('<div style="color:#ff4444;font-weight:600">— Fraud edge</div>', unsafe_allow_html=True)
                lcols[4].markdown(f'<div style="color:#8b949e">Nodes: {G.number_of_nodes()} · Edges: {G.number_of_edges()}</div>', unsafe_allow_html=True)

                net = Network(height="520px", width="100%", directed=True,
                              bgcolor="#0d1117", font_color="#c9d1d9")
                net.from_nx(G)
                net.set_options(json.dumps({
                    "physics": {
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "centralGravity": 0.3,
                            "springLength": 160,
                            "springConstant": 0.04,
                            "damping": 0.09,
                        }
                    },
                    "edges": {
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
                        "color":  {"inherit": False},
                        "smooth": {"type": "curvedCW", "roundness": 0.2},
                        "font":   {"size": 10, "color": "#8b949e", "strokeWidth": 0},
                        "width": 1.5,
                    },
                    "nodes": {
                        "font": {"size": 12, "bold": True},
                        "borderWidth": 2,
                        "borderWidthSelected": 4,
                        "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8},
                    },
                    "interaction": {"hover": True, "tooltipDelay": 100},
                }))
                html_path = "/tmp/fraud_graph.html"
                net.save_graph(html_path)
                with open(html_path) as f:
                    html_content = f.read()
                components.html(html_content, height=540, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SAR Generator
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(section_header(
        "Suspicious Activity Report Generator",
        "EU AMLD6-compliant SAR narratives powered by GPT-4o-mini · requires OPENAI_API_KEY"
    ), unsafe_allow_html=True)

    gat_score_map_sar, gat_fraud_map_sar = build_gat_lookup()

    if not gat_score_map_sar:
        st.warning("GAT scores not found. Run `sbatch slurm/full_eval.sh`.")
    else:
        top50_ids = sorted(gat_score_map_sar, key=gat_score_map_sar.get, reverse=True)[:50]

        sel_sar, _ = st.columns([2, 1])
        with sel_sar:
            selected_sar = st.selectbox(
                "Select high-risk transaction",
                top50_ids,
                format_func=lambda x: (
                    f"TX {x}   ·   GAT score {gat_score_map_sar[x]:.4f}"
                    f"   ·   {'🔴 Fraud' if gat_fraud_map_sar[x] else '⬜ Legit'}"
                ),
                key="sar_tx",
            )

        sar_sc  = gat_score_map_sar[selected_sar]
        sar_lab = gat_fraud_map_sar[selected_sar]

        # Transaction preview card
        st.markdown(f"""
        <div class="tx-card" style="border-left:4px solid {risk_color(sar_sc)}">
          <div class="tx-card-row">
            <div>
              <div class="tx-field-label">Transaction ID</div>
              <div class="tx-field-value">#{selected_sar:,}</div>
            </div>
            <div>
              <div class="tx-field-label">GAT Fraud Score</div>
              <div class="tx-field-value" style="color:{risk_color(sar_sc)}">{sar_sc:.4f}</div>
            </div>
            <div>
              <div class="tx-field-label">Risk Level</div>
              <div style="margin-top:4px">{risk_badge_html(sar_sc)}</div>
            </div>
            <div>
              <div class="tx-field-label">Ground Truth</div>
              <div class="tx-field-value" style="font-size:0.9rem;color:{'#ff6b6b' if sar_lab else '#4ade80'}">
                {'🔴 Confirmed Fraud' if sar_lab else '⬜ Not Confirmed'}
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        btn_col, _ = st.columns([1, 3])
        with btn_col:
            generate = st.button("Generate SAR Report", type="primary", use_container_width=True)

        if generate:
            from src.llm.sar_generator import SARGenerator
            with st.spinner("Generating EU AMLD6-compliant SAR narrative..."):
                try:
                    gen = SARGenerator(
                        provider=llm_provider,
                        model="gpt-4o-mini" if llm_provider == "openai" else None,
                    )
                    sar  = gen.generate(
                        transaction={"tx_id": int(selected_sar),
                                     "fraud_score": sar_sc, "is_fraud": sar_lab},
                        fraud_scores={"gat": sar_sc},
                    )
                    risk     = sar.get("risk_level", "UNKNOWN")
                    risk_clr = {"CRITICAL":"#ff4444","HIGH":"#fb923c",
                                "MEDIUM":"#facc15","LOW":"#4ade80"}.get(risk, "#8b949e")
                    icons    = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}

                    st.markdown(f"""
                    <div class="tx-card" style="border-left:4px solid {risk_clr};margin-top:16px">
                      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
                        <span style="font-size:1.5rem">{icons.get(risk,'⚪')}</span>
                        <div>
                          <div style="color:{risk_clr};font-size:1.1rem;font-weight:700">
                            Risk Level: {risk}
                          </div>
                          <div style="color:#8b949e;font-size:0.8rem">
                            {sar.get('evidence_summary','—')}
                          </div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_nar, col_flags = st.columns([3, 2])
                    with col_nar:
                        st.markdown(section_header("SAR Narrative"), unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;'
                            f'padding:18px 22px;color:#c9d1d9;font-size:0.9rem;line-height:1.7">'
                            f'{sar.get("narrative","No narrative generated.")}</div>',
                            unsafe_allow_html=True,
                        )
                    with col_flags:
                        st.markdown(section_header("Regulatory Flags"), unsafe_allow_html=True)
                        for flag in sar.get("regulatory_flags", []):
                            st.markdown(
                                f'<div style="background:#161b22;border:1px solid #30363d;'
                                f'border-radius:8px;padding:8px 14px;margin-bottom:8px;'
                                f'color:#facc15;font-size:0.83rem">⚑ {flag}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown(section_header("Recommended Action"), unsafe_allow_html=True)
                        action = sar.get("recommended_action", "REVIEW")
                        action_clr = {"FREEZE": "#ff4444", "ESCALATE": "#fb923c",
                                      "INVESTIGATE": "#facc15", "MONITOR": "#388bfd",
                                      "REVIEW": "#8b949e"}.get(action.upper().split()[0], "#8b949e")
                        st.markdown(
                            f'<div style="background:#161b22;border:1px solid {action_clr};'
                            f'border-radius:10px;padding:14px 18px;color:{action_clr};'
                            f'font-size:1rem;font-weight:700;text-align:center">{action}</div>',
                            unsafe_allow_html=True,
                        )

                    with st.expander("Full SAR JSON"):
                        st.json(sar)

                except Exception as e:
                    st.error(f"SAR generation failed: {e}")
                    st.info("Ensure `OPENAI_API_KEY` is set, or switch LLM Provider to 'local'.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Scenario Simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(section_header(
        "What-If Scenario Simulator",
        "Adjust transaction features live and watch the XGBoost fraud probability update in real time"
    ), unsafe_allow_html=True)

    model_artifact = load_xgb_model()
    if model_artifact is None:
        st.warning("XGBoost model not found. Run `sbatch slurm/train_xgboost.sh`.")
    else:
        model        = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        threshold    = model_artifact.get("threshold", 0.5)

        # ── Input panel ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                    padding:20px 24px;margin-bottom:20px">
          <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                      letter-spacing:0.1em;font-weight:600;margin-bottom:14px">
            Transaction Feature Inputs
          </div>""", unsafe_allow_html=True)

        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            amount       = st.number_input("Amount ($)", 0.0, 1_000_000.0, 500.0, 50.0)
            velocity_1h  = st.slider("Transactions (last 1 h)", 0, 50, 2)
            is_night_val = st.toggle("Night transaction (22:00–06:00)", value=False)
        with ci2:
            velocity_24h  = st.slider("Transactions (last 24 h)", 0, 200, 5)
            hour          = st.slider("Hour of day", 0, 23, 14)
            is_round      = st.toggle("Round amount (e.g. $500.00)", value=False)
        with ci3:
            day_of_week   = st.slider("Day of week (0 = Mon)", 0, 6, 1)
            amount_zscore = st.number_input("Amount Z-score", -10.0, 10.0, 0.5, 0.1)
            is_first_pair = st.toggle("First transaction between this pair", value=False)

        st.markdown("</div>", unsafe_allow_html=True)

        features = {
            "amount":           amount,
            "amount_log":       float(np.log1p(amount)),
            "amount_zscore":    amount_zscore,
            "velocity_1h":      float(velocity_1h),
            "velocity_24h":     float(velocity_24h),
            "hour":             float(hour),
            "day_of_week":      float(day_of_week),
            "is_night":         float(is_night_val),
            "is_round_amount":  float(is_round),
            "is_first_pair_tx": float(is_first_pair),
        }
        X    = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
        prob = float(model.predict_proba(X)[0, 1])

        # ── Prediction panel ──────────────────────────────────────────────────
        col_gauge, col_decision, col_info = st.columns([2, 1.5, 1.5])

        with col_gauge:
            gauge_color  = risk_color(prob)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 2),
                number={"suffix": "%", "font": {"color": gauge_color, "size": 42}, "valueformat": ".2f"},
                title={"text": "Fraud Probability", "font": {"color": "#8b949e", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#484f58",
                             "tickwidth": 1, "tickfont": {"color": "#8b949e", "size": 11}},
                    "bar":  {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "borderwidth": 1,
                    "steps": [
                        {"range": [0,  50], "color": "#0d1a0d"},
                        {"range": [50, 70], "color": "#1a1800"},
                        {"range": [70, 90], "color": "#1a0e00"},
                        {"range": [90,100], "color": "#1a0000"},
                    ],
                    "threshold": {
                        "line":      {"color": "#e6edf3", "width": 2},
                        "thickness": 0.8,
                        "value":     threshold * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9",
                height=260,
                margin=dict(t=30, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_decision:
            decision_txt  = "🚨 FLAGGED" if prob >= threshold else "✅ CLEAR"
            decision_clr  = "#ff4444" if prob >= threshold else "#3fb950"
            decision_bg   = "rgba(255,68,68,0.1)" if prob >= threshold else "rgba(63,185,80,0.1)"
            decision_bdr  = "rgba(255,68,68,0.35)" if prob >= threshold else "rgba(63,185,80,0.35)"
            st.markdown(f"""
            <div style="background:{decision_bg};border:1px solid {decision_bdr};
                        border-radius:12px;padding:24px;text-align:center;margin-top:8px">
              <div style="color:{decision_clr};font-size:1.6rem;font-weight:800;margin-bottom:8px">
                {decision_txt}
              </div>
              <div style="color:#8b949e;font-size:0.8rem">
                Threshold: <b style="color:#e6edf3">{threshold:.3f}</b>
              </div>
              <div style="margin-top:10px">{risk_badge_html(prob)}</div>
            </div>""", unsafe_allow_html=True)

        with col_info:
            confidence = abs(prob - threshold) / max(threshold, 1 - threshold)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;
                        padding:20px;margin-top:8px">
              <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:8px">Raw Score</div>
              <div style="color:{risk_color(prob)};font-size:1.8rem;font-weight:800">
                {prob:.4f}
              </div>
              <hr style="border-color:#21262d;margin:12px 0">
              <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:6px">Confidence</div>
              <div style="background:#21262d;border-radius:6px;height:8px;overflow:hidden">
                <div style="background:{risk_color(prob)};width:{min(confidence*100,100):.1f}%;
                            height:100%;border-radius:6px;transition:width 0.3s"></div>
              </div>
              <div style="color:#8b949e;font-size:0.75rem;margin-top:4px">
                {confidence*100:.1f}% from threshold
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Sensitivity analysis ───────────────────────────────────────────────
        st.markdown(section_header(
            "Feature Sensitivity Analysis",
            "How fraud probability responds as one feature varies — all others fixed at current values"
        ), unsafe_allow_html=True)

        RANGES = {
            "amount":           (0.0,   10_000.0),
            "amount_log":       (0.0,   10.0),
            "amount_zscore":    (-5.0,  5.0),
            "velocity_1h":      (0.0,   30.0),
            "velocity_24h":     (0.0,   100.0),
            "hour":             (0.0,   23.0),
            "day_of_week":      (0.0,   6.0),
            "is_night":         (0.0,   1.0),
            "is_round_amount":  (0.0,   1.0),
            "is_first_pair_tx": (0.0,   1.0),
        }

        sens_col, _ = st.columns([1, 2])
        with sens_col:
            sens_feat = st.selectbox(
                "Feature to vary",
                [c for c in feature_cols if c in RANGES],
                format_func=prettify,
            )

        lo, hi      = RANGES.get(sens_feat, (0.0, 100.0))
        feat_range  = np.linspace(lo, hi, 100)
        sens_scores = []
        for val in feat_range:
            f = features.copy()
            f[sens_feat] = float(val)
            if sens_feat == "amount":
                f["amount_log"] = float(np.log1p(val))
            Xs = np.array([[f.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
            sens_scores.append(float(model.predict_proba(Xs)[0, 1]))

        cur_x = features.get(sens_feat, (lo + hi) / 2)

        fig_s = go.Figure()
        # Shaded alert zone above threshold
        fig_s.add_hrect(
            y0=threshold, y1=1.0,
            fillcolor="rgba(231,76,60,0.06)", layer="below", line_width=0,
        )
        # Fill under the curve
        fig_s.add_trace(go.Scatter(
            x=feat_range, y=sens_scores,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(56,139,253,0.08)",
            line=dict(color="#388bfd", width=2.5),
            hovertemplate=f"<b>{prettify(sens_feat)}</b>: %{{x:.3f}}<br>Fraud probability: %{{y:.4f}}<extra></extra>",
        ))
        # Threshold line
        fig_s.add_hline(y=threshold, line_dash="dash", line_color="#666", line_width=1.5,
                         annotation_text=f"Decision threshold ({threshold:.3f})",
                         annotation_position="top right",
                         annotation_font_color="#8b949e", annotation_font_size=11)
        # Current value marker
        cur_y = float(np.interp(cur_x, feat_range, sens_scores))
        fig_s.add_trace(go.Scatter(
            x=[cur_x], y=[cur_y],
            mode="markers+text",
            marker=dict(color=risk_color(cur_y), size=12, line=dict(color="#e6edf3", width=2)),
            text=[f"  Current: {cur_y:.4f}"],
            textposition="top right",
            textfont=dict(color="#e6edf3", size=12),
            hovertemplate=f"Current value<br>{prettify(sens_feat)}: {cur_x:.3f}<br>Probability: {cur_y:.4f}<extra></extra>",
            showlegend=False,
        ))
        fig_s.update_layout(
            **_C,
            height=340,
            margin=dict(t=20, b=50, l=60, r=20),
            xaxis=dict(title=prettify(sens_feat), gridcolor="#21262d",
                       linecolor="#30363d", title_font_size=12),
            yaxis=dict(title="Fraud Probability", range=[-0.02, 1.05],
                       gridcolor="#21262d", linecolor="#30363d",
                       title_font_size=12, showgrid=True, tickformat=".2f"),
            showlegend=False,
        )
        st.plotly_chart(fig_s, use_container_width=True)
