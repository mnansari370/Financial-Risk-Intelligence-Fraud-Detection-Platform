"""
Financial Risk Intelligence & Fraud Detection Dashboard
Streamlit 5-panel interface:
  1. Transaction Queue (live alert feed)
  2. Transaction Detail (per-alert view)
  3. Graph Visualisation (PyVis neighbourhood)
  4. SAR Generation Panel (LLM output)
  5. Scenario Simulator (what-if analysis)
"""

import json
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from pyvis.network import Network

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Fraud Detection Platform")
st.sidebar.markdown("*MSc AI — University of Luxembourg*")

MODEL_PATHS = {
    "gat_checkpoint": "src/models/gnn/checkpoints/best_model.pt",
    "xgb_checkpoint": "src/models/baseline/xgboost_model.pkl",
    "graph_path": "data/processed/graph.pt",
    "gat_scores": "results/evaluation/gat/scores.parquet",
    "xgb_scores": "results/evaluation/xgboost/scores.parquet",
}

score_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.5, 0.01)
llm_provider = st.sidebar.selectbox("LLM Provider", ["openai", "local"])
top_k_display = st.sidebar.number_input("Top-K Alerts to Show", 10, 500, 50)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_scores():
    scores_frames = {}
    for key in ["gat_scores", "xgb_scores"]:
        path = MODEL_PATHS[key]
        if Path(path).exists():
            scores_frames[key] = pd.read_parquet(path)
    return scores_frames


@st.cache_resource
def load_xgb_model():
    path = MODEL_PATHS["xgb_checkpoint"]
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_graph():
    path = MODEL_PATHS["graph_path"]
    if not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu")


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Transaction Queue",
    "🔎 Alert Detail",
    "🕸 Graph View",
    "📄 SAR Generator",
    "⚙ Scenario Simulator",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Transaction Queue
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Transaction Alert Queue")
    scores = load_scores()

    if not scores:
        st.warning("No scores found. Run the evaluation pipeline first: `sbatch slurm/full_eval.sh`")
    else:
        gat_df = scores.get("gat_scores", pd.DataFrame())
        xgb_df = scores.get("xgb_scores", pd.DataFrame())

        if not gat_df.empty:
            alerts = gat_df[gat_df["score"] >= score_threshold].nlargest(
                int(top_k_display), "score"
            )
            st.metric("Alerts above threshold", len(alerts))

            # Colour code by score
            def score_colour(val):
                if val >= 0.9:
                    return "background-color: #ff4444; color: white"
                elif val >= 0.7:
                    return "background-color: #ff8800; color: white"
                elif val >= 0.5:
                    return "background-color: #ffcc00"
                return ""

            display_cols = ["tx_id", "score", "amount", "sender_id",
                            "receiver_id", "timestamp"]
            display_cols = [c for c in display_cols if c in alerts.columns]
            styled = alerts[display_cols].style.applymap(
                score_colour, subset=["score"]
            ).format({"score": "{:.4f}", "amount": "${:,.2f}"})

            st.dataframe(styled, use_container_width=True)

            # Score distribution
            fig = px.histogram(gat_df, x="score", nbins=50,
                               title="GAT Fraud Score Distribution",
                               color_discrete_sequence=["#e74c3c"])
            fig.add_vline(x=score_threshold, line_dash="dash",
                          annotation_text=f"Threshold={score_threshold}")
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Alert Detail
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Alert Detail View")
    scores = load_scores()

    if scores:
        gat_df = scores.get("gat_scores", pd.DataFrame())
        if not gat_df.empty:
            tx_options = gat_df[gat_df["score"] >= score_threshold]["tx_id"].tolist()
            if tx_options:
                selected_tx = st.selectbox("Select Transaction", tx_options[:100])
                row = gat_df[gat_df["tx_id"] == selected_tx].iloc[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("GAT Score", f"{row.get('score', 0):.4f}")

                xgb_df = scores.get("xgb_scores", pd.DataFrame())
                if not xgb_df.empty and selected_tx in xgb_df["tx_id"].values:
                    xgb_row = xgb_df[xgb_df["tx_id"] == selected_tx].iloc[0]
                    col2.metric("XGBoost Score", f"{xgb_row.get('score', 0):.4f}")

                col3.metric("Amount", f"${row.get('amount', 0):,.2f}")

                st.subheader("Transaction Details")
                detail_fields = {k: v for k, v in row.items()
                                 if k not in ["score"]}
                st.json(detail_fields)

                # SHAP waterfall (if available)
                shap_path = f"results/shap/xgboost/shap_values_xgb.parquet"
                if Path(shap_path).exists():
                    shap_df = pd.read_parquet(shap_path)
                    if selected_tx in shap_df.get("tx_index", pd.Series()).values:
                        st.subheader("SHAP Feature Contributions")
                        tx_shap = shap_df[shap_df["tx_index"] == selected_tx].iloc[0]
                        feature_cols = [c for c in shap_df.columns
                                        if c not in ["tx_index"]]
                        shap_vals = {c: float(tx_shap[c]) for c in feature_cols}
                        shap_sorted = sorted(shap_vals.items(),
                                             key=lambda x: abs(x[1]), reverse=True)[:10]
                        features, values = zip(*shap_sorted)
                        fig = px.bar(
                            x=list(values), y=list(features),
                            orientation="h",
                            color=list(values),
                            color_continuous_scale="RdBu",
                            title="Top-10 SHAP Feature Contributions",
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Graph Visualisation
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Transaction Graph Neighbourhood")

    scores = load_scores()
    if scores:
        gat_df = scores.get("gat_scores", pd.DataFrame())
        if not gat_df.empty:
            tx_options = gat_df.nlargest(100, "score")["tx_id"].tolist()
            selected_tx_graph = st.selectbox("Select Transaction (Graph View)",
                                              tx_options, key="graph_tx")
            hop = st.slider("Neighbourhood hops", 1, 3, 2)

            # Build a small ego graph for visualisation
            G = nx.DiGraph()
            G.add_node(str(selected_tx_graph), node_type="transaction",
                       color="#e74c3c", size=25, title=f"TX: {selected_tx_graph}")

            # Add dummy neighbours (real version uses actual graph edges)
            row = gat_df[gat_df["tx_id"] == selected_tx_graph].iloc[0]
            sender = str(row.get("sender_id", "ACC-???"))
            receiver = str(row.get("receiver_id", "ACC-???"))
            G.add_node(sender, node_type="account", color="#3498db", size=15,
                       title=f"Sender: {sender}")
            G.add_node(receiver, node_type="account", color="#2ecc71", size=15,
                       title=f"Receiver: {receiver}")
            G.add_edge(str(selected_tx_graph), sender, label="sent_by")
            G.add_edge(str(selected_tx_graph), receiver, label="received_by")

            # Render with PyVis
            net = Network(height="500px", width="100%", directed=True,
                          bgcolor="#1a1a2e", font_color="white")
            net.from_nx(G)
            net.set_options(json.dumps({
                "physics": {"barnesHut": {"gravitationalConstant": -8000}},
                "edges": {"arrows": {"to": {"enabled": True}}},
            }))

            html_path = "/tmp/fraud_graph.html"
            net.save_graph(html_path)
            with open(html_path) as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=520, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SAR Generator
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("SAR Report Generator")
    st.markdown("*Generates EU-compliant Suspicious Activity Reports using LLM*")

    scores = load_scores()
    if scores:
        gat_df = scores.get("gat_scores", pd.DataFrame())
        if not gat_df.empty:
            tx_options = gat_df.nlargest(50, "score")["tx_id"].tolist()
            selected_tx_sar = st.selectbox("Select Alert", tx_options, key="sar_tx")
            row = gat_df[gat_df["tx_id"] == selected_tx_sar].iloc[0]

            if st.button("Generate SAR Report", type="primary"):
                from src.llm.sar_generator import SARGenerator

                with st.spinner("Generating SAR using LLM..."):
                    try:
                        gen = SARGenerator(
                            provider=llm_provider,
                            model="gpt-4o-mini" if llm_provider == "openai" else None,
                        )
                        transaction_data = row.to_dict()
                        fraud_scores = {"gat": float(row.get("score", 0))}

                        sar = gen.generate(transaction_data, fraud_scores)

                        # Display results
                        risk_colours = {
                            "CRITICAL": "🔴", "HIGH": "🟠",
                            "MEDIUM": "🟡", "LOW": "🟢"
                        }
                        risk = sar.get("risk_level", "UNKNOWN")
                        st.markdown(f"### {risk_colours.get(risk, '⚪')} Risk Level: {risk}")

                        st.subheader("SAR Narrative")
                        st.markdown(sar.get("narrative", "No narrative generated."))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Regulatory Flags")
                            for flag in sar.get("regulatory_flags", []):
                                st.markdown(f"- {flag}")
                        with col2:
                            st.subheader("Recommended Action")
                            st.info(sar.get("recommended_action", "REVIEW"))

                        st.subheader("Full SAR JSON")
                        st.json(sar)

                    except Exception as e:
                        st.error(f"SAR generation failed: {e}")
                        st.info("Make sure OPENAI_API_KEY is set in your .env file, "
                                "or switch to 'local' provider.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Scenario Simulator
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("What-If Scenario Simulator")
    st.markdown("Modify transaction features and see how the model score changes.")

    model_artifact = load_xgb_model()
    if model_artifact is None:
        st.warning("XGBoost model not found. Train it first: `sbatch slurm/train_gat.sh`")
    else:
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        threshold = model_artifact.get("threshold", 0.5)

        st.subheader("Adjust Transaction Features")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount ($)", 0.0, 100000.0, 500.0, 10.0)
            velocity_1h = st.number_input("Transactions in last 1h", 0, 100, 2)
            hour = st.slider("Hour of day", 0, 23, 14)
        with col2:
            velocity_24h = st.number_input("Transactions in last 24h", 0, 200, 5)
            day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 1)
            amount_zscore = st.number_input("Amount Z-score", -10.0, 10.0, 0.5, 0.1)

        import numpy as np
        features = {
            "amount": amount,
            "amount_log": float(np.log1p(amount)),
            "amount_zscore": amount_zscore,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "hour": hour,
            "day_of_week": day_of_week,
        }
        X = np.array([[features.get(c, 0) for c in feature_cols]], dtype=np.float32)
        score = model.predict_proba(X)[0, 1]

        st.metric("Predicted Fraud Score", f"{score:.4f}")
        st.metric("Decision", "🚨 FRAUD" if score >= threshold else "✅ LEGIT",
                  delta=f"threshold={threshold:.3f}")

        # Gauge chart
        fig = px.bar(x=["Fraud Score"], y=[score],
                     color=[score], color_continuous_scale="RdYlGn_r",
                     range_y=[0, 1],
                     title=f"Fraud Probability: {score:.4f}")
        fig.add_hline(y=threshold, line_dash="dash",
                      annotation_text=f"Threshold ({threshold:.2f})")
        st.plotly_chart(fig, use_container_width=True)
