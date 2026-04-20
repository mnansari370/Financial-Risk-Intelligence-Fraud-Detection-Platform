"""
Explainability for XGBoost and FraudEdgeGAT.

XGBoost  — true SHAP TreeExplainer (exact Shapley values, fast).
GAT      — gradient × input saliency over edge attributes.

Note on GAT explainability: SHAP's GradientExplainer requires a fixed-size
tensor input, but our GAT operates on variable-size sparse graphs.  Gradient ×
input saliency (integrated-gradients lite) is the standard approach for GNNs:
it attributes each edge feature's contribution to the final fraud probability
by measuring how much a small change in that feature shifts the score.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Canonical edge feature names produced by build_graph.py
EDGE_FEATURE_NAMES = [
    "amount",
    "amount_log",
    "hour",
    "day_of_week",
    "is_night",
    "is_round_amount",
    "is_first_pair_tx",
]


def explain_xgboost(
    checkpoint_path: str,
    features_path: str,
    scores_path: str,
    top_k: int,
    output_dir: str,
) -> None:
    """
    Compute SHAP values for the top-K highest-scored transactions using
    TreeExplainer, then save a per-transaction SHAP table and a global
    feature-importance CSV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "rb") as f:
        artifact = pickle.load(f)

    model        = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = pd.read_parquet(features_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if "tx_id" not in df.columns:
        raise ValueError("features_tabular.parquet must contain 'tx_id'.")

    scores_df  = pd.read_parquet(scores_path)
    top_tx_ids = scores_df.nlargest(top_k, "score")[["tx_id"]].copy()

    df_top = (
        df.merge(top_tx_ids, on="tx_id", how="inner")
          .set_index("tx_id")
          .loc[top_tx_ids["tx_id"]]
          .reset_index()
    )
    X_top = df_top[feature_cols].fillna(0).values.astype(np.float32)

    log.info(f"Computing SHAP values for top-{top_k} XGBoost predictions ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_top)

    shap_df          = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["tx_id"] = df_top["tx_id"].values
    shap_df.to_parquet(Path(output_dir) / "shap_values_xgb.parquet", index=False)
    log.info(f"SHAP values saved → {output_dir}/shap_values_xgb.parquet")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance_df.to_csv(Path(output_dir) / "feature_importance_xgb.csv", index=False)

    log.info("Top-5 features by SHAP importance:")
    print(importance_df.head(5).to_string(index=False))


def explain_gat(
    checkpoint_path: str,
    graph_path: str,
    scores_path: str,
    top_k: int,
    output_dir: str,
) -> None:
    """
    Compute gradient × input saliency over edge attributes for the top-K
    highest-scored transactions.

    For each target transaction edge, we compute the gradient of the fraud
    probability with respect to that edge's feature vector, then multiply by
    the feature values to get a signed attribution score.  Positive values
    push toward fraud; negative values push toward legitimate.
    """
    from src.models.gnn.model import FraudEdgeGAT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]["model"]

    graphs         = torch.load(graph_path, map_location="cpu")
    test_snapshots = graphs["test"]
    sample         = test_snapshots[0]

    model = FraudEdgeGAT(
        node_input_dim=sample.x.shape[1],
        edge_input_dim=sample.edge_attr.shape[1],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=0.0,          # disable dropout at inference time
        residual=cfg["residual"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores_df  = pd.read_parquet(scores_path)
    top_tx_ids = set(scores_df.nlargest(top_k, "score")["tx_id"].tolist())

    records = []
    log.info(f"Computing gradient saliency for top-{top_k} GAT predictions ...")

    for snapshot in test_snapshots:
        snapshot = snapshot.to(device)
        tx_ids   = snapshot.tx_id.detach().cpu().tolist()

        target_positions = [i for i, tid in enumerate(tx_ids) if tid in top_tx_ids]
        if not target_positions:
            continue

        # Enable gradients on edge_attr so we can attribute by feature
        edge_attr = snapshot.edge_attr.detach().clone().requires_grad_(True)
        logits    = model(snapshot.x, snapshot.edge_index, edge_attr)
        probs     = torch.sigmoid(logits)

        model.zero_grad(set_to_none=True)
        probs[target_positions].sum().backward()

        grads    = edge_attr.grad[target_positions].detach().cpu().numpy()
        feats    = edge_attr.detach()[target_positions].cpu().numpy()
        saliency = grads * feats   # gradient × input attribution

        for local_i, pos in enumerate(target_positions):
            records.append({
                "tx_id":    int(tx_ids[pos]),
                "saliency": saliency[local_i].tolist(),
                "gradient": grads[local_i].tolist(),
            })

    if not records:
        log.warning("No target transactions found in test snapshots.")
        return

    df_out = pd.DataFrame(records)
    df_out.to_parquet(Path(output_dir) / "shap_values_gat.parquet", index=False)
    log.info(f"GAT saliency saved → {output_dir}/shap_values_gat.parquet")

    saliency_matrix = np.stack(df_out["saliency"].values)
    mean_abs_sal    = np.abs(saliency_matrix).mean(axis=0)

    feat_names = EDGE_FEATURE_NAMES[:saliency_matrix.shape[1]]
    (
        pd.DataFrame({"feature": feat_names, "mean_abs_saliency": mean_abs_sal})
        .sort_values("mean_abs_saliency", ascending=False)
        .to_csv(Path(output_dir) / "feature_importance_gat.csv", index=False)
    )
    log.info(f"GAT feature importance saved → {output_dir}/feature_importance_gat.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         choices=["xgboost", "gat"], required=True)
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--features_path")
    parser.add_argument("--graph_path")
    parser.add_argument("--scores_path",   required=True)
    parser.add_argument("--top_k",         type=int, default=1000)
    parser.add_argument("--output_dir",    required=True)
    args = parser.parse_args()

    if args.model == "xgboost":
        explain_xgboost(
            args.checkpoint, args.features_path,
            args.scores_path, args.top_k, args.output_dir,
        )
    else:
        explain_gat(
            args.checkpoint, args.graph_path,
            args.scores_path, args.top_k, args.output_dir,
        )
