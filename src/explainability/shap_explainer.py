"""
SHAP explainability for XGBoost and GAT models.

XGBoost uses TreeExplainer (exact, fast).
GAT uses KernelExplainer with a model wrapper since GradientExplainer doesn't
support heterogeneous PyG graphs out of the box.
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


def explain_xgboost(checkpoint_path: str, features_path: str,
                    scores_path: str, top_k: int, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = pd.read_parquet(features_path)
    X = df[feature_cols].fillna(0).values

    scores_df = pd.read_parquet(scores_path)
    top_indices = scores_df.nlargest(top_k, "score").index.tolist()
    X_top = X[top_indices]

    log.info(f"Computing SHAP values for top-{top_k} XGBoost predictions...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_top)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["tx_index"] = top_indices
    shap_df.to_parquet(Path(output_dir) / "shap_values_xgb.parquet", index=False)
    log.info(f"SHAP values saved → {output_dir}/shap_values_xgb.parquet")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(Path(output_dir) / "feature_importance_xgb.csv", index=False)
    log.info("Top-5 features by SHAP importance:")
    print(importance_df.head(5).to_string(index=False))


def explain_gat(checkpoint_path: str, graph_path: str,
                scores_path: str, top_k: int, output_dir: str):
    from src.models.gnn.model import FraudGAT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]["model"]

    graphs = torch.load(graph_path, map_location="cpu")
    test_snapshots = graphs["test"]
    sample = test_snapshots[0]

    model = FraudGAT(
        tx_in_channels=sample["transaction"].x.shape[1],
        acc_in_channels=sample["account"].x.shape[1],
        mer_in_channels=sample["merchant"].x.shape[1],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=0.0,    # disable dropout for explanation
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores_df = pd.read_parquet(scores_path)
    top_tx_ids = set(scores_df.nlargest(top_k, "score")["tx_id"].tolist())

    all_shap_values = []
    log.info(f"Computing SHAP values for top-{top_k} GAT predictions...")

    for snapshot in test_snapshots:
        snapshot = snapshot.to(device)
        tx_ids = snapshot["transaction"].tx_id.tolist()
        mask = [i for i, tid in enumerate(tx_ids) if tid in top_tx_ids]
        if not mask:
            continue

        X_tx = snapshot["transaction"].x

        def model_fn(x_tx):
            snapshot["transaction"].x = torch.tensor(x_tx, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(snapshot.x_dict, snapshot.edge_index_dict)
            return torch.sigmoid(logits).cpu().numpy()

        background = X_tx[:min(50, len(X_tx))].detach().cpu().numpy()
        explainer = shap.KernelExplainer(model_fn, background)
        X_explain = X_tx[mask].detach().cpu().numpy()

        shap_vals = explainer.shap_values(X_explain, nsamples=100)
        for i, idx in enumerate(mask):
            all_shap_values.append({
                "tx_id": tx_ids[idx],
                "shap_values": shap_vals[i].tolist(),
            })

    pd.DataFrame(all_shap_values).to_parquet(
        Path(output_dir) / "shap_values_gat.parquet", index=False
    )
    log.info(f"GAT SHAP values saved → {output_dir}/shap_values_gat.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "gat"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features_path")
    parser.add_argument("--graph_path")
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.model == "xgboost":
        explain_xgboost(args.checkpoint, args.features_path,
                        args.scores_path, args.top_k, args.output_dir)
    else:
        explain_gat(args.checkpoint, args.graph_path,
                    args.scores_path, args.top_k, args.output_dir)
