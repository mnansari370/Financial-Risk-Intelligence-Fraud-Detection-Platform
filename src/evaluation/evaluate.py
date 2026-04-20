"""
Evaluation script — generates scores.parquet for each model on the test set.

Usage:
  python -m src.evaluation.evaluate --model gat --checkpoint ... --graph_path ...
  python -m src.evaluation.evaluate --model xgboost --checkpoint ... --features_path ...
  python -m src.evaluation.evaluate --model anomaly --if_checkpoint ... --ae_checkpoint ... --features_path ...
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate_gat(checkpoint_path, graph_path, output_dir):
    from src.models.gnn.model import FraudEdgeGAT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]["model"]

    graphs = torch.load(graph_path, map_location="cpu")
    test_snapshots = graphs["test"]
    sample = test_snapshots[0]

    model = FraudEdgeGAT(
        node_input_dim=sample.x.shape[1],
        edge_input_dim=sample.edge_attr.shape[1],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=0.0,
        residual=cfg["residual"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores, all_labels, all_tx_ids = [], [], []

    with torch.no_grad():
        for snapshot in test_snapshots:
            snapshot = snapshot.to(device)
            logits = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            scores = torch.sigmoid(logits).cpu().numpy()
            labels = snapshot.edge_label.cpu().numpy()
            tx_ids = snapshot.tx_id.cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels)
            all_tx_ids.append(tx_ids)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    tx_ids = np.concatenate(all_tx_ids)

    metrics = compute_metrics(labels, scores)
    log.info("\n=== GAT Test Metrics ===")
    print_metrics(metrics)

    df = pd.DataFrame({"tx_id": tx_ids, "score": scores, "is_fraud": labels.astype(int)})
    df.to_parquet(Path(output_dir) / "scores.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(Path(output_dir) / "metrics.csv", index=False)
    log.info(f"Saved scores -> {output_dir}/scores.parquet")


def evaluate_xgboost(checkpoint_path, features_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("threshold", 0.5)

    df = pd.read_parquet(features_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    test_start = int(n * 0.90)
    df_test = df.iloc[test_start:]

    X_test = df_test[feature_cols].fillna(0).values.astype(np.float32)
    y_test = df_test["is_fraud"].values
    tx_ids = df_test["tx_id"].values if "tx_id" in df_test.columns else np.arange(test_start, n)

    scores = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, scores, threshold=threshold)
    log.info("\n=== XGBoost Test Metrics ===")
    print_metrics(metrics)

    df_out = pd.DataFrame({
        "tx_id": tx_ids,
        "score": scores,
        "is_fraud": y_test,
    })
    df_out.to_parquet(Path(output_dir) / "scores.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(Path(output_dir) / "metrics.csv", index=False)
    log.info(f"Saved scores -> {output_dir}/scores.parquet")


def evaluate_anomaly(if_checkpoint, ae_checkpoint, features_path, output_dir):
    from src.models.anomaly.autoencoder import Autoencoder

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    test_start = int(n * 0.90)
    df_test = df.iloc[test_start:]
    y_test = df_test["is_fraud"].values
    tx_ids = df_test["tx_id"].values if "tx_id" in df_test.columns else np.arange(test_start, n)

    # -------------------------------
    # Isolation Forest
    # -------------------------------
    with open(if_checkpoint, "rb") as f:
        if_artifact = pickle.load(f)

    if_model = if_artifact["model"]
    if_feature_cols = if_artifact["feature_cols"]
    if_mean = np.asarray(if_artifact["feature_mean"], dtype=np.float32)
    if_std = np.asarray(if_artifact["feature_std"], dtype=np.float32)

    X_test_if = df_test[if_feature_cols].fillna(0).values.astype(np.float32)
    X_test_if = (X_test_if - if_mean) / (if_std + 1e-8)

    if_scores = -if_model.score_samples(X_test_if)
    if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)

    # -------------------------------
    # Autoencoder
    # -------------------------------
    ae_ckpt = torch.load(ae_checkpoint, map_location="cpu")
    ae_cfg = ae_ckpt["config"]
    ae_feature_cols = ae_ckpt["feature_cols"]
    ae_mean = np.asarray(ae_ckpt["feature_mean"], dtype=np.float32)
    ae_std = np.asarray(ae_ckpt["feature_std"], dtype=np.float32)

    X_test_ae = df_test[ae_feature_cols].fillna(0).values.astype(np.float32)
    X_test_ae = (X_test_ae - ae_mean) / (ae_std + 1e-8)

    ae_model = Autoencoder(
        input_dim=ae_ckpt["input_dim"],
        encoder_dims=ae_cfg["encoder_dims"],
        latent_dim=ae_cfg["latent_dim"],
        decoder_dims=ae_cfg["decoder_dims"],
        dropout=0.0,
    )
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model.eval()

    with torch.no_grad():
        ae_errors = ae_model.reconstruction_error(torch.tensor(X_test_ae, dtype=torch.float32)).numpy()
    ae_scores = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-8)

    # -------------------------------
    # Ensemble
    # -------------------------------
    ensemble_scores = (if_scores + ae_scores) / 2.0

    for name, scores in [
        ("isolation_forest", if_scores),
        ("autoencoder", ae_scores),
        ("ensemble", ensemble_scores),
    ]:
        metrics = compute_metrics(y_test, scores)
        log.info(f"\n=== {name} Test Metrics ===")
        print_metrics(metrics)

        df_out = pd.DataFrame({
            "tx_id": tx_ids,
            "score": scores,
            "is_fraud": y_test,
        })
        df_out.to_parquet(Path(output_dir) / f"scores_{name}.parquet", index=False)
        pd.DataFrame([metrics]).to_csv(Path(output_dir) / f"metrics_{name}.csv", index=False)

    log.info(f"Saved anomaly scores -> {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gat", "xgboost", "anomaly"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--graph_path")
    parser.add_argument("--features_path")
    parser.add_argument("--if_checkpoint")
    parser.add_argument("--ae_checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.model == "gat":
        evaluate_gat(args.checkpoint, args.graph_path, args.output_dir)
    elif args.model == "xgboost":
        evaluate_xgboost(args.checkpoint, args.features_path, args.output_dir)
    elif args.model == "anomaly":
        evaluate_anomaly(args.if_checkpoint, args.ae_checkpoint, args.features_path, args.output_dir)