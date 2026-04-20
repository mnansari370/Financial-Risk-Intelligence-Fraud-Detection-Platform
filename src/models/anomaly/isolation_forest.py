"""
Isolation Forest for unsupervised anomaly detection.
Trained on legitimate transactions only.

Corrected version:
- uses explicit feature columns from config
- standardizes inputs using training-legitimate statistics
- stores normalization stats in the saved artifact
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

from src.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def standardize(train_x: np.ndarray, other_x: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (other_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_isolation_forest(config_path: str, features_path: str, output_dir: str):
    with open(config_path) as f:
        cfg_all = yaml.safe_load(f)

    if_cfg = cfg_all["isolation_forest"]
    feature_cols = cfg_all["data"]["feature_columns"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in feature_cols + ["is_fraud"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    df_train = df.iloc[:train_end]
    df_test = df.iloc[val_end:]

    df_legit = df_train[df_train["is_fraud"] == 0]
    log.info(f"Legitimate training samples: {len(df_legit):,}")

    X_train = df_legit[feature_cols].fillna(0).values.astype(np.float32)
    X_test = df_test[feature_cols].fillna(0).values.astype(np.float32)
    y_test = df_test["is_fraud"].values

    X_train_std, X_test_std, mean_vec, std_vec = standardize(X_train, X_test)

    model = IsolationForest(
        n_estimators=if_cfg["n_estimators"],
        max_samples=if_cfg["max_samples"],
        contamination=if_cfg["contamination"],
        max_features=if_cfg["max_features"],
        bootstrap=if_cfg["bootstrap"],
        n_jobs=if_cfg["n_jobs"],
        random_state=if_cfg["random_state"],
    )

    log.info("Training Isolation Forest...")
    model.fit(X_train_std)

    scores = -model.score_samples(X_test_std)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    test_metrics = compute_metrics(y_test, scores)
    log.info("\n=== Test Set Metrics (Isolation Forest) ===")
    print_metrics(test_metrics, prefix="  ")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "feature_mean": mean_vec.astype(np.float32),
        "feature_std": std_vec.astype(np.float32),
        "test_metrics": test_metrics,
    }

    model_path = Path(output_dir) / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    log.info(f"Saved -> {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_config.yaml")
    parser.add_argument("--features_path", default="data/processed/features_tabular.parquet")
    parser.add_argument("--output_dir", default="src/models/anomaly")
    args = parser.parse_args()

    train_isolation_forest(args.config, args.features_path, args.output_dir)