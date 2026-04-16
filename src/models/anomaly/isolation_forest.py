"""
Isolation Forest for unsupervised anomaly detection.
Trained on legitimate transactions only.
Scores are negated (higher = more anomalous) and normalised to [0, 1].
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


def train_isolation_forest(config_path: str, features_path: str, output_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if_cfg = cfg["isolation_forest"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c != "is_fraud"]

    n = len(df)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    # Train on legitimate transactions only
    df_legit = df_train[df_train["is_fraud"] == 0]
    log.info(f"Legitimate training samples: {len(df_legit):,}")
    X_train = df_legit[feature_cols].fillna(0).values

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
    model.fit(X_train)

    # score_samples returns lower values for anomalies; negate so higher = more suspicious
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test["is_fraud"].values
    scores = -model.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    test_metrics = compute_metrics(y_test, scores)
    log.info("\n=== Test Set Metrics (Isolation Forest) ===")
    print_metrics(test_metrics, prefix="  ")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
    }
    model_path = Path(output_dir) / "isolation_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info(f"Saved → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_config.yaml")
    parser.add_argument("--features_path", default="data/processed/features_tabular.parquet")
    parser.add_argument("--output_dir", default="src/models/anomaly")
    args = parser.parse_args()

    train_isolation_forest(args.config, args.features_path, args.output_dir)
