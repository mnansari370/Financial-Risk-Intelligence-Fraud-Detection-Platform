"""
XGBoost Baseline Pipeline
Includes: SMOTE oversampling, scale_pos_weight, threshold calibration.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.evaluation.metrics import calibrate_threshold, compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(features_path: str, cfg: dict) -> tuple:
    df = pd.read_parquet(features_path)
    target = cfg["features"]["target"]
    feature_cols = cfg["features"]["numerical"]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.int32)

    # Temporal split (already sorted by timestamp in build_graph.py)
    n = len(df)
    train_end = int(n * cfg["data"]["train_ratio"])
    val_end   = int(n * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    log.info(f"Train fraud rate: {y_train.mean():.4f}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_smote(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    smote_cfg = cfg["training"]["smote"]
    if not smote_cfg["enabled"]:
        return X, y

    log.info(f"Applying SMOTE (strategy={smote_cfg['sampling_strategy']})...")
    smote = SMOTE(
        sampling_strategy=smote_cfg["sampling_strategy"],
        k_neighbors=smote_cfg["k_neighbors"],
        random_state=smote_cfg["random_state"],
    )
    X_res, y_res = smote.fit_resample(X, y)
    log.info(f"After SMOTE: {len(X_res):,} samples, fraud rate: {y_res.mean():.4f}")
    return X_res, y_res


def train(config_path: str, features_path: str, model_path: str, results_dir: str):
    cfg = load_config(config_path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_features(features_path, cfg)

    # Apply SMOTE on training set only
    X_train_res, y_train_res = apply_smote(X_train, y_train, cfg)

    # Compute scale_pos_weight = neg / pos on ORIGINAL training set
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    log.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    m_cfg = cfg["model"]
    model = XGBClassifier(
        n_estimators=m_cfg["n_estimators"],
        max_depth=m_cfg["max_depth"],
        learning_rate=m_cfg["learning_rate"],
        subsample=m_cfg["subsample"],
        colsample_bytree=m_cfg["colsample_bytree"],
        min_child_weight=m_cfg["min_child_weight"],
        gamma=m_cfg["gamma"],
        reg_alpha=m_cfg["reg_alpha"],
        reg_lambda=m_cfg["reg_lambda"],
        tree_method=m_cfg["tree_method"],
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=cfg["training"]["n_jobs"],
    )

    log.info("Training XGBoost...")
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=cfg["training"]["early_stopping_rounds"],
        verbose=cfg["training"]["verbose_eval"],
    )

    # Threshold calibration on validation set
    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = calibrate_threshold(y_val, val_scores, target_recall=0.80)
    log.info(f"Calibrated threshold for 80% recall: {threshold:.4f}")

    # Evaluate on test set
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold=threshold)
    log.info("\n=== Test Set Metrics ===")
    print_metrics(test_metrics, prefix="  ")

    # Save model + metadata
    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_cols": cfg["features"]["numerical"],
        "test_metrics": test_metrics,
        "config": cfg,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info(f"Model saved → {model_path}")

    # Save metrics
    pd.DataFrame([test_metrics]).to_csv(
        Path(results_dir) / "xgboost_metrics.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xgboost_config.yaml")
    parser.add_argument("--features_path", default="data/processed/features_tabular.parquet")
    parser.add_argument("--model_path", default="src/models/baseline/xgboost_model.pkl")
    parser.add_argument("--results_dir", default="results/xgboost")
    args = parser.parse_args()

    train(args.config, args.features_path, args.model_path, args.results_dir)
