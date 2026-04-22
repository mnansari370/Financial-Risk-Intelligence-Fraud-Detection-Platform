"""
XGBoost baseline pipeline.

Aligned to the corrected tabular features:
- transaction-level features only
- strict temporal split
- SMOTE on training split only
- threshold calibration on validation split
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.evaluation.metrics import calibrate_threshold, compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(features_path: str, cfg: dict):
    df = pd.read_parquet(features_path)

    target_col = cfg["features"]["target"]
    feature_cols = cfg["features"]["numerical"]

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")

    # Ensure temporal order using timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)
    tx_ids = df["tx_id"].values if "tx_id" in df.columns else np.arange(len(df))

    n = len(df)
    train_end = int(n * cfg["data"]["train_ratio"])
    val_end = int(n * (cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"]))

    X_train, y_train, tx_train = X[:train_end], y[:train_end], tx_ids[:train_end]
    X_val, y_val, tx_val = X[train_end:val_end], y[train_end:val_end], tx_ids[train_end:val_end]
    X_test, y_test, tx_test = X[val_end:], y[val_end:], tx_ids[val_end:]

    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    log.info(f"Train fraud rate: {y_train.mean():.4f}")
    log.info(f"Val fraud rate:   {y_val.mean():.4f}")
    log.info(f"Test fraud rate:  {y_test.mean():.4f}")

    return (
        X_train, y_train, tx_train,
        X_val, y_val, tx_val,
        X_test, y_test, tx_test,
        feature_cols,
    )


def apply_smote(X: np.ndarray, y: np.ndarray, cfg: dict):
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

    (
        X_train, y_train, tx_train,
        X_val, y_val, tx_val,
        X_test, y_test, tx_test,
        feature_cols,
    ) = load_features(features_path, cfg)

    X_train_res, y_train_res = apply_smote(X_train, y_train, cfg)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
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
        eval_metric=m_cfg["eval_metric"],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=cfg["training"]["n_jobs"],
    )

    log.info("Training XGBoost...")
    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=cfg["training"]["verbose_eval"],
    )

    val_scores = model.predict_proba(X_val)[:, 1]
    if cfg["training"]["threshold_calibration"]["enabled"]:
        threshold = calibrate_threshold(
            y_val,
            val_scores,
            target_recall=cfg["training"]["threshold_calibration"]["target_recall"],
        )
    else:
        threshold = 0.5

    log.info(f"Calibrated threshold: {threshold:.4f}")

    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_scores, threshold=threshold)

    log.info("\n=== Test Set Metrics ===")
    print_metrics(test_metrics, prefix="  ")

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "test_metrics": test_metrics,
        "config": cfg,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    log.info(f"Model saved -> {model_path}")

    pd.DataFrame([test_metrics]).to_csv(Path(results_dir) / "xgboost_metrics.csv", index=False)

    # Optional convenience output
    df_scores = pd.DataFrame({
        "tx_id": tx_test,
        "score": test_scores,
        "is_fraud": y_test,
    })
    df_scores.to_parquet(Path(results_dir) / "scores_test_xgboost.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xgboost_config.yaml")
    parser.add_argument("--features_path", default="data/processed/features_tabular.parquet")
    parser.add_argument("--model_path", default="src/models/baseline/xgboost_model.pkl")
    parser.add_argument("--results_dir", default="results/xgboost")
    args = parser.parse_args()

    train(args.config, args.features_path, args.model_path, args.results_dir)
