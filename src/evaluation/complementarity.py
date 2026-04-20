"""
Complementarity analysis for fraud models.

Produces:
- average-score ensemble metrics
- overlap at each model's operating threshold
- matched-topk overlap for fair comparison
- unique fraud catches

Usage:
  python -m src.evaluation.complementarity \
      --gat_scores results/evaluation/gat/scores.parquet \
      --xgb_scores results/evaluation/xgboost/scores.parquet \
      --anomaly_scores results/evaluation/anomaly/scores_ensemble.parquet \
      --output_dir results/evaluation/ensemble
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, print_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_score_df(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)[["tx_id", "score", "is_fraud"]].copy()
    return df.rename(columns={"score": model_name})


def overlap_stats(flags: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            both = flags[:, i] & flags[:, j]
            either = flags[:, i] | flags[:, j]
            rows.append({
                "model_a": names[i],
                "model_b": names[j],
                "flagged_by_both": int(both.sum()),
                "fraud_caught_by_both": int((both & labels).sum()),
                "jaccard": float(both.sum()) / float(either.sum() + 1e-8),
            })
    return pd.DataFrame(rows)


def unique_catches(flags: np.ndarray, labels: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        others = np.delete(flags, i, axis=1)
        only_this = flags[:, i] & ~others.any(axis=1)
        rows.append({
            "model": name,
            "unique_fraud_caught": int((only_this & labels).sum()),
            "unique_flags": int(only_this.sum()),
        })
    return pd.DataFrame(rows)


def build_flags_by_topk(score_matrix: np.ndarray, k: int) -> np.ndarray:
    flags = np.zeros_like(score_matrix, dtype=bool)
    for col in range(score_matrix.shape[1]):
        top_idx = np.argsort(score_matrix[:, col])[-k:]
        flags[top_idx, col] = True
    return flags


def main(gat_path: str, xgb_path: str, anomaly_path: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sources = []
    available = []

    for name, path in [("gat", gat_path), ("xgboost", xgb_path), ("anomaly", anomaly_path)]:
        p = Path(path)
        if p.exists():
            df = load_score_df(str(p), name)
            sources.append(df)
            available.append(name)
        else:
            log.warning(f"Scores file not found, skipping: {path}")

    if len(sources) < 2:
        raise RuntimeError("Need at least 2 model score files for complementarity analysis.")

    merged = sources[0]
    for i, df in enumerate(sources[1:], start=1):
        merged = merged.merge(df[["tx_id", available[i]]], on="tx_id", how="inner")

    labels = merged["is_fraud"].values.astype(bool)
    score_matrix = merged[available].values

    # ------------------------------------------------------------------
    # Average ensemble
    # ------------------------------------------------------------------
    ensemble_scores = score_matrix.mean(axis=1)
    ens_metrics = compute_metrics(labels.astype(int), ensemble_scores)
    log.info("\n=== Ensemble (average) Metrics ===")
    print_metrics(ens_metrics)

    pd.DataFrame({
        "tx_id": merged["tx_id"].values,
        "score": ensemble_scores,
        "is_fraud": labels.astype(int),
    }).to_parquet(Path(output_dir) / "scores_ensemble.parquet", index=False)
    pd.DataFrame([ens_metrics]).to_csv(Path(output_dir) / "metrics_ensemble.csv", index=False)

    # ------------------------------------------------------------------
    # Overlap at default operating threshold 0.5
    # ------------------------------------------------------------------
    threshold_flags = score_matrix >= 0.5
    overlap_threshold_df = overlap_stats(threshold_flags, labels, available)
    unique_threshold_df = unique_catches(threshold_flags, labels, available)

    overlap_threshold_df.to_csv(Path(output_dir) / "overlap_threshold.csv", index=False)
    unique_threshold_df.to_csv(Path(output_dir) / "unique_catches_threshold.csv", index=False)

    # ------------------------------------------------------------------
    # Matched-topk overlap: use smallest alert count among models
    # ------------------------------------------------------------------
    alert_counts = [int((score_matrix[:, i] >= 0.5).sum()) for i in range(score_matrix.shape[1])]
    matched_k = max(1, min(alert_counts))
    topk_flags = build_flags_by_topk(score_matrix, matched_k)

    overlap_topk_df = overlap_stats(topk_flags, labels, available)
    unique_topk_df = unique_catches(topk_flags, labels, available)

    overlap_topk_df.to_csv(Path(output_dir) / "overlap_topk.csv", index=False)
    unique_topk_df.to_csv(Path(output_dir) / "unique_catches_topk.csv", index=False)

    # Backward-compatible outputs
    overlap_topk_df.to_csv(Path(output_dir) / "overlap.csv", index=False)
    unique_topk_df.to_csv(Path(output_dir) / "unique_catches.csv", index=False)

    log.info(f"\nMatched-topk overlap stats:\n{overlap_topk_df.to_string(index=False)}")
    log.info(f"\nMatched-topk unique catches:\n{unique_topk_df.to_string(index=False)}")
    log.info(f"Saved ensemble results -> {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat_scores", required=True)
    parser.add_argument("--xgb_scores", required=True)
    parser.add_argument("--anomaly_scores", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.gat_scores, args.xgb_scores, args.anomaly_scores, args.output_dir)