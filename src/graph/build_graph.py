"""
Transaction graph construction pipeline.

Design decisions (aligned to the project report):
  - Dataset:   PaySim (6.3M transactions) — explicit account-to-account transfers
  - Nodes:     bank accounts
  - Edges:     directed transactions (sender → receiver) carrying fraud labels
  - Snapshots: rolling 24-hour windows with 12-hour stride for temporal dynamics
  - Split:     strict temporal 80/10/10 — never random-shuffle financial time-series
  - Features:  Z-score normalised using TRAINING statistics only (no leakage)

The normalisation step is critical for the GAT. Without it, features like
total_out_volume_30d (millions) dominate gradient updates and the model fails
to learn anything meaningful from features like tx_count_24h (single digits).

Outputs:
  data/processed/graph_account_edge.pt   — train/val/test snapshot dicts + norm stats
  data/processed/features_tabular.parquet — tabular features for XGBoost / anomaly models
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Edge feature column order — keep in sync with the model's edge_input_dim.
# Continuous features occupy the first CONT_EDGE_COUNT positions; binary flags follow.
EDGE_FEATURE_COLS = [
    "amount",           # 0 — continuous, normalised
    "amount_log",       # 1 — continuous, normalised
    "hour",             # 2 — continuous, normalised
    "day_of_week",      # 3 — continuous, normalised
    "is_night",         # 4 — binary, kept as-is
    "is_round_amount",  # 5 — binary, kept as-is
    "is_first_pair_tx", # 6 — binary, kept as-is
]
CONT_EDGE_COUNT = 4  # indices 0-3 are normalised; 4-6 are binary flags

NODE_FEATURE_COLS = [
    "total_out_volume_30d",
    "total_in_volume_30d",
    "avg_out_amount_30d",
    "avg_in_amount_30d",
    "unique_out_counterparties_7d",
    "unique_in_counterparties_7d",
    "tx_count_24h",
    "median_tx_hour_30d",
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_paysim(path: str) -> pd.DataFrame:
    """
    Load PaySim and rename columns to a consistent internal schema.

    PaySim fraud only occurs in TRANSFER and CASH_OUT transactions.
    We keep all types for graph connectivity but the label is only meaningful
    for those two types — the model must learn this from the data.
    """
    cols = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    ]
    df = pd.read_csv(path, usecols=cols)

    df = df.rename(columns={
        "step":           "timestamp_step",
        "type":           "tx_type",
        "nameOrig":       "sender_id",
        "nameDest":       "receiver_id",
        "oldbalanceOrg":  "oldbalance_sender",
        "newbalanceOrig": "newbalance_sender",
        "oldbalanceDest": "oldbalance_receiver",
        "newbalanceDest": "newbalance_receiver",
        "isFraud":        "is_fraud",
        "isFlaggedFraud": "is_flagged_fraud",
    })

    # Convert PaySim's integer hour-steps to proper timestamps for temporal splits.
    df["timestamp"] = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(df["timestamp_step"], unit="h")
    )
    df["sender_id"]   = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df["amount"]      = df["amount"].astype(np.float32)
    df["is_fraud"]    = df["is_fraud"].astype(np.int64)

    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive transaction-level (edge) features from raw PaySim columns."""
    df = df.copy()
    dt = pd.to_datetime(df["timestamp"])

    df["hour"]        = dt.dt.hour.astype(np.int16)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int16)
    df["amount_log"]  = np.log1p(df["amount"]).astype(np.float32)
    df["is_night"]    = ((df["hour"] <= 5) | (df["hour"] >= 23)).astype(np.int8)

    # Round-number transactions (exactly divisible by 500 or 1000) are a known AML signal.
    df["is_round_amount"] = (
        np.isclose(df["amount"] % 500, 0.0, atol=1e-4)
        | np.isclose(df["amount"] % 1000, 0.0, atol=1e-4)
    ).astype(np.int8)

    # First-ever transaction between this sender-receiver pair — new counterparties are risky.
    pair_key = df["sender_id"] + "___" + df["receiver_id"]
    df["is_first_pair_tx"] = (~pair_key.duplicated()).astype(np.int8)

    df["sender_balance_drop"]    = (df["oldbalance_sender"]   - df["newbalance_sender"]).astype(np.float32)
    df["receiver_balance_gain"]  = (df["newbalance_receiver"] - df["oldbalance_receiver"]).astype(np.float32)

    return df


def _count_prior_events(df: pd.DataFrame, entity_col: str, hours: int) -> pd.Series:
    """
    For each row, count how many prior transactions the entity made in the
    previous `hours` hours.  Uses a sorted history per entity for O(n log n).
    """
    import bisect

    entity_vals = df[entity_col].values
    ts_ns       = pd.to_datetime(df["timestamp"]).astype(np.int64).values
    window_ns   = int(hours * 3600 * 1e9)
    counts      = np.zeros(len(df), dtype=np.int32)
    hist: Dict[str, List[int]] = {}

    for i in range(len(df)):
        ent = entity_vals[i]
        t   = int(ts_ns[i])
        if ent not in hist:
            hist[ent] = [t]
            counts[i] = 0
            continue
        h    = hist[ent]
        left = bisect.bisect_left(h, t - window_ns)
        counts[i] = len(h) - left
        h.append(t)

    return pd.Series(counts, index=df.index)


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction-level tabular features for XGBoost / anomaly detection.
    Does NOT include graph neighbourhood information — the whole point of the
    ablation is to show what graph structure adds beyond these features.
    """
    out = pd.DataFrame({
        "tx_id":                  np.arange(len(df), dtype=np.int64),
        "amount":                 df["amount"].astype(np.float32),
        "amount_log":             df["amount_log"].astype(np.float32),
        "hour":                   df["hour"].astype(np.int16),
        "day_of_week":            df["day_of_week"].astype(np.int16),
        "is_night":               df["is_night"].astype(np.int8),
        "is_round_amount":        df["is_round_amount"].astype(np.int8),
        "is_first_pair_tx":       df["is_first_pair_tx"].astype(np.int8),
        "sender_balance_drop":    df["sender_balance_drop"].astype(np.float32),
        "receiver_balance_gain":  df["receiver_balance_gain"].astype(np.float32),
        "is_flagged_fraud":       df["is_flagged_fraud"].astype(np.int8),
        "is_fraud":               df["is_fraud"].astype(np.int8),
        "timestamp":              pd.to_datetime(df["timestamp"]),
    })

    out["velocity_1h_sender"]    = _count_prior_events(df, "sender_id",   hours=1)
    out["velocity_24h_sender"]   = _count_prior_events(df, "sender_id",   hours=24)
    out["velocity_1h_receiver"]  = _count_prior_events(df, "receiver_id", hours=1)
    out["velocity_24h_receiver"] = _count_prior_events(df, "receiver_id", hours=24)

    amt_mean = float(out["amount"].mean())
    amt_std  = float(out["amount"].std()) + 1e-8
    out["amount_zscore"] = ((out["amount"] - amt_mean) / amt_std).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# 3. Per-window graph snapshot
# ---------------------------------------------------------------------------

def _build_node_features(
    hist_df: pd.DataFrame, window_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute account node features from transaction history leading up to the
    window, then return a DataFrame and a node-index mapping.
    """
    send = hist_df.groupby("sender_id").agg(
        total_out_volume_30d=("amount", "sum"),
        avg_out_amount_30d=("amount", "mean"),
        unique_out_counterparties_7d=("receiver_id", "nunique"),
    )
    recv = hist_df.groupby("receiver_id").agg(
        total_in_volume_30d=("amount", "sum"),
        avg_in_amount_30d=("amount", "mean"),
        unique_in_counterparties_7d=("sender_id", "nunique"),
    )
    send_count   = hist_df.groupby("sender_id").size().rename("tx_count_24h")
    median_hour  = hist_df.groupby("sender_id")["hour"].median().rename("median_tx_hour_30d")

    all_accs = pd.unique(window_df[["sender_id", "receiver_id"]].values.ravel()).tolist()
    acc_df   = pd.DataFrame({"account_id": all_accs})

    acc_df = acc_df.merge(send,       how="left", left_on="account_id", right_index=True)
    acc_df = acc_df.merge(recv,       how="left", left_on="account_id", right_index=True)
    acc_df = acc_df.merge(send_count, how="left", left_on="account_id", right_index=True)
    acc_df = acc_df.merge(median_hour,how="left", left_on="account_id", right_index=True)

    acc_df[NODE_FEATURE_COLS] = acc_df[NODE_FEATURE_COLS].fillna(0.0)

    acc_map = {aid: i for i, aid in enumerate(acc_df["account_id"].tolist())}
    return acc_df, acc_map


def _build_snapshot(
    full_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Data | None:
    """
    Build one PyG Data object for a 24-hour window.
    Returns None if the window has fewer than 2 transactions (can't form an edge).
    """
    window_mask = (full_df["timestamp"] >= window_start) & (full_df["timestamp"] < window_end)
    window_df   = full_df.loc[window_mask].copy()
    if len(window_df) < 2:
        return None

    # 30-day history for node feature computation (look-back, not look-ahead).
    hist_mask = (
        (full_df["timestamp"] < window_end)
        & (full_df["timestamp"] >= window_end - pd.Timedelta(days=30))
    )
    hist_df  = full_df.loc[hist_mask].copy()
    acc_df, acc_map = _build_node_features(hist_df, window_df)

    src = window_df["sender_id"].map(acc_map).values
    dst = window_df["receiver_id"].map(acc_map).values

    node_feats = acc_df[NODE_FEATURE_COLS].values.astype(np.float32)
    edge_feats = window_df[EDGE_FEATURE_COLS].fillna(0).values.astype(np.float32)

    snap = Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long),
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float32),
        edge_label = torch.tensor(window_df["is_fraud"].values.astype(np.float32)),
        tx_id      = torch.tensor(window_df["tx_id"].values.astype(np.int64)),
    )
    snap.timestamp_start = str(window_start)
    snap.timestamp_end   = str(window_end)
    return snap


def _snapshots_for_split(
    split_df: pd.DataFrame,
    full_df:  pd.DataFrame,
    window_h: int,
    stride_h: int,
) -> List[Data]:
    """Generate rolling snapshots over a time split using the full history for node features."""
    start = pd.to_datetime(split_df["timestamp"].min()).floor("h")
    end   = pd.to_datetime(split_df["timestamp"].max()).ceil("h")
    delta_w = pd.Timedelta(hours=window_h)
    delta_s = pd.Timedelta(hours=stride_h)

    snaps, cursor = [], start
    while cursor < end:
        s = _build_snapshot(full_df, cursor, cursor + delta_w)
        if s is not None:
            snaps.append(s)
        cursor += delta_s

    return snaps


# ---------------------------------------------------------------------------
# 4. Feature normalisation — computed from TRAIN split only
# ---------------------------------------------------------------------------

def normalise_splits(
    train: List[Data],
    val:   List[Data],
    test:  List[Data],
) -> Tuple[List[Data], List[Data], List[Data], dict]:
    """
    Z-score normalise node features and continuous edge features.

    Statistics are computed from the training split ONLY to avoid data leakage
    into validation and test sets.  Binary edge flags (is_night, is_round_amount,
    is_first_pair_tx) are kept at {0, 1} — normalising binary flags loses their
    interpretability and doesn't help the model.

    The normalisation stats are stored in the graph file so inference code can
    apply the same transformation without re-computing from training data.
    """
    # Collect all training node vectors (each snapshot contributes N_nodes rows).
    all_x  = np.concatenate([s.x.numpy() for s in train], axis=0)
    n_mean = all_x.mean(axis=0).astype(np.float32)
    n_std  = (all_x.std(axis=0) + 1e-8).astype(np.float32)

    # Continuous edge features only (indices 0-3: amount, amount_log, hour, day_of_week).
    cont_idx = list(range(CONT_EDGE_COUNT))
    all_ea   = np.concatenate([s.edge_attr[:, cont_idx].numpy() for s in train], axis=0)
    e_mean   = all_ea.mean(axis=0).astype(np.float32)
    e_std    = (all_ea.std(axis=0) + 1e-8).astype(np.float32)

    stats = {
        "node_mean":    n_mean,
        "node_std":     n_std,
        "edge_mean":    e_mean,
        "edge_std":     e_std,
        "cont_edge_idx": cont_idx,
    }

    def _apply(snaps: List[Data]) -> List[Data]:
        nm = torch.from_numpy(n_mean)
        ns = torch.from_numpy(n_std)
        em = torch.from_numpy(e_mean)
        es = torch.from_numpy(e_std)
        out = []
        for s in snaps:
            ea = s.edge_attr.clone()
            ea[:, cont_idx] = (ea[:, cont_idx] - em) / es
            out.append(Data(
                x               = (s.x - nm) / ns,
                edge_index      = s.edge_index,
                edge_attr       = ea,
                edge_label      = s.edge_label,
                tx_id           = s.tx_id,
                timestamp_start = s.timestamp_start,
                timestamp_end   = s.timestamp_end,
            ))
        return out

    return _apply(train), _apply(val), _apply(test), stats


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def build_graphs(
    paysim_path:    str,
    output_dir:     str,
    snapshot_hours: int = 24,
    stride_hours:   int = 12,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading PaySim …")
    df = load_paysim(paysim_path)
    log.info(f"Loaded {len(df):,} transactions | fraud rate: {df['is_fraud'].mean():.4f}")

    df = add_edge_features(df)
    df["tx_id"] = np.arange(len(df), dtype=np.int64)

    # Save tabular features before any graph-specific transforms.
    tabular_df = build_tabular_features(df)
    tab_path   = output_dir / "features_tabular.parquet"
    tabular_df.to_parquet(tab_path, index=False)
    log.info(f"Saved tabular features → {tab_path}")

    # Temporal split — strictly by position after sorting by timestamp.
    n         = len(df)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info(f"  {name}: {len(sdf):,} tx | fraud rate: {sdf['is_fraud'].mean():.4f}")

    # Build raw (un-normalised) snapshots for each split.
    raw = {}
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info(f"Building {name} snapshots ({snapshot_hours}h window / {stride_hours}h stride) …")
        snaps = _snapshots_for_split(sdf, df, snapshot_hours, stride_hours)
        raw[name] = snaps
        log.info(f"  → {len(snaps)} snapshots")

    # Normalise using training statistics.
    log.info("Normalising node and edge features using training statistics …")
    train_norm, val_norm, test_norm, norm_stats = normalise_splits(
        raw["train"], raw["val"], raw["test"]
    )

    graph_data = {
        "train":      train_norm,
        "val":        val_norm,
        "test":       test_norm,
        "norm_stats": norm_stats,
    }

    graph_path = output_dir / "graph_account_edge.pt"
    torch.save(graph_data, graph_path)
    log.info(f"Saved graph → {graph_path}")
    log.info("Graph construction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build account-node / transaction-edge graph snapshots from PaySim"
    )
    parser.add_argument("--paysim_path",     default="data/raw/paysim.csv")
    parser.add_argument("--output_dir",      default="data/processed")
    parser.add_argument("--snapshot_hours",  type=int, default=24)
    parser.add_argument("--stride_hours",    type=int, default=12)
    args = parser.parse_args()

    build_graphs(
        paysim_path=args.paysim_path,
        output_dir=args.output_dir,
        snapshot_hours=args.snapshot_hours,
        stride_hours=args.stride_hours,
    )
