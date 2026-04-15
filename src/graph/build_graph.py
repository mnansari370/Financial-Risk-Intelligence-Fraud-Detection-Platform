"""
Graph Construction Pipeline
Builds PyTorch Geometric HeteroData objects from raw transaction logs.

Nodes:
  - transaction: one node per transaction
  - account: one node per unique account (sender/receiver)
  - merchant: one node per unique merchant

Edges:
  - transaction → account (sender)
  - transaction → account (receiver)
  - transaction → merchant
  - account → account (shared transactions within 24h window)

Temporal snapshots: one PyG graph per 24h window.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Feature engineering helpers ──────────────────────────────────────────────

def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[ts_col])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["amount_log"] = np.log1p(df["amount"])
    return df


def compute_velocity(df: pd.DataFrame, window_hours: int = 1) -> pd.Series:
    """Number of transactions from the same sender in the last `window_hours`."""
    df = df.sort_values("timestamp")
    window_ns = window_hours * 3600 * 1_000_000_000
    df["ts_ns"] = pd.to_datetime(df["timestamp"]).astype(np.int64)

    velocities = []
    for _, row in df.iterrows():
        mask = (
            (df["sender_id"] == row["sender_id"]) &
            (df["ts_ns"] >= row["ts_ns"] - window_ns) &
            (df["ts_ns"] < row["ts_ns"])
        )
        velocities.append(mask.sum())
    return pd.Series(velocities, index=df.index)


def build_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-account statistics from transaction history."""
    sender_stats = df.groupby("sender_id").agg(
        avg_tx_amount=("amount", "mean"),
        tx_count_30d=("amount", "count"),
        unique_merchants=("merchant_id", "nunique"),
    ).reset_index().rename(columns={"sender_id": "account_id"})
    return sender_stats


def build_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-merchant statistics."""
    stats = df.groupby("merchant_id").agg(
        avg_tx_amount=("amount", "mean"),
        tx_volume=("amount", "count"),
        fraud_rate_30d=("is_fraud", "mean"),
    ).reset_index()
    return stats


# ── Graph snapshot builder ────────────────────────────────────────────────────

def build_snapshot(df_window: pd.DataFrame,
                   account_features: pd.DataFrame,
                   merchant_features: pd.DataFrame,
                   tx_feature_cols: list[str]) -> HeteroData:
    """Build a single HeteroData snapshot from one time window."""

    data = HeteroData()

    # ── Index mappings ────────────────────────────────────────────────────────
    tx_ids = df_window.index.tolist()
    tx_map = {tid: i for i, tid in enumerate(tx_ids)}

    all_accounts = pd.unique(
        df_window[["sender_id", "receiver_id"]].values.ravel()
    ).tolist()
    acc_map = {aid: i for i, aid in enumerate(all_accounts)}

    all_merchants = df_window["merchant_id"].unique().tolist()
    mer_map = {mid: i for i, mid in enumerate(all_merchants)}

    # ── Node features ─────────────────────────────────────────────────────────
    # Transactions
    tx_feats = df_window[tx_feature_cols].fillna(0).values.astype(np.float32)
    data["transaction"].x = torch.tensor(tx_feats)
    data["transaction"].y = torch.tensor(
        df_window["is_fraud"].values.astype(np.int64)
    )
    data["transaction"].tx_id = torch.tensor(tx_ids)

    # Accounts
    acc_df = pd.DataFrame({"account_id": all_accounts})
    acc_df = acc_df.merge(account_features, on="account_id", how="left").fillna(0)
    acc_feat_cols = ["avg_tx_amount", "tx_count_30d", "unique_merchants"]
    acc_feats = acc_df[acc_feat_cols].values.astype(np.float32)
    data["account"].x = torch.tensor(acc_feats)

    # Merchants
    mer_df = pd.DataFrame({"merchant_id": all_merchants})
    mer_df = mer_df.merge(merchant_features, on="merchant_id", how="left").fillna(0)
    mer_feat_cols = ["avg_tx_amount", "tx_volume", "fraud_rate_30d"]
    mer_feats = mer_df[mer_feat_cols].values.astype(np.float32)
    data["merchant"].x = torch.tensor(mer_feats)

    # ── Edge indices ──────────────────────────────────────────────────────────
    n_tx = len(tx_ids)

    # transaction → sender account
    src_tx = list(range(n_tx))
    dst_acc_sender = [acc_map[df_window.loc[t, "sender_id"]] for t in tx_ids]
    data["transaction", "sent_by", "account"].edge_index = torch.tensor(
        [src_tx, dst_acc_sender], dtype=torch.long
    )

    # transaction → receiver account
    dst_acc_receiver = [acc_map[df_window.loc[t, "receiver_id"]] for t in tx_ids]
    data["transaction", "received_by", "account"].edge_index = torch.tensor(
        [src_tx, dst_acc_receiver], dtype=torch.long
    )

    # transaction → merchant
    dst_mer = [mer_map[df_window.loc[t, "merchant_id"]] for t in tx_ids]
    data["transaction", "at", "merchant"].edge_index = torch.tensor(
        [src_tx, dst_mer], dtype=torch.long
    )

    return data


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_paysim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # PaySim columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    #                 nameDest, oldbalanceDest, newbalanceDest, isFraud
    df = df.rename(columns={
        "step": "timestamp",      # step = hour index
        "amount": "amount",
        "nameOrig": "sender_id",
        "nameDest": "receiver_id",
        "isFraud": "is_fraud",
        "type": "tx_type",
    })
    df["merchant_id"] = df["receiver_id"]   # merchants are destinations in PaySim
    df["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["timestamp"], unit="h")
    return df


def load_ieee_cis(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "TransactionDT": "timestamp",
        "TransactionAmt": "amount",
        "card1": "sender_id",
        "addr1": "receiver_id",
        "ProductCD": "merchant_id",
        "isFraud": "is_fraud",
    })
    df["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["timestamp"], unit="s")
    df["tx_type"] = "PAYMENT"
    df = df.fillna({"receiver_id": "unknown", "merchant_id": "unknown"})
    df["sender_id"] = df["sender_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df["merchant_id"] = df["merchant_id"].astype(str)
    return df


def build_graphs(paysim_path: str,
                 ieee_path: str,
                 output_dir: str,
                 snapshot_hours: int = 24,
                 n_workers: int = 8) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading datasets...")
    dfs = []
    if Path(paysim_path).exists():
        dfs.append(load_paysim(paysim_path))
        log.info(f"  PaySim: {len(dfs[-1]):,} transactions")
    if Path(ieee_path).exists():
        dfs.append(load_ieee_cis(ieee_path))
        log.info(f"  IEEE-CIS: {len(dfs[-1]):,} transactions")

    if not dfs:
        raise FileNotFoundError("No dataset files found. Download data first.")

    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    log.info(f"Total transactions: {len(df):,} | Fraud rate: {df['is_fraud'].mean():.4f}")

    log.info("Engineering features...")
    df = add_time_features(df)
    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-8)
    df["velocity_1h"] = compute_velocity(df, window_hours=1)
    df["velocity_24h"] = compute_velocity(df, window_hours=24)

    account_features = build_account_features(df)
    merchant_features = build_merchant_features(df)

    # Save flat tabular features for XGBoost / anomaly detection
    tabular_cols = [
        "amount", "amount_log", "amount_zscore", "velocity_1h", "velocity_24h",
        "hour", "day_of_week", "is_fraud"
    ]
    df[tabular_cols].to_parquet(output_dir / "features_tabular.parquet", index=False)
    log.info(f"Saved tabular features → {output_dir}/features_tabular.parquet")

    # Temporal train/val/test split (80/10/10 by timestamp)
    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    splits = {
        "train": df.iloc[:train_end],
        "val":   df.iloc[train_end:val_end],
        "test":  df.iloc[val_end:],
    }
    for split_name, split_df in splits.items():
        log.info(f"  {split_name}: {len(split_df):,} tx, fraud rate: {split_df['is_fraud'].mean():.4f}")

    # Build per-split graph snapshots
    tx_feature_cols = [
        "amount", "amount_log", "amount_zscore",
        "velocity_1h", "velocity_24h", "hour", "day_of_week"
    ]

    all_snapshots = {}
    for split_name, split_df in splits.items():
        log.info(f"Building {split_name} graph snapshots (window={snapshot_hours}h)...")
        split_df = split_df.copy()
        split_df["window"] = (
            pd.to_datetime(split_df["timestamp"]).astype(np.int64)
            // (snapshot_hours * 3600 * 1_000_000_000)
        )
        snapshots = []
        for window_id, window_df in tqdm(split_df.groupby("window"),
                                          desc=f"  {split_name} snapshots"):
            if len(window_df) < 2:
                continue
            snapshot = build_snapshot(
                window_df, account_features, merchant_features, tx_feature_cols
            )
            snapshots.append(snapshot)
        all_snapshots[split_name] = snapshots
        log.info(f"  → {len(snapshots)} snapshots for {split_name}")

    # Save all graphs
    graph_path = output_dir / "graph.pt"
    torch.save(all_snapshots, graph_path)
    log.info(f"Saved graph → {graph_path}")
    log.info("Graph construction complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PyG transaction graphs")
    parser.add_argument("--paysim_path", default="data/raw/paysim.csv")
    parser.add_argument("--ieee_path", default="data/raw/ieee_cis_train.csv")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--snapshot_hours", type=int, default=24)
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    build_graphs(
        paysim_path=args.paysim_path,
        ieee_path=args.ieee_path,
        output_dir=args.output_dir,
        snapshot_hours=args.snapshot_hours,
        n_workers=args.n_workers,
    )
