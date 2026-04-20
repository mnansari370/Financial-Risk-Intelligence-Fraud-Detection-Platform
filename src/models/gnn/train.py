"""
Training script for FraudEdgeGAT.

Usage:
  python -m src.models.gnn.train \
      --config configs/gat_config.yaml \
      --graph_path data/processed/graph_account_edge.pt \
      --checkpoint_dir src/models/gnn/checkpoints \
      --results_dir results/gat
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.gnn.model import FocalLoss, FraudEdgeGAT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_epoch(model, snapshots, optimizer, loss_fn, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    all_scores, all_labels = [], []

    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        optimizer.zero_grad()

        logits = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        labels = snapshot.edge_label.float()
        loss   = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_scores.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.detach().cpu())

    metrics = compute_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_scores).numpy(),
    )
    return total_loss / max(len(snapshots), 1), metrics


@torch.no_grad()
def eval_epoch(model, snapshots, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_scores, all_labels = [], []

    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        logits   = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        labels   = snapshot.edge_label.float()

        total_loss += loss_fn(logits, labels).item()
        all_scores.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    metrics = compute_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_scores).numpy(),
    )
    return total_loss / max(len(snapshots), 1), metrics


@torch.no_grad()
def score_snapshots(model, snapshots, device):
    """Return (scores, labels, tx_ids) arrays across all snapshots."""
    model.eval()
    all_scores, all_labels, all_tx_ids = [], [], []

    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        logits   = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(snapshot.edge_label.cpu().numpy())
        all_tx_ids.append(snapshot.tx_id.cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_tx_ids),
    )


def train(config_path: str, graph_path: str, checkpoint_dir: str, results_dir: str,
          sweep_id: int = None):
    cfg = load_config(config_path)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    log.info(f"Loading graph snapshots from {graph_path} ...")
    graphs = torch.load(graph_path, map_location="cpu")
    train_snapshots = graphs["train"]
    val_snapshots   = graphs["val"]
    test_snapshots  = graphs["test"]

    log.info(
        f"Snapshots — train: {len(train_snapshots)} | "
        f"val: {len(val_snapshots)} | test: {len(test_snapshots)}"
    )

    sample = train_snapshots[0]
    node_input_dim = sample.x.shape[1]
    edge_input_dim = sample.edge_attr.shape[1]
    log.info(f"Node features: {node_input_dim} | Edge features: {edge_input_dim}")

    m_cfg = cfg["model"]
    model = FraudEdgeGAT(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_channels=m_cfg["hidden_channels"],
        out_channels=m_cfg["out_channels"],
        num_layers=m_cfg["num_layers"],
        heads=m_cfg["heads"],
        dropout=m_cfg["dropout"],
        residual=m_cfg["residual"],
    ).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    t_cfg = cfg["training"]
    optimizer = Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])
    loss_fn   = FocalLoss(
        alpha=t_cfg["focal_loss"]["alpha"],
        gamma=t_cfg["focal_loss"]["gamma"],
    )

    best_val_pr_auc  = -1.0
    patience_counter = 0
    best_ckpt        = Path(checkpoint_dir) / "best_model.pt"

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, train_m = train_epoch(
            model=model, snapshots=train_snapshots,
            optimizer=optimizer, loss_fn=loss_fn,
            device=device, grad_clip=t_cfg["grad_clip"],
        )
        val_loss, val_m = eval_epoch(
            model=model, snapshots=val_snapshots,
            loss_fn=loss_fn, device=device,
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}  PR-AUC {train_m['pr_auc']:.4f} | "
            f"val loss {val_loss:.4f}  PR-AUC {val_m['pr_auc']:.4f}  "
            f"F1 {val_m['f1']:.4f}"
        )

        if val_m["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc  = val_m["pr_auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics":     val_m,
                    "config":          cfg,
                },
                best_ckpt,
            )
            log.info(f"  → Best checkpoint saved (val PR-AUC={best_val_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["early_stopping_patience"]:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    log.info(f"Training complete. Best val PR-AUC: {best_val_pr_auc:.4f}")

    # ── Final test-set evaluation ──────────────────────────────────────────────
    log.info("Loading best checkpoint for test-set evaluation ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_m = eval_epoch(model, test_snapshots, loss_fn, device)
    log.info("\n=== Test Set Metrics ===")
    print_metrics(test_m)

    scores, labels, tx_ids = score_snapshots(model, test_snapshots, device)

    pd.DataFrame({
        "tx_id":    tx_ids,
        "score":    scores,
        "is_fraud": labels.astype(int),
    }).to_parquet(Path(results_dir) / "scores.parquet", index=False)

    pd.DataFrame([test_m]).to_csv(Path(results_dir) / "metrics.csv", index=False)
    log.info(f"Test scores saved → {results_dir}/scores.parquet")
    log.info(f"Test metrics saved → {results_dir}/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--graph_path",     required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir",    required=True)
    parser.add_argument("--sweep_id", type=int, default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        graph_path=args.graph_path,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        sweep_id=args.sweep_id,
    )
