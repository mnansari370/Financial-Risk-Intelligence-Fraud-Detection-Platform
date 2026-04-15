"""
GAT Training Script

Usage:
  python -m src.models.gnn.train --config configs/gat_config.yaml \
      --graph_path data/processed/graph.pt --checkpoint_dir src/models/gnn/checkpoints
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.gnn.model import FocalLoss, FraudGAT
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_epoch(model, snapshots, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        optimizer.zero_grad()

        logits = model(snapshot.x_dict, snapshot.edge_index_dict)
        labels = snapshot["transaction"].y.float()

        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(labels.numpy(), torch.sigmoid(logits).numpy())
    return total_loss / len(snapshots), metrics


@torch.no_grad()
def eval_epoch(model, snapshots, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for snapshot in snapshots:
        snapshot = snapshot.to(device)
        logits = model(snapshot.x_dict, snapshot.edge_index_dict)
        labels = snapshot["transaction"].y.float()

        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(labels.numpy(), torch.sigmoid(logits).numpy())
    return total_loss / len(snapshots), metrics


def train(config_path: str, graph_path: str, checkpoint_dir: str,
          results_dir: str, sweep_id: int = None):
    cfg = load_config(config_path)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    log.info(f"Loading graphs from {graph_path}...")
    graphs = torch.load(graph_path, map_location="cpu")
    train_snapshots = graphs["train"]
    val_snapshots   = graphs["val"]
    log.info(f"  Train snapshots: {len(train_snapshots)} | Val: {len(val_snapshots)}")

    # Infer node feature dimensions from first snapshot
    sample = train_snapshots[0]
    tx_in  = sample["transaction"].x.shape[1]
    acc_in = sample["account"].x.shape[1]
    mer_in = sample["merchant"].x.shape[1]
    log.info(f"  Feature dims — tx:{tx_in}, acc:{acc_in}, mer:{mer_in}")

    m_cfg = cfg["model"]
    model = FraudGAT(
        tx_in_channels=tx_in,
        acc_in_channels=acc_in,
        mer_in_channels=mer_in,
        hidden_channels=m_cfg["hidden_channels"],
        out_channels=m_cfg["out_channels"],
        num_layers=m_cfg["num_layers"],
        heads=m_cfg["heads"],
        dropout=m_cfg["dropout"],
    ).to(device)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    t_cfg = cfg["training"]
    optimizer = Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])
    loss_fn = FocalLoss(
        alpha=t_cfg["focal_loss"]["alpha"],
        gamma=t_cfg["focal_loss"]["gamma"],
    )

    best_val_pr_auc = 0.0
    patience_counter = 0
    best_ckpt = Path(checkpoint_dir) / "best_model.pt"

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, train_metrics = train_epoch(
            model, train_snapshots, optimizer, loss_fn, device
        )
        val_loss, val_metrics = eval_epoch(
            model, val_snapshots, loss_fn, device
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f} PR-AUC: {train_metrics['pr_auc']:.4f} | "
            f"Val loss: {val_loss:.4f} PR-AUC: {val_metrics['pr_auc']:.4f} "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
            }, best_ckpt)
            log.info(f"  → New best (PR-AUC={best_val_pr_auc:.4f}), saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["early_stopping_patience"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Training complete. Best val PR-AUC: {best_val_pr_auc:.4f}")
    log.info(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sweep_id", type=int, default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        graph_path=args.graph_path,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        sweep_id=args.sweep_id,
    )
