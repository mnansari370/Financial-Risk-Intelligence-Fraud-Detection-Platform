"""
GAT Attention Weight Extractor
Extracts per-edge attention weights from the trained GAT model.
These are used in the Streamlit dashboard to highlight suspicious edges.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_attention(checkpoint_path: str, graph_path: str,
                      scores_path: str, top_k: int, output_dir: str):
    from src.models.gnn.model import FraudGAT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]["model"]

    graphs = torch.load(graph_path, map_location="cpu")
    test_snapshots = graphs["test"]
    sample = test_snapshots[0]

    model = FraudGAT(
        tx_in_channels=sample["transaction"].x.shape[1],
        acc_in_channels=sample["account"].x.shape[1],
        mer_in_channels=sample["merchant"].x.shape[1],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores_df = pd.read_parquet(scores_path)
    top_tx_ids = set(scores_df.nlargest(top_k, "score")["tx_id"].tolist())

    all_attention = []
    log.info(f"Extracting attention weights for top-{top_k} transactions...")

    for snapshot in test_snapshots:
        snapshot = snapshot.to(device)
        tx_ids = snapshot["transaction"].tx_id.tolist()

        if not any(tid in top_tx_ids for tid in tx_ids):
            continue

        with torch.no_grad():
            logits, attn_weights = model(
                snapshot.x_dict,
                snapshot.edge_index_dict,
                return_attention=True
            )

        scores = torch.sigmoid(logits).cpu().numpy()

        for i, tx_id in enumerate(tx_ids):
            if tx_id not in top_tx_ids:
                continue
            all_attention.append({
                "tx_id": tx_id,
                "fraud_score": float(scores[i]),
                # attn_weights is a list of (edge_index, alpha) per layer per edge type
                # Store layer-0 attention as a proxy for interpretability
                "attention_layer0": [
                    a[i].mean().item() if a.dim() > 1 else float(a[i])
                    for a in attn_weights[0].values()
                    if isinstance(a, torch.Tensor)
                ],
            })

    df_out = pd.DataFrame(all_attention)
    out_path = Path(output_dir) / "attention_weights.parquet"
    df_out.to_parquet(out_path, index=False)
    log.info(f"Attention weights saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    extract_attention(
        args.checkpoint, args.graph_path,
        args.scores_path, args.top_k, args.output_dir
    )
