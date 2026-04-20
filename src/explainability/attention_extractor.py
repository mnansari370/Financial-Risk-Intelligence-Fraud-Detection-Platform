"""
Attention extractor for FraudEdgeGAT.

Extracts per-edge mean attention weights from the first GAT layer
for the top-K highest-scored transactions.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_attention(checkpoint_path: str, graph_path: str, scores_path: str, top_k: int, output_dir: str):
    from src.models.gnn.model import FraudEdgeGAT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]["model"]

    graphs = torch.load(graph_path, map_location="cpu")
    test_snapshots = graphs["test"]
    sample = test_snapshots[0]

    model = FraudEdgeGAT(
        node_input_dim=sample.x.shape[1],
        edge_input_dim=sample.edge_attr.shape[1],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
        dropout=0.0,
        residual=cfg["residual"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores_df = pd.read_parquet(scores_path)
    top_tx_ids = set(scores_df.nlargest(top_k, "score")["tx_id"].tolist())

    records = []
    log.info(f"Extracting attention weights for top-{top_k} transactions...")

    first_conv = model.convs[0]

    for snapshot in test_snapshots:
        snapshot = snapshot.to(device)

        tx_ids = snapshot.tx_id.detach().cpu().tolist()
        wanted_positions = [i for i, tx_id in enumerate(tx_ids) if tx_id in top_tx_ids]
        if not wanted_positions:
            continue

        x0 = torch.relu(model.node_proj(snapshot.x))

        with torch.no_grad():
            out, (edge_idx_used, alpha) = first_conv(
                x0,
                snapshot.edge_index,
                return_attention_weights=True,
            )
            logits = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            scores = torch.sigmoid(logits).detach().cpu().numpy()

        alpha_mean = alpha.mean(dim=1).detach().cpu().numpy()
        edge_src = edge_idx_used[0].detach().cpu().numpy()
        edge_dst = edge_idx_used[1].detach().cpu().numpy()

        for pos in wanted_positions:
            tx_id = tx_ids[pos]
            src = int(snapshot.edge_index[0, pos].item())
            dst = int(snapshot.edge_index[1, pos].item())

            matching = (edge_src == src) & (edge_dst == dst)
            edge_attention = float(alpha_mean[matching].mean()) if matching.any() else 0.0

            records.append({
                "tx_id": int(tx_id),
                "fraud_score": float(scores[pos]),
                "src_node": src,
                "dst_node": dst,
                "mean_attention": edge_attention,
            })

    df_out = pd.DataFrame(records)
    out_path = Path(output_dir) / "attention_weights.parquet"
    df_out.to_parquet(out_path, index=False)
    log.info(f"Attention weights saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph_path", required=True)
    parser.add_argument("--scores_path", required=True)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    extract_attention(
        args.checkpoint,
        args.graph_path,
        args.scores_path,
        args.top_k,
        args.output_dir,
    )