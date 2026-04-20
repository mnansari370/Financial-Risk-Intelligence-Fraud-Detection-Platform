"""
FraudEdgeGAT — Graph Attention Network for edge-level transaction fraud detection.

Architecture overview:
  1. A linear projection maps raw account features into the hidden space.
  2. Multiple GATConv layers aggregate information from neighbouring accounts,
     learning which relationships matter via attention weights.
  3. For each transaction edge, the source embedding, destination embedding,
     and edge features are concatenated and passed through a small MLP that
     outputs a single fraud logit.

This edge-centric design means the model answers: "given everything the graph
knows about the sender account, the receiver account, and the transaction
itself — how suspicious is this specific payment?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification under extreme class imbalance.

    Standard cross-entropy treats every sample equally, so on PaySim's 0.13%
    fraud rate the gradient is overwhelmingly dominated by easy legitimate
    transactions.  Focal loss down-weights well-classified examples via the
    (1 - p_t)^gamma term, forcing the model to focus on hard/rare cases.

    alpha upweights the POSITIVE (fraud) class.  For a 0.13% fraud rate,
    alpha ≈ 0.95 gives roughly 19x more weight to fraud samples — see
    gat_config.yaml for the chosen value and the rationale.
    """

    def __init__(self, alpha: float = 0.95, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs   = torch.sigmoid(logits)
        # p_t: probability of the true class
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t: per-sample class weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class FraudEdgeGAT(nn.Module):
    """
    Graph Attention Network for edge-level fraud scoring.

    Graph convention:
      - Nodes  = accounts (features: velocity stats, volume stats, risk scores)
      - Edges  = transactions  (features: amount, time, behavioural flags)
      - Labels = per-edge binary fraud indicator

    Forward pass:
      encode_nodes  → multi-layer GAT builds contextual account embeddings
      decode_edges  → MLP(src_emb ‖ dst_emb ‖ edge_attr) → fraud logit
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout  = dropout
        self.residual = residual

        # Project raw node features to hidden_channels before the first GATConv
        self.node_proj = nn.Linear(node_input_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = hidden_channels if layer_idx == 0 else hidden_channels * heads
            self.convs.append(GATConv(
                in_channels=in_dim,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=True,       # concatenate heads → output dim = hidden * heads
                add_self_loops=False,
            ))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))

        final_node_dim = hidden_channels * heads

        # Edge classifier: [src ‖ dst ‖ edge_attr] → 1 logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(final_node_dim * 2 + edge_input_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GAT layers to produce contextual account embeddings."""
        h = F.relu(self.node_proj(x))
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual connection when shapes match (layers 1+)
            h = h + h_new if (self.residual and h_new.shape == h.shape) else h_new
        return h

    def decode_edges(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        chunk_size: int = 300_000,
    ) -> torch.Tensor:
        """
        Score each transaction edge using the embeddings of both endpoints.

        Large snapshots (millions of edges) require chunking: building the full
        [N_edges × (2*hidden + edge_dim)] tensor at once exhausts 16GB VRAM.
        Chunking processes 300K edges at a time; gradients flow correctly because
        the final torch.cat re-joins the chunks before the loss is computed.
        """
        src, dst = edge_index
        if src.shape[0] <= chunk_size:
            edge_input = torch.cat([node_emb[src], node_emb[dst], edge_attr], dim=-1)
            return self.edge_mlp(edge_input).squeeze(-1)

        # Process in chunks to avoid OOM on large snapshots
        chunks = []
        for i in range(0, src.shape[0], chunk_size):
            s = src[i : i + chunk_size]
            d = dst[i : i + chunk_size]
            a = edge_attr[i : i + chunk_size]
            chunks.append(self.edge_mlp(
                torch.cat([node_emb[s], node_emb[d], a], dim=-1)
            ).squeeze(-1))
        return torch.cat(chunks)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        node_emb = self.encode_nodes(x, edge_index)
        return self.decode_edges(node_emb, edge_index, edge_attr)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid-calibrated fraud probabilities (no gradient tracking)."""
        return torch.sigmoid(self.forward(x, edge_index, edge_attr))
