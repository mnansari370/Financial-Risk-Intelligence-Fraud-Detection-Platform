"""
Graph Attention Network (GAT) for fraud detection.

Architecture:
  - 3-layer GATConv on the 'transaction' node type
  - Heterogeneous message passing via to_homogeneous projection
  - Focal loss for class imbalance
  - Final binary classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear


class FocalLoss(nn.Module):
    """Focal loss for extreme class imbalance (Lin et al., 2017)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class FraudGAT(nn.Module):
    """
    Heterogeneous GAT for transaction fraud detection.

    Node types: transaction, account, merchant
    Edge types: (transaction, sent_by, account),
                (transaction, received_by, account),
                (transaction, at, merchant)
    """

    def __init__(self,
                 tx_in_channels: int,
                 acc_in_channels: int,
                 mer_in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # ── Input projections ─────────────────────────────────────────────────
        self.tx_proj  = Linear(tx_in_channels,  hidden_channels)
        self.acc_proj = Linear(acc_in_channels, hidden_channels)
        self.mer_proj = Linear(mer_in_channels, hidden_channels)

        # ── Heterogeneous GAT layers ──────────────────────────────────────────
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels * heads
            conv = HeteroConv({
                ("transaction", "sent_by", "account"):    GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
                ("transaction", "received_by", "account"): GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
                ("transaction", "at", "merchant"):         GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
                # Reverse edges for bidirectional message passing
                ("account",  "rev_sent_by",      "transaction"): GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
                ("account",  "rev_received_by",  "transaction"): GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
                ("merchant", "rev_at",           "transaction"): GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            }, aggr="sum")
            self.convs.append(conv)

        # ── Batch norms ───────────────────────────────────────────────────────
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * heads) for _ in range(num_layers)
        ])

        # ── Classification head (operates on transaction nodes only) ─────────
        self.head = nn.Sequential(
            Linear(hidden_channels * heads, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(out_channels, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict,
                return_attention: bool = False):
        # Project all node types to the same dimension
        h = {
            "transaction": F.relu(self.tx_proj(x_dict["transaction"])),
            "account":     F.relu(self.acc_proj(x_dict["account"])),
            "merchant":    F.relu(self.mer_proj(x_dict["merchant"])),
        }

        attention_weights = []
        for i, conv in enumerate(self.convs):
            if return_attention:
                h_new, attn = conv(h, edge_index_dict, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                h_new = conv(h, edge_index_dict)

            # Apply batch norm + activation on transaction nodes
            h_new["transaction"] = self.bns[i](h_new["transaction"])
            h_new = {k: F.relu(v) for k, v in h_new.items()}
            h_new = {k: F.dropout(v, p=self.dropout, training=self.training)
                     for k, v in h_new.items()}

            # Residual connection (same shape after first layer)
            if i > 0:
                h_new = {k: h_new[k] + h.get(k, 0) for k in h_new}

            h = h_new

        # Classification: only transaction nodes are scored
        tx_emb = h["transaction"]
        logits = self.head(tx_emb).squeeze(-1)

        if return_attention:
            return logits, attention_weights
        return logits

    def predict_proba(self, x_dict, edge_index_dict) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(logits)
