"""
FraudEdgeGAT — Graph Attention Network for transaction-level fraud detection.

Key exports:
  FraudEdgeGAT  — the GAT model class
  FocalLoss     — focal loss for extreme class imbalance (alpha upweights fraud)
"""

from src.models.gnn.model import FocalLoss, FraudEdgeGAT

__all__ = ["FraudEdgeGAT", "FocalLoss"]
