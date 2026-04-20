"""
Unsupervised anomaly detection models.

Both models are trained on LEGITIMATE transactions only, so they learn the
normal distribution of payments.  High reconstruction error (Autoencoder) or
low normality score (Isolation Forest) flags a transaction as suspicious.

  autoencoder.py      — feed-forward AE with MSE reconstruction error
  isolation_forest.py — sklearn IsolationForest with feature standardisation
"""

from src.models.anomaly.autoencoder import Autoencoder

__all__ = ["Autoencoder"]
