"""
Feed-forward autoencoder for unsupervised fraud detection.
Trained on legitimate transactions only.

This corrected version:
- uses explicit feature columns from config
- excludes tx_id / timestamp / target from model input
- standardizes inputs using training-legitimate statistics
- stores normalization stats in the checkpoint
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoder_dims: list, latent_dim: int, decoder_dims: list, dropout: float = 0.1):
        super().__init__()

        enc_layers = []
        in_dim = input_dim
        for dim in encoder_dims:
            enc_layers += [nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = dim
        enc_layers += [nn.Linear(in_dim, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = [nn.Linear(latent_dim, decoder_dims[0]), nn.ReLU()]
        in_dim = decoder_dims[0]
        for dim in decoder_dims[1:]:
            dec_layers += [nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = dim
        dec_layers += [nn.Linear(in_dim, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def standardize(train_x: np.ndarray, other_x: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (other_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def train_autoencoder(config_path: str, features_path: str, output_dir: str):
    cfg_all = load_config(config_path)
    cfg = cfg_all["autoencoder"]
    feature_cols = cfg_all["data"]["feature_columns"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    df = pd.read_parquet(features_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in feature_cols + ["is_fraud"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]

    df_legit = df_train[df_train["is_fraud"] == 0]
    log.info(f"Legitimate training samples: {len(df_legit):,} (from {len(df_train):,} total train)")

    X_train = df_legit[feature_cols].fillna(0).values.astype(np.float32)
    X_val_legit = df_val[df_val["is_fraud"] == 0][feature_cols].fillna(0).values.astype(np.float32)

    X_train_std, X_val_std, mean_vec, std_vec = standardize(X_train, X_val_legit)

    dataset = TensorDataset(torch.tensor(X_train_std, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    input_dim = X_train_std.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        encoder_dims=cfg["encoder_dims"],
        latent_dim=cfg["latent_dim"],
        decoder_dims=cfg["decoder_dims"],
        dropout=cfg["dropout"],
    ).to(device)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(output_dir) / "autoencoder.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat = model(batch_x)
            loss = loss_fn(x_hat, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(len(loader), 1)

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:03d} | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "config": cfg,
                    "feature_cols": feature_cols,
                    "feature_mean": mean_vec.astype(np.float32),
                    "feature_std": std_vec.astype(np.float32),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # Threshold on val legitimate reconstruction errors
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32).to(device)
        recon_errors = model.reconstruction_error(X_val_tensor).cpu().numpy()

    threshold = float(np.percentile(recon_errors, cfg["threshold_percentile"]))
    log.info(f"Anomaly threshold (p{cfg['threshold_percentile']}): {threshold:.6f}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["threshold"] = threshold
    torch.save(ckpt, ckpt_path)
    log.info(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/anomaly_config.yaml")
    parser.add_argument("--features_path", default="data/processed/features_tabular.parquet")
    parser.add_argument("--output_dir", default="src/models/anomaly")
    args = parser.parse_args()

    train_autoencoder(args.config, args.features_path, args.output_dir)