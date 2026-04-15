#!/bin/bash
#SBATCH --job-name=train_autoencoder
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/train_autoencoder_%j.out
#SBATCH --error=slurm/logs/train_autoencoder_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Autoencoder Training (legitimate-only) ==="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

mkdir -p src/models/anomaly results/anomaly

python -m src.models.anomaly.train_autoencoder \
    --config configs/anomaly_config.yaml \
    --features_path data/processed/features_tabular.parquet \
    --output_dir src/models/anomaly

echo "=== Autoencoder training complete: $(date) ==="
