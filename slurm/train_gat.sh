#!/bin/bash
#SBATCH --job-name=train_gat
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/train_gat_%j.out
#SBATCH --error=slurm/logs/train_gat_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"
CONFIG="${CONFIG:-configs/gat_config.yaml}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GAT Training ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Config: $CONFIG"

mkdir -p src/models/gnn/checkpoints results/gat/logs

python -m src.models.gnn.train \
    --config "$CONFIG" \
    --graph_path data/processed/graph_account_edge.pt \
    --checkpoint_dir src/models/gnn/checkpoints \
    --results_dir results/gat

echo "=== Training complete: $(date) ==="