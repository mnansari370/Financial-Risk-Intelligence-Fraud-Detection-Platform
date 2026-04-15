#!/bin/bash
#SBATCH --job-name=hparam_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-19          # 20 hyperparameter configurations
#SBATCH --output=slurm/logs/hparam_%A_%a.out
#SBATCH --error=slurm/logs/hparam_%A_%a.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Hyperparameter Sweep ==="
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Each array task uses a different config from configs/sweep/
CONFIG="configs/sweep/config_${SLURM_ARRAY_TASK_ID}.yaml"

mkdir -p "results/sweep/run_${SLURM_ARRAY_TASK_ID}"

python -m src.models.gnn.train \
    --config "$CONFIG" \
    --graph_path data/processed/graph.pt \
    --checkpoint_dir "src/models/gnn/checkpoints/sweep_${SLURM_ARRAY_TASK_ID}" \
    --results_dir "results/sweep/run_${SLURM_ARRAY_TASK_ID}" \
    --sweep_id "$SLURM_ARRAY_TASK_ID"

echo "=== Sweep task $SLURM_ARRAY_TASK_ID complete: $(date) ==="
