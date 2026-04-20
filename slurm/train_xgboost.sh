#!/bin/bash
#SBATCH --job-name=train_xgboost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/train_xgboost_%j.out
#SBATCH --error=slurm/logs/train_xgboost_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "=== XGBoost Training ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

mkdir -p src/models/baseline results/xgboost

python -m src.models.baseline.xgboost_pipeline \
    --config configs/xgboost_config.yaml \
    --features_path data/processed/features_tabular.parquet \
    --model_path src/models/baseline/xgboost_model.pkl \
    --results_dir results/xgboost

echo "=== XGBoost training complete: $(date) ==="
