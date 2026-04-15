#!/bin/bash
#SBATCH --job-name=full_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/full_eval_%j.out
#SBATCH --error=slurm/logs/full_eval_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Full End-to-End Evaluation ==="
echo "Start time: $(date)"

mkdir -p results/evaluation

# 1. Evaluate GAT
python -m src.evaluation.evaluate \
    --model gat \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph.pt \
    --config configs/gat_config.yaml \
    --output_dir results/evaluation/gat

# 2. Evaluate XGBoost baseline
python -m src.evaluation.evaluate \
    --model xgboost \
    --checkpoint src/models/baseline/xgboost_model.pkl \
    --features_path data/processed/features_tabular.parquet \
    --config configs/xgboost_config.yaml \
    --output_dir results/evaluation/xgboost

# 3. Evaluate anomaly detection (Isolation Forest + Autoencoder)
python -m src.evaluation.evaluate \
    --model anomaly \
    --if_checkpoint src/models/anomaly/isolation_forest.pkl \
    --ae_checkpoint src/models/anomaly/autoencoder.pt \
    --features_path data/processed/features_tabular.parquet \
    --output_dir results/evaluation/anomaly

# 4. Compute complementarity metrics (ensemble overlap analysis)
python -m src.evaluation.complementarity \
    --gat_scores results/evaluation/gat/scores.parquet \
    --xgb_scores results/evaluation/xgboost/scores.parquet \
    --anomaly_scores results/evaluation/anomaly/scores.parquet \
    --output_dir results/evaluation/ensemble

echo "=== Full evaluation complete: $(date) ==="
echo "Results saved to results/evaluation/"
