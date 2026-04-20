#!/bin/bash
#SBATCH --job-name=full_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "=== Full End-to-End Evaluation ==="
echo "Start time: $(date)"

mkdir -p results/evaluation

python -m src.evaluation.evaluate \
    --model gat \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph_account_edge.pt \
    --config configs/gat_config.yaml \
    --output_dir results/evaluation/gat

python -m src.evaluation.evaluate \
    --model xgboost \
    --checkpoint src/models/baseline/xgboost_model.pkl \
    --features_path data/processed/features_tabular.parquet \
    --config configs/xgboost_config.yaml \
    --output_dir results/evaluation/xgboost

python -m src.evaluation.evaluate \
    --model anomaly \
    --if_checkpoint src/models/anomaly/isolation_forest.pkl \
    --ae_checkpoint src/models/anomaly/autoencoder.pt \
    --features_path data/processed/features_tabular.parquet \
    --output_dir results/evaluation/anomaly

python -m src.evaluation.complementarity \
    --gat_scores results/evaluation/gat/scores.parquet \
    --xgb_scores results/evaluation/xgboost/scores.parquet \
    --anomaly_scores results/evaluation/anomaly/scores_ensemble.parquet \
    --output_dir results/evaluation/ensemble

echo "=== Full evaluation complete: $(date) ==="
echo "Results saved to results/evaluation/"