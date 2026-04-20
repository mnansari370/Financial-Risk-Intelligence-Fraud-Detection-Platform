#!/bin/bash
#SBATCH --job-name=shap_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/shap_batch_%j.out
#SBATCH --error=slurm/logs/shap_batch_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"
TOP_K="${TOP_K:-1000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "=== Batch Explainability (top-$TOP_K transactions) ==="
echo "Start time: $(date)"

mkdir -p results/shap

python -m src.explainability.shap_explainer \
    --model xgboost \
    --checkpoint src/models/baseline/xgboost_model.pkl \
    --features_path data/processed/features_tabular.parquet \
    --scores_path results/evaluation/xgboost/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/xgboost

python -m src.explainability.shap_explainer \
    --model gat \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph_account_edge.pt \
    --scores_path results/evaluation/gat/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/gat

python -m src.explainability.attention_extractor \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph_account_edge.pt \
    --scores_path results/evaluation/gat/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/attention

echo "=== Explainability batch complete: $(date) ==="