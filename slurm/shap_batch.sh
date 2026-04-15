#!/bin/bash
#SBATCH --job-name=shap_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/shap_batch_%j.out
#SBATCH --error=slurm/logs/shap_batch_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/aiongpfs/users/nmo/financial-fraud-detection"
CONDA_ENV="fraud-detection"
TOP_K="${TOP_K:-1000}"     # number of top-scored transactions to explain

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Batch SHAP Explanations (top-$TOP_K transactions) ==="
echo "Start time: $(date)"

mkdir -p results/shap

# SHAP for XGBoost (TreeExplainer — fast)
python -m src.explainability.shap_explainer \
    --model xgboost \
    --checkpoint src/models/baseline/xgboost_model.pkl \
    --features_path data/processed/features_tabular.parquet \
    --scores_path results/evaluation/xgboost/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/xgboost

# SHAP for GAT (GradientExplainer — GPU-accelerated)
python -m src.explainability.shap_explainer \
    --model gat \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph.pt \
    --scores_path results/evaluation/gat/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/gat

# Extract GAT attention weights for top-K transactions
python -m src.explainability.attention_extractor \
    --checkpoint src/models/gnn/checkpoints/best_model.pt \
    --graph_path data/processed/graph.pt \
    --scores_path results/evaluation/gat/scores.parquet \
    --top_k "$TOP_K" \
    --output_dir results/shap/attention

echo "=== SHAP batch complete: $(date) ==="
