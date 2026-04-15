#!/bin/bash
# Environment setup script for the Fraud Detection Platform
# Run once on the HPC login node: bash setup_env.sh

set -euo pipefail

CONDA_ENV="fraud-detection"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Financial Fraud Detection Platform — Environment Setup ==="
echo "Project: $PROJECT_DIR"
echo ""

# ── 1. Activate conda environment ─────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "✓ Conda env '$CONDA_ENV' activated ($(python --version))"

# ── 2. Install package in editable mode ───────────────────────────────────────
cd "$PROJECT_DIR"
pip install -e . --quiet 2>/dev/null || echo "  (no setup.py — using PYTHONPATH)"

# ── 3. Set PYTHONPATH so src.* imports work ───────────────────────────────────
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
echo "✓ PYTHONPATH set to include $PROJECT_DIR"
echo ""
echo "Add this to your ~/.bashrc to make it permanent:"
echo "  export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\""
echo ""

# ── 4. Check .env file ────────────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠  No .env file found. Creating template..."
    cat > "$PROJECT_DIR/.env" << 'EOF'
# OpenAI API key for SAR generation (GPT-4o-mini)
OPENAI_API_KEY=your_key_here

# (Optional) HuggingFace token for gated models like Llama-3
HF_TOKEN=your_hf_token_here
EOF
    echo "   Created .env — fill in your API keys before running the SAR generator."
else
    echo "✓ .env file found"
fi

# ── 5. Verify all key imports ─────────────────────────────────────────────────
echo ""
echo "=== Verifying package imports ==="
python -c "
import torch, torch_geometric, xgboost, shap, imblearn, networkx
import streamlit, openai, pandas, transformers
print(f'  torch:           {torch.__version__}')
print(f'  torch_geometric: {torch_geometric.__version__}')
print(f'  xgboost:         {xgboost.__version__}')
print(f'  shap:            {shap.__version__}')
print(f'  imbalanced-learn:{imblearn.__version__}')
print(f'  streamlit:       {streamlit.__version__}')
print(f'  openai:          {openai.__version__}')
print()
print(f'  CUDA available:  {torch.cuda.is_available()} (expected False on login node)')
print()
print('✓ All packages verified successfully')
" 2>&1 | grep -v UserWarning

echo ""
echo "=== Setup complete! Next steps: ==="
echo ""
echo "1. Download datasets:"
echo "   data/raw/paysim.csv          (from Kaggle: ntnu-testimon-paysim1)"
echo "   data/raw/ieee_cis_train.csv  (from Kaggle: ieee-fraud-detection)"
echo ""
echo "2. Build the graph:"
echo "   sbatch slurm/build_graph.sh"
echo ""
echo "3. Train XGBoost baseline:"
echo "   conda activate fraud-detection"
echo "   python -m src.models.baseline.xgboost_pipeline"
echo ""
echo "4. Train GAT (on GPU node):"
echo "   sbatch slurm/train_gat.sh"
echo ""
echo "5. Run full evaluation:"
echo "   sbatch slurm/full_eval.sh"
echo ""
echo "6. Launch dashboard:"
echo "   streamlit run src/dashboard/app.py --server.port 8501"
