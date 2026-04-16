# Financial Risk Intelligence & Fraud Detection Platform

A graph-based fraud detection system that combines Graph Attention Networks (GAT), XGBoost, and unsupervised anomaly detection to identify fraudulent financial transactions. Built as part of my MSc AI thesis at the University of Luxembourg.

The core idea is to model transactions as a heterogeneous graph — connecting transactions to sender accounts, receiver accounts, and merchants — and use attention mechanisms to capture relational fraud patterns that tabular models miss entirely.

---

## Architecture

```
Raw transactions (PaySim + IEEE-CIS)
         │
         ▼
   Graph Construction
   ┌─────────────────────────────────┐
   │  Nodes: transaction, account,   │
   │         merchant                │
   │  Edges: sent_by, received_by,   │
   │         at  (+reverse edges)    │
   │  Temporal 24h snapshots         │
   └─────────────────────────────────┘
         │
    ┌────┴─────┬──────────────┐
    ▼          ▼              ▼
 FraudGAT   XGBoost     Anomaly Detection
 (PyG)      + SMOTE     (Isolation Forest
            + Focal     + Autoencoder)
            Loss
    │          │              │
    └────┬─────┴──────────────┘
         ▼
   SHAP Explainability
   + GAT Attention Weights
         │
         ▼
   LLM SAR Generator (GPT-4o-mini)
         │
         ▼
   Streamlit Dashboard
```

### Models

**FraudGAT** — 3-layer `HeteroConv` with `GATConv` operators. Each transaction attends over its connected accounts and merchants, with reverse edges for bidirectional message passing. Trained with Focal Loss (α=0.25, γ=2.0) to handle the extreme class imbalance (~0.1–3.5% fraud rate). Early stopping on validation PR-AUC.

**XGBoost Baseline** — Gradient boosting on tabular features (amount, velocity, temporal features). SMOTE applied on training data only; threshold calibrated at 80% recall on the validation set. Serves as a strong non-graph baseline.

**Anomaly Detection** — Both Isolation Forest and a feed-forward Autoencoder are trained exclusively on legitimate transactions. Fraud signals as reconstruction error (autoencoder) or anomaly score (isolation forest). No labels required during training.

**SAR Generator** — Calls GPT-4o-mini (or local Llama-3-8B) with transaction context, SHAP feature contributions, and multi-model fraud scores to generate EU AMLD6-compliant Suspicious Activity Reports in structured JSON.

---

## Results

| Model | PR-AUC | ROC-AUC | F1 | Prec@80%R | FPR |
|-------|--------|---------|-----|-----------|-----|
| FraudGAT | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| Isolation Forest | — | — | — | — | — |
| Autoencoder | — | — | — | — | — |

*Results will be updated after training completes on ULHPC cluster.*

---

## Datasets

| Dataset | Source | Size | Fraud Rate |
|---------|--------|------|------------|
| PaySim | Kaggle (ntnu-testimon/paysim1) | ~6.3M transactions | ~0.13% |
| IEEE-CIS Fraud Detection | Kaggle (ieee-fraud-detection) | ~590K transactions | ~3.5% |

Download and place in `data/raw/`:
- `data/raw/paysim.csv`
- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

---

## Setup

```bash
# Clone the repo
git clone https://github.com/mnansari370/Financial-Risk-Intelligence-Fraud-Detection-Platform
cd Financial-Risk-Intelligence-Fraud-Detection-Platform

# Create and activate conda environment
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt

# Set up PYTHONPATH and verify imports
bash setup_env.sh

# Add your OpenAI key to .env (for SAR generation)
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## Running the Pipeline

All GPU jobs are submitted via SLURM (tested on ULHPC):

```bash
# 1. Build graphs (CPU node, ~4h for full dataset)
sbatch slurm/build_graph.sh

# 2. Train XGBoost baseline (CPU, runs locally)
conda activate fraud-detection
python -m src.models.baseline.xgboost_pipeline

# 3. Train FraudGAT (GPU node, ~12h)
sbatch slurm/train_gat.sh

# 4. Train anomaly detection models (GPU/CPU)
sbatch slurm/train_autoencoder.sh

# 5. Full evaluation across all models
sbatch slurm/full_eval.sh

# 6. Generate SHAP explanations
sbatch slurm/shap_batch.sh

# 7. Launch dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

---

## Project Structure

```
.
├── configs/                  # YAML configs for all models
│   ├── gat_config.yaml
│   ├── xgboost_config.yaml
│   └── anomaly_config.yaml
├── data/
│   ├── raw/                  # Raw CSVs (gitignored)
│   └── processed/            # Built graphs + tabular features (gitignored)
├── notebooks/                # Exploratory analysis
├── results/                  # Evaluation outputs (gitignored)
│   ├── evaluation/
│   └── shap/
├── slurm/                    # SLURM job scripts
├── src/
│   ├── augmentation/         # SMOTE helpers
│   ├── dashboard/            # Streamlit app
│   ├── evaluation/           # Metrics (PR-AUC, F1, FPR, ...)
│   ├── explainability/       # SHAP + attention weight extraction
│   ├── graph/                # PyG HeteroData graph builder
│   ├── llm/                  # SAR generator
│   └── models/
│       ├── anomaly/          # Autoencoder + Isolation Forest
│       ├── baseline/         # XGBoost pipeline
│       └── gnn/              # FraudGAT model + training loop
├── requirements.txt
└── setup_env.sh
```

---

## Key Design Decisions

**Why graphs?** Fraud rarely happens in isolation — a compromised account makes multiple transactions, merchants get targeted in bursts, and money mule chains involve rings of accounts. A graph lets the model see these structural patterns directly.

**Why Focal Loss?** With ~0.1% fraud rate (PaySim), standard cross-entropy gets dominated by legitimate transactions. Focal Loss down-weights easy negatives so the model actually learns to identify the hard fraud cases.

**Why temporal splits?** Random train/test splits leak future information (a model trained on shuffled data will see future account behaviour during training). All splits are strictly by timestamp: first 80% → train, next 10% → val, last 10% → test.

**Why PR-AUC?** With extreme class imbalance, ROC-AUC can look great even with poor fraud recall. PR-AUC focuses on the minority class performance that actually matters.

---

## Technologies

- PyTorch 2.2 + PyTorch Geometric 2.5 (GATConv, HeteroConv)
- XGBoost 2.0 + imbalanced-learn (SMOTE)
- SHAP 0.45
- Streamlit 1.35 + PyVis (graph visualisation)
- OpenAI API (GPT-4o-mini) / HuggingFace Transformers (Llama-3-8B)
- ULHPC cluster with SLURM (CUDA 12.1, A100/V100 GPUs)
