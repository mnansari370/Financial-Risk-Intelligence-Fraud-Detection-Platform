# Financial Risk Intelligence & Fraud Detection Platform

A production-style fraud detection system that combines Graph Attention Networks (GAT), XGBoost, and unsupervised anomaly detection to identify fraudulent financial transactions — complete with SHAP explainability, an interactive monitoring dashboard, and automated Suspicious Activity Report (SAR) generation.

Built and trained on ULHPC HPC infrastructure (Tesla V100, 32 GB VRAM) as part of my MSc Computer Science degree at the University of Luxembourg.

---

## Results

Evaluated on the held-out test set of 6.36M PaySim transactions (strict temporal split — no data leakage). The random baseline PR-AUC at 0.59% fraud rate is **0.006**.

| Model | PR-AUC | ROC-AUC | F1 | Prec @ 80% Recall | FPR |
|---|---|---|---|---|---|
| **XGBoost** | **0.797** | **0.985** | 0.578 | 48.3% | 0.58% |
| **Full Ensemble** | 0.739 | 0.984 | **0.679** | 38.9% | **0.04%** |
| Autoencoder (unsupervised) | 0.436 | 0.874 | — | 1.8% | 0.0% |
| Isolation Forest (unsupervised) | 0.220 | 0.870 | 0.155 | 2.1% | 3.0% |
| FraudGAT | 0.154 | 0.819 | 0.203 | 1.6% | 0.27% |

**XGBoost achieves PR-AUC 0.797 — 133× above the random baseline.** The full ensemble reaches F1 0.679, a 17% improvement over XGBoost alone, with an extremely low false positive rate of 0.04% (crucial for production systems where analyst time is the bottleneck).

**A note on FraudGAT:** The GAT's lower PR-AUC reflects a well-known challenge in temporal fraud detection — distribution shift. The training period contains only 0.08% fraud, while the test period (PaySim's final 10% by timestamp) contains 0.59% fraud, a 7× shift. The graph model trains on a fraud signal that is effectively absent, then is evaluated on a period where fraud has surged. This is a real-world condition that tabular models with hand-engineered velocity features handle more robustly. The GAT still achieves ROC-AUC 0.819, demonstrating it has learned a meaningful fraud signal.

---

## Architecture

```
Raw Transactions (PaySim — 6.36M)
              │
              ▼
    ┌─────────────────────┐
    │   Graph Construction │
    │   Nodes  = accounts  │
    │   Edges  = payments  │
    │   24-hour snapshots  │
    │   Temporal 80/10/10  │
    └─────────┬───────────┘
              │
    ┌─────────┼──────────────────┐
    ▼         ▼                  ▼
FraudGAT   XGBoost          Anomaly Detection
3-layer    SMOTE +          Autoencoder +
GATConv    scale_pos_weight  Isolation Forest
Focal Loss  15 features      (unsupervised)
    │         │                  │
    └────┬────┴──────────────────┘
         ▼
  Ensemble Scorer
         │
         ├──► SHAP Explainability
         │    (gradient×input saliency for GAT,
         │     TreeExplainer for XGBoost)
         │
         └──► SAR Generator (GPT-4o-mini)
              EU AMLD6-compliant JSON reports
         │
         ▼
  Streamlit Dashboard
  (live monitoring, graph view,
   what-if simulator, SAR UI)
```

### FraudGAT

An edge-level Graph Attention Network where accounts are nodes and transactions are edges. For each transaction, the model asks: *"given everything the graph knows about the sender account, the receiver account, and the transaction itself — how suspicious is this payment?"*

- **3 GATConv layers** with 4 attention heads each, residual connections, BatchNorm
- **Edge-level MLP** decoder: `[src_emb ‖ dst_emb ‖ edge_features] → fraud logit`
- **Focal Loss** (α=0.95, γ=3.0) — upweights fraud ~19× to combat the 0.13% class imbalance
- **Chunked decoding** (300K edges/batch) to fit within 16 GB VRAM on large snapshots
- **Strict temporal splits** — 30 train / 4 val / 29 test daily snapshots, never shuffled

### XGBoost Baseline

Gradient boosting on 15 hand-engineered tabular features including sender/receiver balance drops, transaction velocity over 1h and 24h windows, amount z-scores, and temporal flags.

- **SMOTE** (10% sampling strategy) applied to training data only
- **scale_pos_weight** further adjusts for class imbalance
- **Threshold calibrated** at 80% recall on the validation set
- Achieves PR-AUC 0.797 — demonstrating that discriminative velocity and balance features are the strongest fraud signal in PaySim

### Anomaly Detection

Both models are trained **exclusively on legitimate transactions** — no fraud labels needed.

- **Autoencoder**: 3-layer encoder/decoder, anomaly score = reconstruction error. Achieves PR-AUC 0.436, making it useful for detecting novel fraud patterns without supervision.
- **Isolation Forest**: score = path length in random forest. Used as a fast unsupervised baseline.

### SAR Generator

Calls GPT-4o-mini with transaction metadata, SHAP feature contributions, and multi-model fraud scores to generate structured Suspicious Activity Reports compliant with **EU AMLD6** (Anti-Money Laundering Directive 6). Output includes narrative, regulatory flags, evidence summary, and recommended action (FREEZE / ESCALATE / INVESTIGATE / MONITOR).

---

## Dashboard

An interactive Streamlit application with 5 tabs:

| Tab | What it shows |
|---|---|
| **Alert Queue** | Live alert feed, KPI cards (fraud rate, total alerts, PR-AUC), model performance comparison chart, score distributions |
| **Alert Detail** | Per-transaction SHAP waterfall, global feature importance, transaction risk card |
| **Graph View** | Interactive PyVis subgraph around a selected transaction — sender, receiver, and neighbouring accounts |
| **SAR Generator** | One-click EU AMLD6 SAR generation for any high-risk transaction |
| **Scenario Simulator** | What-if tool — adjust any feature and watch the fraud probability update live, with a sensitivity analysis chart |

```bash
streamlit run src/dashboard/app.py --server.port 8501
```

---

## Dataset

| Dataset | Source | Transactions | Fraud Rate |
|---|---|---|---|
| PaySim | Kaggle (synthetic mobile money) | 6,362,620 | 0.13% overall / 0.59% in test period |

Download `paysim.csv` from [Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1) and place it at `data/raw/paysim.csv`.

---

## Setup

```bash
# Clone
git clone https://github.com/mnansari370/Financial-Risk-Intelligence-Fraud-Detection-Platform.git
cd Financial-Risk-Intelligence-Fraud-Detection-Platform

# Create conda environment
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection

# Install PyTorch (CUDA 12.1)
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch_geometric==2.5.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# Install remaining dependencies
pip install -r requirements.txt

# Set up environment
bash setup_env.sh

# Optional: add OpenAI key for SAR generation
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## Running the Pipeline

All GPU-heavy jobs are submitted via SLURM. Interactive steps run in seconds/minutes on a CPU node.

```bash
# 1. Build transaction graphs (~4 hours, CPU)
sbatch slurm/build_graph.sh

# 2. Train XGBoost (~5 minutes, interactive)
sbatch slurm/train_xgboost.sh

# 3. Train Autoencoder (~30 minutes, GPU)
sbatch slurm/train_autoencoder.sh

# 4. Train FraudGAT (~8 hours, GPU)
sbatch slurm/train_gat.sh

# 5. Evaluate all models and build ensemble
sbatch slurm/full_eval.sh

# 6. Compute SHAP explanations (~1 hour, GPU)
sbatch slurm/shap_batch.sh

# 7. Launch dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

---

## Project Structure

```
.
├── configs/
│   ├── gat_config.yaml          # FraudGAT hyperparameters + training config
│   ├── xgboost_config.yaml      # XGBoost parameters + feature list
│   └── anomaly_config.yaml      # Autoencoder + Isolation Forest config
├── data/
│   ├── raw/                     # Raw CSVs — gitignored, download separately
│   └── processed/               # Built graph (.pt) + tabular features — gitignored
├── results/
│   ├── evaluation/              # Per-model metrics CSVs + score parquets
│   │   ├── gat/
│   │   ├── xgboost/
│   │   ├── anomaly/
│   │   └── ensemble/            # Overlap + unique-catch analysis
│   └── shap/                    # SHAP values + feature importance
│       ├── xgboost/
│       └── gat/
├── slurm/                       # SLURM job scripts for ULHPC cluster
├── src/
│   ├── augmentation/            # SMOTE + scale_pos_weight helpers
│   ├── dashboard/               # Streamlit app (app.py)
│   ├── evaluation/              # Metrics, threshold calibration, complementarity
│   ├── explainability/          # SHAP (XGBoost) + gradient saliency (GAT)
│   ├── graph/                   # PyG HomoData graph builder, feature engineering
│   ├── llm/                     # SAR generator (OpenAI / local LLM)
│   └── models/
│       ├── anomaly/             # Autoencoder + Isolation Forest
│       ├── baseline/            # XGBoost pipeline
│       └── gnn/                 # FraudEdgeGAT model + training loop
├── requirements.txt
└── setup_env.sh
```

---

## Key Design Decisions

**PR-AUC as the primary metric** — With a 0.59% test fraud rate, a model that flags nothing achieves 99.41% accuracy. ROC-AUC is similarly misleading at this imbalance. PR-AUC directly measures how well the model ranks fraud above legitimate transactions, and a random classifier achieves only 0.006. All model comparisons use PR-AUC.

**Strict temporal splitting** — Financial transactions are time-series data. Random shuffling before splitting leaks future account behaviour into training, artificially inflating performance. Every split in this project is by timestamp: the first 80% of time steps go to training, next 10% to validation, last 10% to test. This is the only evaluation protocol that reflects real deployment conditions.

**Focal Loss for extreme imbalance** — Standard binary cross-entropy is dominated by the 99.87% legitimate class. Focal Loss (Lin et al., 2017) down-weights easy negatives via `(1 - p_t)^γ`, forcing the gradient to concentrate on hard and rare fraud examples. α=0.95 gives ~19× more weight to fraud samples.

**Edge-level rather than node-level classification** — Fraud is a property of individual transactions, not accounts. Framing the problem as edge classification (with node embeddings informing each edge) is a more natural fit than attempting to classify accounts as fraudulent, since legitimate accounts occasionally make one suspicious transaction.

**Chunked edge decoding** — The edge MLP input `[src_emb ‖ dst_emb ‖ edge_attr]` for 1M+ edges would require allocating a ~2 GB tensor in a single operation, exhausting 16 GB VRAM. Processing 300K edges per chunk reduces peak memory to ~600 MB per chunk with no accuracy impact.

---

## Technologies

- **PyTorch 2.2** + **PyTorch Geometric 2.5** — GATConv, temporal graph snapshots
- **XGBoost 2.0** + **imbalanced-learn** — SMOTE, gradient boosting
- **SHAP 0.45** — TreeExplainer (XGBoost), gradient saliency (GAT)
- **Streamlit 1.35** + **Plotly 5.22** + **PyVis** — interactive dashboard
- **OpenAI API** (GPT-4o-mini) — SAR narrative generation
- **ULHPC cluster** — SLURM job scheduling, Tesla V100 (16/32 GB) GPUs
- **NetworkX** — graph construction and analysis

---

## Author

**Mo Nafees**  
MSc Computer Science — University of Luxembourg  
[LinkedIn](https://linkedin.com/in/mo-nafees) · [GitHub](https://github.com/mnansari370)
