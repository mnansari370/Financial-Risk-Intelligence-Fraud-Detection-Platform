# Interview Preparation Notes
# Financial Risk Intelligence & Fraud Detection Platform

---

## 1. Project Overview — What to Say in 60 Seconds

"I built a full end-to-end fraud detection platform that combines four different models: a Graph Attention Network, XGBoost, an Autoencoder, and Isolation Forest. The idea was to model bank transactions as a graph — accounts are nodes and payments are edges — so the model can learn from the relationships between accounts, not just individual transaction features. I trained everything on a university HPC cluster using SLURM job scheduling, built a complete Streamlit dashboard for monitoring and explainability, and added an automated SAR (Suspicious Activity Report) generator using GPT-4o-mini. The best model, XGBoost, achieved PR-AUC of 0.797, which is 133 times above the random baseline for this dataset."

---

## 2. The Dataset — PaySim

**What it is:**
PaySim is a synthetic financial transaction dataset generated using real mobile money transaction patterns from an African bank. It was published by researchers at NTNU (Norwegian University of Science and Technology).

**Key numbers you must know:**
- 6,362,620 total transactions
- 8,213 fraud transactions (0.13% overall fraud rate)
- 5 transaction types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- Fraud only occurs in TRANSFER and CASH_OUT transactions (important!)
- The dataset simulates 30 days of transactions

**Why synthetic data is acceptable:**
PaySim was built by calibrating a multi-agent simulation against real transaction distributions. Real bank fraud data is never publicly available due to privacy laws and competitive sensitivity. Using PaySim is standard practice in academic fraud detection research.

**The temporal distribution shift problem (critical to understand):**
When you split PaySim by time (which you must do — see Section 6), the fraud is not evenly distributed across time. The last 10% of the timeline contains 0.59% fraud, while the first 80% (training) contains only 0.08% fraud. This is a 7× difference. This is why the GAT model struggled — it trained on a nearly fraud-free signal and was tested on a period with 7× more fraud. This is a real-world condition that happens in production systems when fraud patterns evolve over time.

---

## 3. Graph Construction — How You Built the Graph

**The core design decision:**
You modelled the problem as a **homogeneous graph**:
- **Nodes = accounts** (every unique sender or receiver is a node)
- **Edges = transactions** (every payment is a directed edge from sender to receiver)
- **Node features** = account-level statistics (velocity, volume, risk scores)
- **Edge features** = transaction-level features (amount, time, flags)
- **Edge labels** = 1 if fraud, 0 if legitimate

**Why homogeneous (not heterogeneous)?**
A heterogeneous graph would have different node types (e.g. account, merchant, transaction). PaySim only has accounts — there are no merchant entities — so a homogeneous graph is the correct choice. Using heterogeneous would add complexity with no benefit.

**Why edges = transactions (not nodes = transactions)?**
Fraud is fundamentally a property of a transaction, not an account. A legitimate account can make one fraudulent transfer. If you make transactions nodes, you lose the natural sender→receiver relationship. Making transactions edges means the model naturally sees "this account sent money to that account" as the basic unit of analysis.

**Temporal snapshots:**
The full timeline is divided into 24-hour windows. Each snapshot is an independent graph representing all transactions in that day. This is called a **temporal graph** approach. You end up with roughly 63 snapshots (30 days × some structure), split 80/10/10:
- Train: first ~50 snapshots
- Val: next ~6 snapshots
- Test: last ~7 snapshots

**Feature engineering (15 features for XGBoost, subset for GAT):**
- `amount`, `amount_log` (log1p transform for skewed distribution), `amount_zscore`
- `velocity_1h_sender`, `velocity_24h_sender` — how many transactions the sender made
- `velocity_1h_receiver`, `velocity_24h_receiver` — how many transactions the receiver got
- `sender_balance_drop` — how much the sender's balance decreased (strong signal)
- `receiver_balance_gain` — how much receiver's balance increased
- `hour`, `day_of_week`, `is_night` — temporal patterns
- `is_round_amount` — fraudsters often transact in round numbers
- `is_first_pair_tx` — first time this sender-receiver pair has transacted
- `is_flagged_fraud` — PaySim's own flag (weak, often wrong, but informative)

**Why log-transform the amount?**
Transaction amounts are extremely right-skewed — most are small, a few are enormous. Taking log1p compresses the range so the model doesn't treat $100,000 as literally 1000× more important than $100. This is standard preprocessing for monetary features.

**Why z-score the amount?**
The z-score (`(x - mean) / std`) tells you how unusual the amount is relative to the typical transaction in that snapshot. A $50,000 transaction in a snapshot where the average is $500 has a high z-score and is a fraud signal.

---

## 4. FraudGAT — The Graph Attention Network

**The architecture in full:**

```
Input: node features [N_nodes × node_dim]
       edge index [2 × N_edges]
       edge features [N_edges × edge_dim]

Step 1 — Node Projection:
  Linear(node_dim → 64) + ReLU
  → h: [N_nodes × 64]

Step 2 — 3× GATConv layers:
  Layer 1: GATConv(64 → 64, heads=4, concat=True)
           BatchNorm(256) + ReLU + Dropout(0.3)
           → h: [N_nodes × 256]
  
  Layer 2: GATConv(256 → 64, heads=4, concat=True)
           BatchNorm(256) + ReLU + Dropout(0.3)
           Residual: h = h + h_new (shapes match)
           → h: [N_nodes × 256]
  
  Layer 3: GATConv(256 → 64, heads=4, concat=True)
           BatchNorm(256) + ReLU + Dropout(0.3)
           Residual: h = h + h_new
           → h: [N_nodes × 256]

Step 3 — Edge Decoder (MLP):
  For each edge (src, dst):
    input = [h[src] ‖ h[dst] ‖ edge_features]  → [512 + edge_dim] dims
    Linear → 32 → ReLU → Dropout(0.3) → Linear → 1 logit
  → logits: [N_edges]

Step 4 — Loss:
  Focal Loss(logits, labels, α=0.95, γ=3.0)
```

**What GATConv actually does:**
Graph Attention Networks (Veličković et al., 2018) allow each node to attend over its neighbours with learned attention weights. Instead of treating all neighbours equally (like a simple graph convolution), GAT learns *which neighbours matter more*. For a sender account, some receiver accounts are more relevant than others.

The attention weight between node i and neighbour j is:
```
α_ij = softmax( LeakyReLU( a^T [Wh_i ‖ Wh_j] ) )
```
Then the new node representation is: `h_i' = σ( Σ_j α_ij · W h_j )`

With 4 heads (multi-head attention), this runs 4 independent attention mechanisms in parallel and concatenates the results, giving richer representations.

**What does a GATConv layer do for fraud detection?**
After 3 layers, each account's embedding contains information not just from its own features, but from accounts up to 3 hops away. A money mule's account will be influenced by the fraudulent sender it received money from, even if the mule's own features look innocent.

**Residual connections:**
After layer 1, the output shape is 256 (64 × 4 heads). Layers 2 and 3 also output 256, so we add the input to the output: `h = h + h_new`. This prevents gradient vanishing in deep networks and lets earlier layers' information flow through unchanged.

**Why Focal Loss?**
Standard binary cross-entropy at 0.13% fraud rate: the model sees 769 legitimate transactions for every 1 fraud. The gradient is dominated by easy "this is clearly not fraud" examples. Focal Loss adds a weight `(1 - p_t)^γ` to each sample:
- Well-classified samples (p_t close to 1) get small weight → model ignores them
- Hard or misclassified samples (p_t close to 0) get large weight → model focuses on them

α=0.95 means: fraud samples (positive class) get weight 0.95, legitimate samples get weight 0.05. At 0.13% fraud rate with 769:1 ratio, this gives roughly 19× more attention to fraud.

**Why γ=3.0 (not the default 2.0)?**
Higher γ means more aggressive down-weighting of easy examples. With an extreme imbalance like PaySim, γ=3.0 forces the model to concentrate harder on the rare fraud examples.

**The CUDA Out-of-Memory problem and how you fixed it:**
During training at epoch 2, the training crashed with:
```
torch.cuda.OutOfMemoryError: Tried to allocate 1.11 GiB. 14.84 GiB already in use.
```
The cause: in `decode_edges`, this line:
```python
edge_input = torch.cat([node_emb[src], node_emb[dst], edge_attr], dim=-1)
```
For a snapshot with 1M edges and 519 total dimensions: `1,000,000 × 519 × 4 bytes = ~2 GB` for the tensor alone, plus the gradient doubles it to ~4 GB. On a V100 with 16 GB, when combined with the node embeddings and optimizer state, this caused OOM.

The fix: process edges in chunks of 300,000:
```python
def decode_edges(self, node_emb, edge_index, edge_attr, chunk_size=300_000):
    src, dst = edge_index
    if src.shape[0] <= chunk_size:
        # small snapshot — process at once
        edge_input = torch.cat([node_emb[src], node_emb[dst], edge_attr], dim=-1)
        return self.edge_mlp(edge_input).squeeze(-1)
    
    chunks = []
    for i in range(0, src.shape[0], chunk_size):
        s = src[i:i+chunk_size]
        d = dst[i:i+chunk_size]
        a = edge_attr[i:i+chunk_size]
        chunks.append(self.edge_mlp(
            torch.cat([node_emb[s], node_emb[d], a], dim=-1)
        ).squeeze(-1))
    return torch.cat(chunks)
```
Each chunk is ~600 MB. The `torch.cat` at the end re-joins the logits before the loss is computed, so gradients flow correctly.

**Training configuration:**
- Optimizer: Adam, lr=3e-4, weight_decay=1e-4
- Scheduler: cosine annealing (lr decays smoothly to near-zero)
- Early stopping: patience=20 epochs (stops if val PR-AUC doesn't improve for 20 epochs)
- Gradient clipping: max norm=1.0 (prevents gradient explosion)
- Best checkpoint saved based on validation PR-AUC

---

## 5. XGBoost — Why It Won

**What XGBoost is:**
XGBoost (Extreme Gradient Boosting) builds an ensemble of decision trees sequentially. Each tree corrects the errors of the previous trees. It is extremely effective on tabular data with well-engineered features.

**The pipeline:**
1. Load `features_tabular.parquet` (pre-built from the graph construction step)
2. Split by time: 80% train, 10% val, 10% test (same split as GAT)
3. Apply SMOTE on training data only (never on val/test — that would be data leakage)
4. Set `scale_pos_weight = n_negative / n_positive` (~769 for PaySim)
5. Train XGBoost with config from `configs/xgboost_config.yaml`
6. Calibrate threshold on validation set at 80% recall
7. Evaluate on test set, save `results/evaluation/xgboost/scores.parquet`

**SMOTE — what it is and why:**
SMOTE (Synthetic Minority Over-sampling TEchnique) generates synthetic fraud examples by interpolating between real fraud examples in feature space. If two fraud transactions have feature vectors `A` and `B`, SMOTE creates new examples along the line `A + λ(B-A)` for random λ ∈ [0,1]. This increases the fraud class from ~0.13% to 10% of the training set (sampling_strategy=0.1), giving the model more fraud examples to learn from.

**scale_pos_weight:**
An XGBoost parameter that multiplies the gradient of positive (fraud) samples by this weight. Setting it to ~769 (ratio of negatives to positives) is equivalent to telling XGBoost "treat each fraud sample as if it were 769 samples."

**Why use both SMOTE and scale_pos_weight?**
They address different parts of the imbalance problem. SMOTE changes the data distribution before training, giving the model more fraud examples. scale_pos_weight changes the loss function during training. Together they're more effective than either alone.

**Why XGBoost outperforms GAT on this dataset:**
The features `sender_balance_drop` and `receiver_balance_gain` are enormously predictive for PaySim fraud. Fraudulent TRANSFER transactions almost always involve the sender's balance dropping to near-zero (they're draining an account). These are simple tabular features that XGBoost can pick up immediately. The graph structure in PaySim is relatively simple — it doesn't have the complex ring structures or money mule chains that would give a GNN an advantage over tabular models.

**Top SHAP features (from your results):**
1. `sender_balance_drop` — mean |SHAP| 6.97 (by far the strongest signal)
2. `amount_log` — 1.24
3. `hour` — 1.10
4. `receiver_balance_gain` — 0.90
5. `day_of_week` — 0.69

This tells a clear story: fraudulent transactions drain sender accounts (balance drop), happen at specific times (hour, day_of_week), and involve large amounts (amount_log).

---

## 6. Why Temporal Splitting Is Critical

**The wrong way — random split:**
Randomly shuffle all 6.36M transactions, take 80% for training and 20% for test. This leaks future information: the model is trained on transactions from day 25, then tested on transactions from day 3. In production, you never have future data during training.

**The right way — temporal split:**
Sort by timestamp. First 80% of time steps → training. Next 10% → validation. Last 10% → test. This perfectly mirrors the deployment scenario: train on historical data, validate on recent data, evaluate on the most recent period.

**Why this matters for your results:**
With a random split, you'd likely see near-perfect results (PR-AUC > 0.95) because the model has already seen similar transactions from every time period during training. The 0.797 PR-AUC from a temporal split is a more honest and realistic number.

---

## 7. Anomaly Detection Models

**Why use unsupervised models at all?**
In real fraud detection, labels are expensive and delayed. A transaction might not be confirmed as fraud for days or weeks (chargebacks take time). Unsupervised models can flag suspicious patterns without needing any labels — useful for detecting new fraud types that haven't been labelled yet.

**Autoencoder:**
A neural network with an encoder that compresses the input into a small latent representation, and a decoder that reconstructs the original input. Trained only on legitimate transactions. At inference time, legitimate transactions reconstruct well (low error). Fraudulent transactions, being out-of-distribution, reconstruct poorly (high error). The reconstruction error becomes the anomaly score.

Architecture: 15 features → [64, 32, 16] encoder → [32, 64] decoder → 15 features
Loss: Mean Squared Error (reconstruction error)
Fraud score = MSE between input and reconstruction

Result: PR-AUC 0.436 — reasonable for an unsupervised model that never sees a fraud label.

**Isolation Forest:**
Builds random decision trees by randomly selecting features and random split values. Anomalies (fraud) are easier to isolate — they require fewer splits because they're rare and different from normal points. The anomaly score is the average path length to isolate a sample (shorter path = more anomalous).

Result: PR-AUC 0.220 — weaker than the autoencoder but still 37× above random.

**Why the anomaly models have very low F1 scores:**
F1 requires choosing a threshold. At the default threshold of 0.5 for reconstruction error, the anomaly models either flag everything or nothing, giving extreme precision or recall. The PR-AUC (which measures the full precision-recall tradeoff across all thresholds) is more informative.

---

## 8. Evaluation Methodology

**Why PR-AUC is the primary metric:**
- Fraud rate = 0.59% in test set
- A model that flags nothing: 99.41% accuracy, 0.0 PR-AUC
- ROC-AUC can be misleading — at high imbalance, even weak models get high ROC-AUC because there are so many true negatives
- PR-AUC measures precision and recall across all thresholds, directly reflecting real-world usefulness
- **Random baseline PR-AUC = fraud rate ≈ 0.006**

**Precision @ 80% Recall:**
This answers the question: "if we want to catch 80% of all fraud, how many false alarms do we have to deal with per real fraud?" XGBoost's answer is 48.3% precision at 80% recall — meaning for every 2 alerts, ~1 is real fraud. This is operationally very useful.

**FPR (False Positive Rate):**
FPR = false positives / total negatives. In production, this is the proportion of legitimate transactions that get flagged as suspicious. XGBoost's FPR of 0.58% means 58 out of every 10,000 legitimate transactions are wrongly flagged. The full ensemble reduces this to 0.04% — extremely important for customer experience.

**Complementarity analysis:**
You checked whether the models catch different frauds (Jaccard overlap and unique catches). Result: at matched top-K threshold, XGBoost caught all 24 unique fraud cases, GAT caught 0 unique cases. The full ensemble improves F1 by 17% mainly by reducing false positives, not by catching more fraud.

---

## 9. SHAP Explainability

**What SHAP is:**
SHAP (SHapley Additive exPlanations) is a method from cooperative game theory that assigns each feature a contribution to the model's prediction. The SHAP value for feature i tells you: "how much did this feature push the prediction up or down compared to the average prediction?"

For a transaction with fraud score 0.85:
- `sender_balance_drop = +3.2` → this feature pushed the score up by 3.2 (toward fraud)
- `hour = -0.8` → this feature pushed the score down by 0.8 (toward legitimate)
- Sum of all SHAP values + base rate = raw prediction score

**XGBoost SHAP — TreeExplainer:**
SHAP provides a TreeExplainer specifically for tree-based models. It computes exact Shapley values in O(TLD²) time (T = trees, L = leaves, D = depth) — much faster than model-agnostic methods. You compute SHAP values for the top 1,000 highest-scored transactions and save them to `results/shap/xgboost/shap_values_xgb.parquet`.

**GAT SHAP — Gradient × Input saliency:**
SHAP's GradientExplainer requires a fixed-size input. The GAT takes a graph as input — variable number of nodes and edges per snapshot. So instead, you use gradient × input saliency:
```
saliency_i = |gradient of loss w.r.t. edge_feature_i| × |edge_feature_i|
```
This is a standard attribution method for neural networks. It tells you which edge features (amount, time of day, etc.) the model paid most attention to for a given prediction.

---

## 10. The SAR Generator

**What a SAR is:**
A Suspicious Activity Report is a formal document that financial institutions are legally required to file with regulators (FINTRAC in Canada, FinCEN in the US, national FIUs in the EU) when they detect or suspect money laundering or fraud. EU AMLD6 (Anti-Money Laundering Directive 6) specifies what must be included.

**How yours works:**
1. The dashboard user selects a high-risk transaction
2. The system calls GPT-4o-mini with a structured prompt containing:
   - Transaction ID, amount, timestamp
   - Multi-model fraud scores (XGBoost score, GAT score)
   - SHAP feature contributions (what drove the prediction)
   - Model confidence and risk level
3. GPT-4o-mini returns a structured JSON with:
   - `narrative`: Human-readable description of why this is suspicious
   - `regulatory_flags`: List of specific AML red flags triggered
   - `evidence_summary`: Key evidence points
   - `risk_level`: CRITICAL / HIGH / MEDIUM / LOW
   - `recommended_action`: FREEZE / ESCALATE / INVESTIGATE / MONITOR

**Why GPT-4o-mini (not GPT-4o)?**
Cost efficiency. SAR generation is a text generation task that doesn't require frontier-model reasoning. GPT-4o-mini is ~10× cheaper and handles structured JSON output reliably.

---

## 11. The Dashboard — 5 Tabs Explained

**Tab 1 — Alert Queue:**
The main operational view. Shows 4 KPI cards (total transactions, fraud rate, number of active alerts, best model PR-AUC). Then a grouped bar chart comparing all models on PR-AUC, ROC-AUC, and F1. Then the live transaction alert feed filtered by a threshold you set in the sidebar. Finally, score distribution histograms showing fraud vs legitimate score separation.

**Tab 2 — Alert Detail:**
Analyst deep-dive view. Select any transaction from the top-200 highest-scored. See a transaction info card (ID, score, risk level, ground truth). Then SHAP waterfall chart showing exactly which features drove the prediction for that specific transaction. Plus global feature importance (average across all top-1,000 alerts).

**Tab 3 — Graph View:**
Interactive PyVis graph showing the selected transaction as a node, with its sender and receiver accounts, plus up to 10 neighbouring transactions in the same snapshot. Node colour shows risk level. Hover tooltips show fraud scores. Helps analysts understand the network context.

**Tab 4 — SAR Generator:**
Select a high-risk transaction and click Generate. Calls GPT-4o-mini and displays the full AMLD6-compliant report with colour-coded regulatory flags and recommended action.

**Tab 5 — Scenario Simulator:**
What-if analysis. Adjust transaction features (amount, hour, velocity, flags) and the XGBoost model scores it in real time. A gauge chart shows the fraud probability. Below, a sensitivity chart shows how the probability changes as you vary one feature across its full range — useful for understanding model behaviour.

---

## 12. HPC and SLURM

**What ULHPC is:**
University of Luxembourg High Performance Computing. You used the GPU partition with Tesla V100 GPUs (16 GB and 32 GB variants).

**SLURM:**
SLURM (Simple Linux Utility for Resource Management) is a cluster job scheduler. Instead of running code directly, you write a shell script with resource requests (`#SBATCH` directives) and submit it with `sbatch`. SLURM queues the job and runs it when a node with the requested resources is available.

**Key job scripts:**
- `build_graph.sh` — CPU node, 4 hours, builds PyG graph from raw CSV
- `train_gat.sh` — 1 GPU (V100), 16 GB RAM, 8 hours, trains FraudGAT
- `train_xgboost.sh` — 1 GPU (for XGBoost GPU tree algorithm), 16 GB RAM
- `train_autoencoder.sh` — 1 GPU, 16 GB RAM, 30 minutes
- `full_eval.sh` — 1 GPU, runs evaluate.py and complementarity.py for all models
- `shap_batch.sh` — 1 GPU, 1 hour, computes SHAP for top-1,000 transactions

**A bug you hit with SLURM (good interview story):**
The XGBoost job was failing with `PYTHONPATH: unbound variable`. This was caused by:
```bash
set -euo pipefail  # -u treats unset variables as errors
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"  # crashes if PYTHONPATH not set
```
Fix: `${PYTHONPATH:-}` — this means "use $PYTHONPATH if set, otherwise use empty string."

---

## 13. Results Summary — Numbers to Know by Heart

| Model | PR-AUC | ROC-AUC | F1 | FPR |
|---|---|---|---|---|
| XGBoost | **0.797** | **0.985** | 0.578 | 0.58% |
| Full Ensemble | 0.739 | 0.984 | **0.679** | **0.04%** |
| Autoencoder | 0.436 | 0.874 | ~0 | 0.0% |
| Isolation Forest | 0.220 | 0.870 | 0.155 | 3.0% |
| FraudGAT | 0.154 | 0.819 | 0.203 | 0.27% |
| Random baseline | 0.006 | — | — | — |

**Key talking points:**
- XGBoost is 133× above random (0.797 / 0.006)
- The ensemble achieves F1 0.679 — 17% better than XGBoost's 0.578, at only 0.04% FPR
- PR-AUC is the right metric — you chose it deliberately, not because it looks better
- GAT's result reflects a real deployment challenge (temporal distribution shift), which shows you understand the limits of your own work

---

## 14. Likely Interview Questions and Answers

**Q: Why did XGBoost outperform your Graph Neural Network?**
A: "Two reasons. First, the graph structure in PaySim is relatively simple — fraudsters drain one account and move the money out. This pattern is captured directly by the `sender_balance_drop` feature, which is the single strongest predictor with a mean SHAP value of 6.97. Graph models add the most value when fraud patterns span multiple hops — like money mule rings — which PaySim doesn't model as richly. Second, the temporal distribution shift hit the GAT hard. Training fraud rate was 0.08%, test fraud rate was 0.59%, a 7× jump. XGBoost with well-engineered features generalises better under distribution shift."

**Q: How did you handle class imbalance?**
A: "Three layers. For XGBoost: SMOTE to oversample training fraud to 10%, plus scale_pos_weight to upweight fraud in the loss function. For FraudGAT: Focal Loss with α=0.95 and γ=3.0, giving ~19× more weight to fraud examples and focusing gradient on hard cases. For all models: I used PR-AUC as the primary metric, which directly measures minority class performance rather than masking it with accuracy or ROC-AUC."

**Q: Why did you use PR-AUC instead of accuracy or ROC-AUC?**
A: "At 0.59% fraud rate, a model that predicts everything as legitimate achieves 99.41% accuracy but catches zero fraud. ROC-AUC is also misleading at high imbalance — because there are so many true negatives, even a weak model achieves high ROC-AUC. PR-AUC measures how well the model ranks fraud above legitimate transactions across all possible thresholds, which is exactly what you care about in an alert system. The random baseline PR-AUC for this test set is just 0.006."

**Q: What is a Graph Attention Network and why use it?**
A: "A GAT is a neural network that operates on graph-structured data. Each node updates its representation by attending over its neighbours with learned attention weights — some neighbours are more relevant than others. The key advantage over tabular models is that GATs can detect fraud patterns that require looking at relationships: a legitimate-looking account that always receives money from compromised senders, or a receiver that immediately forwards money to many other accounts. These ring or chain structures are invisible to per-transaction models. In my case, accounts are nodes and transactions are edges, so the model learns from the full network context around each payment."

**Q: What would you do differently to improve the GAT?**
A: "Several things. First, use a heterogeneous graph if merchant data were available — type-specific message passing would help. Second, train with data augmentation for the temporal shift — techniques like temporal mixup or fine-tuning on the most recent snapshots before the test period. Third, try a more powerful architecture like a Graph Transformer or RGCN. Fourth, consider making the graph dynamic rather than static snapshots — models like TGN (Temporal Graph Networks) explicitly model the evolution of node state over time. Fifth, train longer with better hardware — 200 epochs of early stopping on only 4 validation snapshots isn't a strong validation signal."

**Q: How did you ensure no data leakage?**
A: "Strict temporal splits throughout. Training data never sees any transaction from the validation or test period. SMOTE is applied after splitting, only to training data. Feature z-scores are fitted on the training split and applied to val/test — the mean and standard deviation come from training only. Threshold calibration uses the validation set only."

**Q: Explain your SAR generator — how does the LLM fit in?**
A: "The LLM handles narrative generation, not prediction. The fraud detection models (XGBoost, GAT) produce the fraud scores and SHAP attributions. I pass those structured signals to GPT-4o-mini with a carefully designed prompt that instructs it to write an EU AMLD6-compliant narrative, identify specific regulatory flags, and recommend an action. The LLM adds no fraud detection capability — it translates machine scores into human-readable regulatory language that a compliance analyst can file with financial intelligence units. This is actually how AI is most responsibly used in regulated industries: the model makes the prediction, the LLM writes the report, the human analyst reviews and approves."

**Q: How did you run training at scale?**
A: "I used the ULHPC cluster at the University of Luxembourg. Each training job is a SLURM script that requests specific resources — number of GPUs, memory, wall time — and joins a queue. The GAT training ran on a Tesla V100 with 16 GB VRAM. I hit an out-of-memory crash when building the edge input matrix for million-edge snapshots, which I fixed by processing edges in 300,000-edge chunks. SLURM handles job dependencies, logging, and resource isolation — it's the standard tool for scientific computing at HPC centres."

**Q: What is SHAP and why is it important for fraud detection?**
A: "SHAP (Shapley Additive exPlanations) gives each feature a contribution value for a specific prediction. In fraud detection this matters for three reasons: compliance (regulators require explainable decisions — you can't freeze an account without justification), debugging (SHAP revealed that sender_balance_drop is the dominant feature, which confirms the model is learning real fraud patterns and not spurious correlations), and analyst trust (showing an analyst exactly why a transaction was flagged helps them quickly assess whether to investigate). Without explainability, a black-box score of 0.92 is hard for an analyst to act on. With SHAP, they see 'sender balance dropped by $8,000, amount is 3.2 standard deviations above normal, first transaction to this receiver' — that's actionable."

---

## 15. Technologies and Why You Chose Each One

| Technology | Why |
|---|---|
| **PyTorch + PyTorch Geometric** | Standard framework for GNNs. PyG provides GATConv, temporal graph utilities, and efficient sparse message passing |
| **XGBoost** | Best-in-class for tabular data, GPU-accelerated tree building, built-in handling of class weights, fast inference |
| **SHAP** | Gold standard for ML explainability, TreeExplainer is exact (not approximate) for tree models |
| **Streamlit** | Fastest way to build a production-looking ML dashboard in Python. Handles state, caching, and interactivity |
| **Plotly** | Interactive charts that work in Streamlit with hover tooltips, zoom, etc. |
| **PyVis** | NetworkX-compatible interactive graph visualisation that renders as HTML/JavaScript |
| **imbalanced-learn** | Standard SMOTE implementation, integrates cleanly with scikit-learn pipelines |
| **SLURM on ULHPC** | The only way to access GPU compute for training. Essential for any deep learning project at a university |
| **GPT-4o-mini** | Cost-effective for structured text generation. At 10× lower cost than GPT-4o with comparable quality for templated outputs |

---

## 16. What Makes This Project Portfolio-Ready

1. **Real problem, real dataset** — 6.36M transactions, 0.13% fraud rate, the same scale and imbalance as real banking data

2. **Multiple model paradigms** — GNN, gradient boosting, unsupervised anomaly detection, LLM — shows breadth across ML

3. **Production engineering** — HPC training, SLURM scripts, chunked GPU memory management, YAML configs, modular codebase, parquet files for efficiency

4. **Honest evaluation** — Temporal splits, PR-AUC, and an honest discussion of GAT limitations. Interviewers notice when candidates hide weaknesses; showing you understand *why* GAT underperformed demonstrates real depth

5. **Regulatory awareness** — EU AMLD6 SAR generation shows you understand the compliance environment of financial services, which most ML candidates don't

6. **End-to-end** — From raw CSV to a deployed dashboard with explainability, report generation, and live scenario simulation. This is a complete system, not just a model

7. **Explainability** — SHAP is table stakes in financial ML. Having it wired into a UI that analysts can actually use shows you understand deployment, not just training

---

*Last updated: April 2026*
