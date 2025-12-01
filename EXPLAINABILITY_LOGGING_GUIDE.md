# Explainability Logging Guide for Fraud Detection GNN Thesis

## Overview

This document explains how the new **structured explainability logging system** works and how to use it for your thesis narrative.

The system captures **two levels of interpretability**:
1. **Graph-Level Explanations** (Structural): Which edges/relationships matter
2. **Feature-Level Explanations** (Financial): Which features matter

---

## Quick Start

### Running the Script with Logging

Simply run the modified `preprocessing_and_training.py`:

```bash
python d:\Thesis\preprocessing_and_training.py
```

The script will automatically:
- Initialize the logger at startup
- Log model metadata during training
- Capture graph-level explanations (from SimpleGNNExplainer)
- Capture feature-level explanations (from SHAP)
- Generate structured output files and a summary report

### Output Directory Structure

After running, you'll find all explanations in:
```
D:\Thesis\explanations\
└── fraud_detection_gnn_YYYYMMDD_HHMMSS/
    ├── model_metadata.json                 # Model config & hyperparams
    ├── EXPLAINABILITY_REPORT.md            # Main summary report (for thesis!)
    ├── graph_explanations_summary.csv      # Aggregated graph explanations
    ├── feature_explanations_summary.csv    # Aggregated feature explanations
    ├── graph_explanations/
    │   ├── node_123_explanation.json       # Individual graph explanations
    │   ├── node_456_explanation.json
    │   └── ...
    └── feature_explanations/
        ├── txn_789_explanation.json        # Individual feature explanations
        ├── txn_1011_explanation.json
        └── ...
```

---

## Output Files Explained

### 1. `model_metadata.json`
Contains all model configuration used during training:
- Model architecture (Hidden Dim, Layers)
- Training hyperparameters (Epochs, LR, Batch size)
- Data configuration (num companies, transactions, loans)
- Loss function and class weights
- Device used (CPU/CUDA)

**Use in thesis:** Include in appendix to show reproducibility.

### 2. `EXPLAINABILITY_REPORT.md`
**The main document for your thesis!** Contains:
- Model configuration summary
- Graph-level explanations with examples
- Feature-level explanations with examples
- Thesis integration guidance

**Use in thesis:** 
- Copy into thesis Chapter on Model Interpretability
- Use examples as figures/tables
- Reference the methodology

### 3. `graph_explanations_summary.csv`
Quick reference of all graph explanations:

| timestamp | node_id | label | num_edges | top_edge_importance | mean_importance |
|-----------|---------|-------|-----------|---------------------|-----------------|
| 2025-12-01T10:30:45 | 283 | fraud | 15000 | 0.9534 | 0.0234 |
| ... | ... | ... | ... | ... | ... |

**Use in thesis:** Create plots showing distribution of edge importance scores.

### 4. `feature_explanations_summary.csv`
Quick reference of all feature explanations:

| timestamp | transaction_id | label | fraud_pushing_count | fraud_reducing_count | max_fraud_push_shap |
|-----------|----------------|-------|---------------------|----------------------|---------------------|
| 2025-12-01T10:30:45 | 789 | fraud | 5 | 3 | 0.4521 |
| ... | ... | ... | ... | ... | ... |

**Use in thesis:** Analyze patterns (which features most frequently trigger fraud flags).

### 5. Individual JSON Explanations
Each explanation is saved as a detailed JSON file:

**`graph_explanations/node_283_explanation.json`** example:
```json
{
  "timestamp": "2025-12-01T10:30:45.123456",
  "node_id": 283,
  "label": "fraud",
  "num_edges": 15000,
  "top_k": 20,
  "top_edges": [
    {
      "src": 1204,
      "dst": 283,
      "importance": 0.9534
    },
    {
      "src": 5021,
      "dst": 283,
      "importance": 0.8764
    }
  ],
  "max_importance": 0.9534,
  "min_importance": 0.0001,
  "mean_importance": 0.0234
}
```

**Use in thesis:** Extract top edges and describe transaction chains.

**`feature_explanations/txn_789_explanation.json`** example:
```json
{
  "timestamp": "2025-12-01T10:30:45.123456",
  "transaction_id": 789,
  "label": "fraud",
  "base_value": 0.12,
  "num_features": 47,
  "top_k": 10,
  "fraud_pushing_features": [
    {
      "name": "spending_deviation_score",
      "shap_value": 0.4521,
      "abs_shap": 0.4521,
      "feature_value": 8.5,
      "direction": "increases_fraud"
    }
  ],
  "fraud_reducing_features": [
    {
      "name": "account_age_months",
      "shap_value": -0.2134,
      "abs_shap": 0.2134,
      "feature_value": 120.0,
      "direction": "decreases_fraud"
    }
  ]
}
```

**Use in thesis:** Create feature importance tables and discuss model behavior.

---

## How to Customize Logging

### Change Output Directory
Edit the configuration in `preprocessing_and_training.py`:

```python
EXPLANATION_OUTPUT_DIR = r"D:\Thesis\explanations"  # Change this path
EXPERIMENT_NAME = "fraud_detection_gnn"              # Change this name
```

### Add More Explanations
The logger can be called from anywhere in your code:

```python
# Log a graph explanation
explainability_logger.log_graph_explanation(
    node_id=node_idx,
    edge_mask=edge_importance_scores,
    edge_index=edge_tuples,
    label="fraud",
    top_k=20
)

# Log a feature explanation
explainability_logger.log_feature_explanation(
    transaction_id=txn_idx,
    shap_values=shap_vals,
    feature_names=feature_list,
    feature_values=feature_vals,
    base_value=base_val,
    label="fraud",
    top_k=10
)

# Save reports at the end
explainability_logger.save_aggregated_csv()
explainability_logger.save_summary_report()
```

---

## Thesis Integration Examples

### Chapter: Model Interpretability & Explainability

**Section 1: Structural Analysis (Graph-Level)**

"Our GNN learns relational patterns by examining edges in the transaction graph. 
For transaction node #283 (flagged as fraudulent), the top-5 important edges were:

| Source → Destination | Edge Importance |
|---------------------|-----------------|
| 1204 → 283 | 0.9534 |
| 5021 → 283 | 0.8764 |
| 2891 → 283 | 0.7245 |
| ...

This shows that the model primarily relied on recent transactions from nodes 1204 and 5021 
to identify node 283 as fraudulent. This validates our hypothesis that fraud often involves 
connected accounts."

**Section 2: Feature Analysis (Feature-Level)**

"Beyond relational patterns, SHAP analysis reveals which financial features drive predictions. 
For transaction #789, the model primarily flagged it due to:

**Fraud-Inducing Features:**
- Spending Deviation Score: 8.5 (SHAP impact: +0.45)
- Transaction Velocity: 12 txns/hour (SHAP impact: +0.38)

**Fraud-Mitigating Features:**
- Account Age: 120 months (SHAP impact: -0.21)

This demonstrates the model respects financial domain knowledge: unusual spending 
patterns and high velocity transactions are flagged as risky, while established 
accounts receive some protection."

**Section 3: Combined Interpretation**

"The two-level explainability provides complete interpretability:
- Graph-level explains *the context* (who is connected, transaction chains)
- Feature-level explains *the characteristics* (amounts, patterns, anomalies)
- Together they justify every fraud prediction."

---

## Python API Reference

### ExplainabilityLogger Class

```python
from explainability_logger import ExplainabilityLogger

# Initialize logger
logger = ExplainabilityLogger(
    output_dir="./explanations",
    experiment_name="my_run"
)

# Log model config
logger.log_model_metadata({
    "hidden_dim": 16,
    "epochs": 8,
    "device": "cpu",
    ...
})

# Log graph explanation
logger.log_graph_explanation(
    node_id=int,
    edge_mask=np.ndarray,
    edge_index=Tuple[List, List],
    label="fraud",
    top_k=20
)

# Log feature explanation
logger.log_feature_explanation(
    transaction_id=int,
    shap_values=np.ndarray,
    feature_names=List[str],
    feature_values=np.ndarray,
    base_value=float,
    label="fraud",
    top_k=10
)

# Save outputs
logger.save_aggregated_csv()
logger.save_summary_report()

# Get statistics
stats = logger.get_statistics()
```

---

## Tips for Thesis Writing

1. **Use the EXPLAINABILITY_REPORT.md**: It's auto-generated and includes examples.
2. **Extract high-quality examples**: Pick 3-5 interesting cases (frauds with clear patterns).
3. **Create visualizations**: Convert CSVs to plots showing importance distributions.
4. **Validate domain knowledge**: Check if model learns expected features (spending, velocity, age, etc.).
5. **Compare graph vs feature**: Show how both levels complement each other.
6. **Archive results**: Save the full explanations folder as appendix or supplementary material.

---

## Troubleshooting

**Q: No explanations saved?**
A: Check if `explainability_logger` is initialized (should print "✓ ExplainabilityLogger initialized").
   Make sure `EXPLANATION_OUTPUT_DIR` exists or is writable.

**Q: Graph explanations empty?**
A: Need fraudulent transactions in test set. Check that `fraud_indices` is non-empty.
   May need more training epochs for model to identify fraud.

**Q: Feature explanations missing?**
A: SHAP requires `pip install shap`. Check console output for SHAP status.
   For quick runs, reduce `QUICK_SHAP_SAMPLES` to speed up computation.

**Q: Report looks incomplete?**
A: Run the full script (not just training) to ensure explainability phase executes.
   Check for exceptions in console output.

---

## Next Steps

1. Run the training script (with logging enabled)
2. Navigate to `D:\Thesis\explanations\<experiment_dir>`
3. Review `EXPLAINABILITY_REPORT.md`
4. Extract examples for thesis chapters
5. Use CSVs to generate additional plots/tables
6. Cite the methodology in your work

Good luck with your thesis! 🚀
