# Quick Reference: Explainability Logging System

## TL;DR (30 seconds)

**What:** Your fraud detection model now has structured logging that captures two types of explanations:
1. **Graph-Level** - Which transaction edges matter (relational/structural)
2. **Feature-Level** - Which financial features matter (domain/financial)

**Where:** All outputs go to `D:\Thesis\explanations\`

**How:** Just run your script - logging happens automatically!

---

## Files You Need to Know

| File | Purpose | Use Case |
|------|---------|----------|
| `explainability_logger.py` | Core logging module | Keep in `D:\Thesis\` |
| `preprocessing_and_training.py` | Updated training script | Already integrated with logging |
| `EXPLAINABILITY_LOGGING_GUIDE.md` | Full user guide | Read for detailed instructions |
| `EXPLAINABILITY_IMPLEMENTATION_SUMMARY.md` | Technical summary | Reference for design decisions |

---

## One-Minute Setup

1. **Already done!** Logger integrated into `preprocessing_and_training.py`
2. **Already done!** `explainability_logger.py` created in your thesis folder
3. **Just run:**
   ```bash
   python "d:\Thesis\preprocessing_and_training.py"
   ```
4. **Check outputs:**
   ```
   D:\Thesis\explanations\fraud_detection_gnn_YYYYMMDD_HHMMSS\
   ```

---

## Output Files at a Glance

After running the script, you'll get:

```
explanations/
└── fraud_detection_gnn_20251201_090757/
    ├── model_metadata.json              ← Your training config
    ├── EXPLAINABILITY_REPORT.md         ← **READ THIS FOR THESIS**
    ├── graph_explanations_summary.csv   ← Graph-level stats
    ├── feature_explanations_summary.csv ← Feature-level stats
    ├── graph_explanations/
    │   └── node_XXX_explanation.json    ← Individual graph explanations
    └── feature_explanations/
        └── txn_XXX_explanation.json     ← Individual feature explanations
```

---

## What Each Output Type Tells You

### Graph-Level (Structural)
**Question:** "Which transaction connections flagged this as fraud?"  
**Answer:** Top edges with importance scores  
**Example:**
```
Node 283 (Fraud):
  1204 → 283 (importance: 0.95) ← Most important edge
  5021 → 283 (importance: 0.88)
  2891 → 283 (importance: 0.72)
```
**Use in thesis:** "The model identified transaction chains between accounts..."

### Feature-Level (Financial)
**Question:** "Which transaction characteristics flagged this as fraud?"  
**Answer:** Features with SHAP values (positive = fraud-pushing, negative = fraud-reducing)  
**Example:**
```
Transaction 789 (Fraud):
Fraud-Pushing:
  - Spending Deviation: 8.5 (SHAP: +0.45) ← Increases fraud likelihood
  - Velocity Score: 12 (SHAP: +0.38)

Fraud-Reducing:
  - Account Age: 120 months (SHAP: -0.21) ← Decreases fraud likelihood
```
**Use in thesis:** "The model respects financial domain knowledge..."

---

## For Your Thesis Narrative

### Chapter: Interpretability & Explainability

**Copy-Paste Template:**

```markdown
### Graph-Level Analysis (Structural)
Our heterogeneous GNN learns relational patterns by analyzing transaction edges. 
[Insert example from graph_explanations_summary.csv]

This demonstrates that [your interpretation of the patterns].

### Feature-Level Analysis (Financial)
Beyond relational context, SHAP analysis reveals the financial characteristics 
driving predictions. [Insert examples from feature_explanations_summary.csv]

This validates that [your domain knowledge interpretation].

### Combined Interpretation
The two-level explainability system provides end-to-end transparency:
- Graph-level explains the relational context
- Feature-level explains domain characteristics
- Together they justify every fraud decision
```

---

## Customization (If Needed)

### Change output directory:
Edit in `preprocessing_and_training.py` line ~48:
```python
EXPLANATION_OUTPUT_DIR = r"D:\Thesis\explanations"  # Change this
EXPERIMENT_NAME = "fraud_detection_gnn"               # Change this
```

### Add more explanations manually:
```python
# In your code:
explainability_logger.log_graph_explanation(...)
explainability_logger.log_feature_explanation(...)
explainability_logger.save_summary_report()
```

### Use specific experiment results:
All results are timestamped, so you can run multiple times and compare:
- `fraud_detection_gnn_20251201_090757/`
- `fraud_detection_gnn_20251201_100000/`
- etc.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ExplainabilityLogger not found" | Make sure `explainability_logger.py` is in `D:\Thesis\` |
| No explanations saved | Check console for errors; may need more epochs to find frauds |
| SHAP explanations missing | Run `pip install shap` or skip SHAP explanations for now |
| Report looks incomplete | Ensure script runs to completion (check for errors in console) |

---

## What You Got

✅ Automatic logging of graph-level explanations  
✅ Automatic logging of feature-level explanations  
✅ Organized JSON, CSV, and Markdown outputs  
✅ Ready-to-use report for thesis  
✅ Windows PowerShell compatible  
✅ Integrated into your training pipeline  

**Nothing extra to install or configure** - just run your script!

---

## Next Actions

1. **Run the script:**
   ```bash
   cd d:\Thesis
   python preprocessing_and_training.py
   ```

2. **Read the report:**
   Open `D:\Thesis\explanations\fraud_detection_gnn_YYYYMMDD_HHMMSS\EXPLAINABILITY_REPORT.md`

3. **Extract examples for thesis:**
   - Copy sections from the report
   - Convert CSVs to plots if needed
   - Reference in your interpretability chapter

4. **Done!** You now have complete end-to-end explainability for your fraud detection model.

---

**Questions?** Check `EXPLAINABILITY_LOGGING_GUIDE.md` for detailed reference.
