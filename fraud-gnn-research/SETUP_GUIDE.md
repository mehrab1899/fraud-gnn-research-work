# Fraud Detection with Heterogeneous Graph Neural Networks - SETUP GUIDE

**A comprehensive thesis project for detecting fraud across multiple datasets using Graph Neural Networks with explainability.**

---

## 📋 Quick Navigation

- **New to this project?** → Start with [Installation & Setup](#installation--setup)
- **Need to download datasets?** → See [Dataset Configuration](#dataset-configuration)
- **Running for the first time?** → Follow [Quick Start](#quick-start-development-mode)
- **Something broken?** → Check [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **Heterogeneous Graph Neural Network (GNN)** for fraud detection that:

- **Integrates multiple datasets**: Financial transactions, credit data, loans, and payment systems
- **Handles class imbalance**: Uses Focal Loss, class weights, and SMOTE
- **Provides explainability**: Two-level interpretability with graph and feature-level explanations
- **Optimizes for efficiency**: Memory-efficient training with AMP (Automatic Mixed Precision)

### Key Features

✅ Multi-dataset heterogeneous graph construction  
✅ Custom attention-based BiLSTM for transaction sequences  
✅ Focal Loss for handling severe class imbalance  
✅ Graph-level explanations via SimpleGNNExplainer  
✅ Feature-level explanations via SHAP  
✅ Quick-run mode for development/testing  
✅ Configuration-driven approach (no hardcoded paths)  
✅ Production-ready with comprehensive logging  

---

## Project Structure

```
fraud_detection_gnn/
├── preprocessing_and_training.py    # Main training script
├── config_loader.py                 # Configuration management
├── explainability_logger.py         # Explanation logging
│
├── config.yaml                      # Configuration (dataset info, settings)
├── .env.example                     # Environment template
├── .env                             # Your local settings (YOU CREATE THIS)
├── requirements.txt                 # Python dependencies
├── SETUP_GUIDE.md                   # This file
│
├── datasets/                        # Create this folder
│   ├── financial_fraud_detection_dataset.csv
│   ├── german_credit_data.csv
│   ├── nz_bank_loans_synthetic_with_dates.csv
│   └── PS_20174392719_1491204439457_log.csv
│
└── explanations/                    # Auto-created, stores results
    └── fraud_detection_gnn/
        ├── aggregated_results.csv
        ├── summary_report.md
        └── visualizations...
```

---

## System Requirements

### Minimum
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 4 GB minimum (8+ GB recommended)
- **Storage**: 500 MB free (datasets + installation)
- **OS**: Windows, Linux, or macOS

### Recommended
- **GPU**: NVIDIA GPU with 4+ GB VRAM
- **RAM**: 16 GB
- **Storage**: SSD with 2+ GB free
- **Python**: 3.10 or 3.11

---

## Installation & Setup

### Step 1: Install Python

If you don't have Python:
1. Download from https://www.python.org/downloads/ (select 3.8+)
2. **Windows users**: Check "Add Python to PATH" during installation

Verify installation:
```bash
python --version
pip --version
```

### Step 2: Set Up Project Directory

```bash
# Create project folder
mkdir fraud_detection_gnn
cd fraud_detection_gnn

# Download/extract the project files into this folder
# You should have:
# - preprocessing_and_training.py
# - config_loader.py
# - explainability_logger.py
# - config.yaml
# - .env.example
# - requirements.txt
# - SETUP_GUIDE.md (this file)
```

### Step 3: Create Virtual Environment (Recommended)

A virtual environment keeps dependencies isolated:

```bash
# Windows
python -m venv fraud_env
fraud_env\Scripts\activate

# Linux/macOS
python3 -m venv fraud_env
source fraud_env/bin/activate
```

You should see `(fraud_env)` at the start of your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you get errors with torch-geometric on Windows:
```bash
pip install torch-geometric --no-build-isolation
```

### Step 5: Set Up Environment Configuration

```bash
# Copy the template
copy .env.example .env          # Windows
cp .env.example .env            # Linux/macOS

# Now edit .env with your editor and set dataset paths
# (See next section: Dataset Configuration)
```

### Step 6: Test Setup

```bash
python -c "from config_loader import Config; Config.load_env(); print('✓ Setup OK')"
```

---

## Dataset Configuration

### Overview

The script uses **4 datasets** (~500 MB total). Each must be downloaded separately.

**Time estimate for full setup**: 15-30 minutes

### Creating datasets/ folder

```bash
# Create folder for datasets
mkdir datasets
```

### Dataset 1: Financial Fraud Detection

**Source**: Kaggle  
**Download size**: ~100 MB  
**Time**: 2-5 minutes

```
1. Go to: https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset
2. Click "Download" button
   (Requires FREE Kaggle account - use Google/email to sign up)
3. Extract the CSV file
4. Rename to: financial_fraud_detection_dataset.csv
5. Place in: ./datasets/
```
---

### Dataset 2: German Credit Data

**Source**: UCI / Kaggle  
**Download size**: ~1 MB  
**Time**: < 1 minute

```
1. Go to: https://www.kaggle.com/datasets/uciml/german-credit
2. Click "Download" 
3. Extract CSV (should be named german_credit_data.csv)
4. Place in: ./datasets/
```

**Verify**: Should have ~1,000 rows

---

### Dataset 3: Credit Risk Loans

**Source**: Synthetic/Kaggle  
**Download size**: ~50 MB  
**Time**: 2-3 minutes

**Choose one option:**

**Option A** (Easier):
```
1. Go to: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
2. Download and extract
3. Rename to: nz_bank_loans_synthetic_with_dates.csv
4. Place in: ./datasets/
```

---

### Dataset 4: Payment System Transactions

**⚠️ LARGEST FILE - ~200 MB** (take your time with this one)

**Source**: Kaggle  
**Download size**: 200+ MB  
**Time**: 5-10 minutes

```
1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1
2. Click "Download"
   (This file is large, may take a few minutes)
3. Extract the CSV
4. Rename to: PS_20174392719_1491204439457_log.csv
5. Place in: ./datasets/
```

**Verify**: Should have ~6,000,000 rows

---

### Updating .env with Dataset Paths

After downloading, edit your `.env` file:

```env
# If datasets are in ./datasets/ (recommended)
FINANCIAL_FRAUD_DATASET_PATH=./datasets/financial_fraud_detection_dataset.csv
GERMAN_CREDIT_DATASET_PATH=./datasets/german_credit_data.csv
CREDIT_RISK_DATASET_PATH=./datasets/nz_bank_loans_synthetic_with_dates.csv
TRANSACTION_DATASET_PATH=./datasets/PS_20174392719_1491204439457_log.csv
```

Or use absolute paths:
```env
# Windows example
FINANCIAL_FRAUD_DATASET_PATH=C:\Users\YourName\Documents\datasets\financial_fraud_detection_dataset.csv

# Linux example
FINANCIAL_FRAUD_DATASET_PATH=/home/user/datasets/financial_fraud_detection_dataset.csv
```

### Verify Datasets

Test that all datasets are found:

```bash
python -c "from config_loader import Config; Config.load_env(); Config.validate_datasets()"
```

Should show:
```
✓ Found financial  : ./datasets/financial_fraud_detection_dataset.csv
✓ Found credit     : ./datasets/german_credit_data.csv
✓ Found loans      : ./datasets/nz_bank_loans_synthetic_with_dates.csv
✓ Found transaction: ./datasets/PS_20174392719_1491204439457_log.csv
```

If you see ✗ errors, fix the paths in `.env`

---

## Quick Start (Development Mode)

### Recommended for testing (2-5 minutes, 5% of data)

The default `.env` uses Quick-Run mode (fast for testing):

```bash
# Make sure virtual environment is activated
# (fraud_env) should show in your prompt

python preprocessing_and_training.py
```

**Expected output**:
```
Device: cuda (or cpu)
Quick Run: True (subsample=0.05)

Dataset 1 loaded: (25000, 45)
Dataset 2 loaded: (1000, 21)
Dataset 3 loaded: (5000, 42)
Dataset 4 loaded: (300000, 11)  # 5% of 6M transactions

Graph created: 1000 companies, 300000 transactions, 5000 loans

=== CLASS IMBALANCE ANALYSIS ===
...

=== TRAINING COMPLETED ===
=== EVALUATION RESULTS ===
              precision    recall  f1-score   support
     Non-Fraud      0.9921    0.9954    0.9937
         Fraud      0.6235    0.5234    0.5689

ROC-AUC Score: 0.7523
PR-AUC Score: 0.6234

=== EXPLAINABILITY PHASE ===
[OK] Graph-level explanation completed
[OK] SHAP explanation completed

[OK] All explanations saved to: ./explanations/fraud_detection_gnn/

--- Script Completed ---
```

### Results Location

Results are saved in:
```
./explanations/fraud_detection_gnn/
├── aggregated_results.csv
├── summary_report.md
├── graph_explanation_node_12345.png
└── shap_explanation_txn_12345.png
```

View the summary:
```bash
# Windows
type explanations\fraud_detection_gnn\summary_report.md

# Linux/macOS
cat explanations/fraud_detection_gnn/summary_report.md
```

---

## Full Training (Production Mode)

For complete results using all data (15-60 minutes):

### Step 1: Edit .env

```env
# Disable quick-run
QUICK_RUN=False
QUICK_RUN_SUBSAMPLE=1.0

# Increase training
EPOCHS=8  # or 20 for better results

# For better features explanations:
QUICK_SHAP_SAMPLES=100
```

### Step 2: Run

```bash
python preprocessing_and_training.py
```

This will use ALL data (~6 million transactions), taking much longer but producing better results.

---

## Configuration Reference

### .env Parameters

#### Paths (You must set these)
```env
FINANCIAL_FRAUD_DATASET_PATH=./datasets/financial_fraud_detection_dataset.csv
GERMAN_CREDIT_DATASET_PATH=./datasets/german_credit_data.csv
CREDIT_RISK_DATASET_PATH=./datasets/nz_bank_loans_synthetic_with_dates.csv
TRANSACTION_DATASET_PATH=./datasets/PS_20174392719_1491204439457_log.csv
```

#### Output
```env
EXPLANATION_OUTPUT_DIR=./explanations    # Where results save
EXPERIMENT_NAME=fraud_detection_gnn     # Experiment folder name
```

#### Model Architecture
```env
HIDDEN_DIM=16                    # Model size (larger = slower but possibly better)
                                 # Options: 8, 16, 32, 64
MAX_TRX_PER_COMPANY=8           # Max transactions per company
LIMIT_PER_ENTITY=8              # Max related entities to track
MAKE_CLIQUES=False              # Create full relationship networks
```

#### Training
```env
EPOCHS=8                         # Training iterations (more = better but slower)
LEARNING_RATE=0.001            # How fast model learns
WEIGHT_DECAY=0.0001            # Regularization strength
LOSS_TYPE=focal                 # Options: weighted_ce, focal, combined
```

#### Quick-Run (for development)
```env
QUICK_RUN=True                  # Use only % of data (faster testing)
QUICK_RUN_SUBSAMPLE=0.05       # 0.05 = 5%, 0.1 = 10%, 1.0 = 100%
QUICK_SHAP_SAMPLES=10          # Background samples for explanations
```

Change to `False`/`1.0`/`100` for full production training.

---

## Troubleshooting

### Issue: "No module named 'torch'"

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "Datasets not found"

```bash
# Check if .env file exists
ls .env  # Linux/macOS
dir .env # Windows

# Make sure datasets are in ./datasets/
ls datasets/  # Linux/macOS
dir datasets  # Windows

# Validate:
python -c "from config_loader import Config; Config.validate_datasets()"
```

The error message will tell you which file is missing. Download it and update `.env`.

### Issue: "CUDA out of memory"

```env
# In .env, reduce model size:
QUICK_RUN=True
QUICK_RUN_SUBSAMPLE=0.01  # Use only 1%
HIDDEN_DIM=8              # Smaller model
```

Then rerun.

### Issue: "SHAP not found"

```bash
# Solution A: Install SHAP
pip install shap

# Solution B: Disable SHAP in .env
ENABLE_SHAP_EXPLANATIONS=False
```

### Issue: "Python command not found"

- **Windows**: Add Python to PATH, or use `python.exe` instead of `python`
- **Linux/macOS**: Use `python3` instead of `python`, or install Python with package manager
  ```bash
  # Ubuntu/Debian
  sudo apt install python3 python3-venv python3-pip
  ```

### Issue: "Very slow training"

This is normal on CPU. Solutions:

1. **Use GPU** (if available)
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # If True: GPU is available, script will use it automatically
   # If False: No GPU, or PyTorch installed without CUDA
   ```

2. **Enable Quick-Run**
   ```env
   QUICK_RUN=True
   QUICK_RUN_SUBSAMPLE=0.01
   ```

3. **GPU Setup** (if you have NVIDIA GPU)
   See: https://pytorch.org/get-started/locally/

---

## Running Multiple Experiments

Each run automatically saves results separately:

```bash
# Experiment 1 (default settings)
python preprocessing_and_training.py
# Saves to: ./explanations/fraud_detection_gnn/

# Before Experiment 2, edit .env:
# Change: EXPERIMENT_NAME=test_run_v2
# Change: HIDDEN_DIM=32 (try different model size)

python preprocessing_and_training.py
# Saves to: ./explanations/test_run_v2/

# Compare results:
ls explanations/  # See all experiments
```

---

## Expected Results

### Metrics You Should See

```
Accuracy: 0.98-0.99 (but this is misleading for imbalanced data)
F1-Score: 0.45-0.65 (better metric for fraud detection)
ROC-AUC: 0.75-0.85  (probability ranking)
Precision: 0.60-0.75 (of detected frauds, how many are real)
Recall: 0.50-0.70    (of real frauds, how many detected)
```

### Output Files

After running, you get:

1. **`aggregated_results.csv`** - All metrics in spreadsheet format
2. **`summary_report.md`** - Human-readable narrative
3. **`graph_explanation_node_X.png`** - Which relationships were important
4. **`shap_explanation_txn_X.png`** - Which features were important

Use these for your thesis write-up!

---

## For Your Supervisor

### Quick Instructions to Give

1. **Setup** (first time only)
   ```bash
   # Copy .env.example to .env
   # Edit .env with dataset paths
   # pip install -r requirements.txt
   ```

2. **Download Datasets**
   - See "Dataset Configuration" section above
   - Place in `./datasets/` folder

3. **Run Training**
   ```bash
   python preprocessing_and_training.py
   ```

4. **View Results**
   - Check `./explanations/fraud_detection_gnn/`
   - View `summary_report.md` for findings

### Maintenance

- **Only one .env file** - Same for all experiments, just change settings
- **Results stay organized** - Each run gets its own folder via `EXPERIMENT_NAME`
- **No hardcoded paths** - Everything configurable via .env
- **Easy to reproduce** - Same .env settings = same results

---

## Advanced: Understanding the Script

### Data Flow

```
Datasets
   ↓
.env (paths) → Config Loader
   ↓
Preprocessing (scaling, encoding)
   ↓
Heterogeneous Graph Construction
   ↓
Model (MultiFraudNet)
   ↓
Training with Focal Loss
   ↓
Evaluation
   ↓
Explanations (Graph + SHAP)
   ↓
Results → ./explanations/
```

### Key Components

1. **config_loader.py** - Reads .env and config.yaml
2. **preprocessing_and_training.py** - Main training loop
3. **explainability_logger.py** - Logs model explanations
4. **config.yaml** - Dataset documentation

You shouldn't need to edit these, just configure via `.env`.

---

## Getting Help

### Before Asking for Help

1. Check **Troubleshooting** section above
2. Read error messages carefully - they usually point to the problem
3. Verify dataset paths:
   ```bash
   python -c "from config_loader import Config; Config.validate_datasets()"
   ```
4. Check .env file:
   ```bash
   cat .env  # Linux/macOS
   type .env # Windows
   ```

### Common Commands

```bash
# Activate environment
fraud_env\Scripts\activate          # Windows
source fraud_env/bin/activate       # Linux/macOS

# Check setup
python -c "from config_loader import Config; Config.validate_datasets()"

# Run training
python preprocessing_and_training.py

# View results
ls explanations/
```

---

## Performance Tips

### For Faster Results
- Use GPU if available
- Enable QUICK_RUN=True (default)
- Disable SHAP: ENABLE_SHAP_EXPLANATIONS=False
- Reduce EPOCHS to 4

### For Better Results
- Use full data: QUICK_RUN=False
- Increase EPOCHS to 20
- Increase HIDDEN_DIM to 32
- Enable SHAP with more samples

### Balance
Default settings provide good balance between speed and accuracy.

---

## Summary

### To get started:

1. ✅ Install Python
2. ✅ Create virtual environment
3. ✅ Install requirements: `pip install -r requirements.txt`
4. ✅ Copy `.env.example` to `.env`
5. ✅ Download 4 datasets
6. ✅ Update `.env` with dataset paths
7. ✅ Run: `python preprocessing_and_training.py`
8. ✅ Check results in `./explanations/`

### Estimated Time

- **Initial setup**: 30 minutes (mostly downloading datasets)
- **First run**: 5 minutes (quick-run mode)
- **Full training**: 30 minutes (full data)

---

## Questions?

Check the relevant section in this guide:
- **Setup issues** → Installation & Setup
- **Missing data** → Dataset Configuration
- **Script errors** → Troubleshooting
- **Configuration** → Configuration Reference
- **Results** → Understanding the Output

---

**Last Updated**: December 2025  
**Status**: Ready to Use ✅  
**Python Version**: 3.8+  
**Total Dataset Size**: ~500 MB
