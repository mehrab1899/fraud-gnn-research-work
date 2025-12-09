# Fraud Detection GNN (Thesis)

This repository contains a heterogeneous GNN-based fraud detection pipeline with two-level explainability: graph-level (structural) and feature-level (financial). The training script is `preprocessing_and_training.py`.

---

## Setup (local development)

Clone the project using this Github repo link:

https://github.com/mehrab1899/fraud-gnn-research-work

Recommended: create a Python virtual environment.

Windows (PowerShell):

```powershell
# In your cloned directory activate virtual env using following command
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
cd /path/to/Thesis
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Install PyTorch and PyTorch Geometric (PyG)

PyTorch and PyG have platform- and CUDA-specific wheels. Install PyTorch first following the official instructions here:

- PyTorch install guide: https://pytorch.org/get-started/locally/

Example (CPU-only):

```bash
pip install "torch" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Example (CUDA 11.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 "torch" torchvision torchaudio
```

After installing PyTorch, install PyG following the official instructions:

- PyG install guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Example (one-line from PyG docs — replace versions as appropriate):

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html
pip install torch-geometric
```

(We intentionally leave exact wheel URLs out of `requirements.txt` since they are platform dependent.)

---

## Environment variables & dataset paths

This script loads configuration from `.env` and `config.yaml` files using the `Config` class. Create a `.env` file in the project root with the following variables:

```
# Dataset paths (absolute paths to your dataset files)
FINANCIAL_FRAUD_DATASET_PATH=D:\Thesis\datasets\financial_fraud_detection_dataset.csv
GERMAN_CREDIT_DATASET_PATH=D:\Thesis\datasets\german_credit_data.csv
CREDIT_RISK_DATASET_PATH=D:\Thesis\datasets\nz_bank_loans_synthetic_with_dates.csv
TRANSACTION_DATASET_PATH=D:\Thesis\datasets\PS_20174392719_1491204439457_log.csv

# Model hyperparameters
HIDDEN_DIM=16
MAX_TRX_PER_COMPANY=8
LIMIT_PER_ENTITY=8
MAKE_CLIQUES=False
EPOCHS=8
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
QUICK_RUN=True
QUICK_RUN_SUBSAMPLE=0.05
QUICK_SHAP_SAMPLES=10
LOSS_TYPE=focal

# Output paths
EXPLANATION_OUTPUT_DIR=./explanations
EXPERIMENT_NAME=fraud_detection_gnn
```

**Note:** Update the dataset paths to match your local file locations. The script will validate that all datasets exist before training.

---

## Run the script

Once dependencies are installed, datasets are available, and the `.env` file is configured, run:

```powershell
python preprocessing_and_training.py
```

This will run a quick subsampled prototype by default (set `QUICK_RUN=False` in `.env` for a full run).

---

## Dataset Sources

Download these datasets and place them in your `datasets/` folder, then update the paths in your `.env` file:

1. **Financial Fraud Detection Dataset**
   - Source: https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset
   - File: `financial_fraud_detection_dataset.csv`

2. **German Credit Dataset**
   - Source: https://www.kaggle.com/datasets/uciml/german-credit
   - File: `german_credit_data.csv`

3. **Credit Risk Dataset (NZ Banking)**
   - Source: https://www.kaggle.com/datasets/sandycandy7/synthetic-nz-banking-loan-and-fraud-risk-dataset
   - File: `nz_bank_loans_synthetic_with_dates.csv`

4. **Transaction/Payments Fraud Detection Dataset**
   - Source: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
   - File: `PS_20174392719_1491204439457_log.csv`

---

## Suggested project structure

```
fraud-gnn-research/
├── .venv/                          # Virtual environment (excluded from git)
├── datasets/                       # Dataset files (excluded from git)
│   ├── financial_fraud_detection_dataset.csv
│   ├── german_credit_data.csv
│   ├── nz_bank_loans_synthetic_with_dates.csv
│   └── PS_20174392719_1491204439457_log.csv
├── explanations/                   # Model explanations output (excluded from git)
├── config.yaml                     # Configuration template
├── config_loader.py                # Configuration loader
├── explainability_logger.py        # Explainability logging
├── preprocessing_and_training.py   # Main training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## Suggested repo layout

```
Thesis/
├─ preprocessing_and_training.py
├─ explainability_logger.py
├─ requirements.txt
├─ README.md
├─ EXPLAINABILITY_LOGGING_GUIDE.md
├─ EXPLAINABILITY_IMPLEMENTATION_SUMMARY.md
├─ QUICK_REFERENCE.md
├─ .gitignore

```

---

## Final notes

- Install PyTorch and PyG using the official guides for correct wheels.
- The explainability outputs are written to `./explanations/` by default; change `EXPLANATION_OUTPUT_DIR` in `.env` if needed.
- Use the `.env` file to configure all dataset paths and hyperparameters.

