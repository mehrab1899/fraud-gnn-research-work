# Fraud Detection GNN (Thesis)

This repository contains a heterogeneous GNN-based fraud detection pipeline with two-level explainability: graph-level (structural) and feature-level (financial). The training script is `preprocessing_and_training.py`.

---

## What to push to GitHub

- All source code (Python files) — e.g. `preprocessing_and_training.py`, `explainability_logger.py`, helper scripts, modules.
- Documentation: `README.md`, `EXPLAINABILITY_LOGGING_GUIDE.md`, `EXPLAINABILITY_IMPLEMENTATION_SUMMARY.md`, `QUICK_REFERENCE.md`.
- Small example data or a small synthetic dataset (OPTIONAL). Do NOT push large datasets (add them to `.gitignore` or use Git LFS).
- Configuration files (.github/workflows/*), requirements files, .gitignore, LICENSE.

DO NOT push:
- Raw datasets (e.g., CSVs in `D:\Thesis\Dataset`) — keep them local or use external storage.
- Large model checkpoints, raw logs, or other heavy binary assets (use Git LFS if necessary).

---

## Setup (local development)

Recommended: create a Python virtual environment.

Windows (PowerShell):

```powershell
cd d:\Thesis
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

This script expects datasets referenced in absolute paths (e.g., `D:\Thesis\Dataset\...`). For portability, set a single environment variable or edit constants at the top of `preprocessing_and_training.py` to point to your dataset folder. Example:

```powershell
$env:THESIS_DATA_DIR = "D:\\Thesis\\Dataset"
```

And update script to read from `os.environ.get('THESIS_DATA_DIR')` or similar.

---

## Run the script

Once dependencies are installed and datasets are available locally, run:

```powershell
python d:\Thesis\preprocessing_and_training.py
```

This will run a quick subsampled prototype by default (check flags at the top of the script: `QUICK_RUN`, `QUICK_RUN_SUBSAMPLE`). Disable `QUICK_RUN` for a full run.

---

## Using Git and pushing to GitHub

1. Initialize git and commit your project:

```bash
cd d:\Thesis
git init
git add .
git commit -m "Initial commit: fraud detection GNN + explainability logger"
```

2. Create an empty repository on GitHub (via web UI or `gh` CLI), then push:

```bash
git remote add origin git@github.com:<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If your data is large, consider using Git LFS:

```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
```

---

## Continuous Integration (optional)

We included a simple Actions workflow skeleton (`.github/workflows/smoke_test.yml`) that you can enable to run a lightweight smoke test on push. It will not run heavy training — update it to your needs.

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
└─ .github/
   └─ workflows/
      └─ smoke_test.yml
```

---

## Final notes

- Keep raw datasets out of Git — use cloud storage (S3, GDrive) or Git LFS.
- Install PyTorch and PyG using the official guides for correct wheels.
- The explainability outputs are written to `D:\Thesis\explanations\` by default; change `EXPLANATION_OUTPUT_DIR` in the script to commit-friendly paths if you want small example outputs saved in the repo.

If you'd like, I can:

- Create a repo skeleton and push these files (I can generate git commands for you to run locally),
- Add a basic GitHub Actions workflow file now (I created a skeleton below), or
- Make the dataset path configurable via environment variables in `preprocessing_and_training.py`.

Which of these would you like me to do next?