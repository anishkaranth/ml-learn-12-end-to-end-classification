# Week 12 — End-to-End Supervised Classification

## Learning goal

This is the curriculum marker that **an actual ML model is built end-to-end** — fuller than the scratch-algorithm series (`ml-learn-01` … `ml-learn-11`). You load a real tabular dataset, run a light EDA, split carefully, preprocess without leakage, train **two** sklearn models, compare metrics, persist artifacts, and score held-out rows.

## Dataset

**UCI Wine** (via `sklearn.datasets.load_wine`), bundled offline as [`data/wine.csv`](data/wine.csv).

| | |
|---|---|
| Rows | 178 |
| Features | 13 chemical measurements (alcohol, malic acid, ash, … proline) |
| Target | Cultivar class `0 / 1 / 2` (multi-class) |
| Source | Written out from sklearn so the project runs **without re-download** |

Class balance is roughly 59 / 71 / 48 — mildly imbalanced, so we use **stratified** splits and report **macro** precision / recall / F1.

## Pipeline (what the code does)

1. Load `data/wine.csv`
2. Light EDA — shapes, class balance, basic feature stats
3. Stratified train / validation / test split (**60 / 20 / 20**, `random_seed=42`)
4. `StandardScaler` fit on **train only**, then transform val & test
5. Train two models for comparison:
   - `LogisticRegression`
   - `MLPClassifier` (small `(32, 16)` network)
6. Metrics on val **and** test: accuracy, macro P/R/F1, confusion matrix
7. Persist best model (by val macro-F1) + scaler with **joblib** under `artifacts/`
8. `predict.py` reloads artifacts and scores held-out test rows

## Project layout

```
README.md
requirements.txt
.gitignore
data/wine.csv              # bundled dataset
data/.gitkeep
src/train.py               # full training pipeline
src/predict.py             # load artifacts → score held-out rows
notebooks/end_to_end.ipynb # walkthrough + plots
artifacts/                 # best_model.joblib, scaler.joblib, meta.json
artifacts/.gitkeep
```

Trained artifacts are **committed** after a smoke train so `clone → predict` works without re-training.

## How to install & run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train (script):**

```bash
python src/train.py
```

Prints EDA, trains both models, writes:

- `artifacts/best_model.joblib`
- `artifacts/scaler.joblib`
- `artifacts/meta.json` (metrics + feature list)

**Predict:**

```bash
python src/predict.py
python src/predict.py --n 8
```

**Notebook (interactive):**

```bash
jupyter notebook notebooks/end_to_end.ipynb
```

## What “done” looks like

- [x] CSV loads from `data/` (no network needed)
- [x] EDA prints shapes + class balance
- [x] Fixed-seed stratified 60/20/20 split
- [x] Scaler fit on train only
- [x] ≥2 models trained and compared
- [x] Val + test metrics (acc, macro P/R/F1, confusion matrix)
- [x] Best model + scaler saved under `artifacts/`
- [x] `predict.py` scores held-out rows from artifacts
- [x] README documents install / train / predict

## What you'll learn

How a **real supervised classification pipeline** hangs together: data → EDA → leakage-safe split & scaling → model comparison → metrics → persisted artifacts → offline prediction.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`, `joblib` (see `requirements.txt`)

## Relation to earlier weeks

Weeks 1–11 built algorithms **from scratch** (linear/logistic, k-means, trees, kNN, NB, PCA, perceptron, softmax, ridge, MLP). This week uses **sklearn** on purpose — the milestone is the **end-to-end workflow**, not another toy optimizer.
