#!/usr/bin/env python3
"""
Load persisted artifacts and score a few held-out rows from the wine CSV.

Usage:
  python src/predict.py              # score a small held-out sample
  python src/predict.py --n 5        # score N rows from the test-like split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "wine.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
TARGET_COL = "target"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rows with the trained classifier")
    parser.add_argument("--n", type=int, default=5, help="Number of held-out rows to score")
    args = parser.parse_args()

    model_path = ARTIFACTS_DIR / "best_model.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    meta_path = ARTIFACTS_DIR / "meta.json"

    for p in (model_path, scaler_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python src/train.py` first to create artifacts."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]

    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # Recreate the same 60/20/20 split so we score true held-out (test) rows
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_SEED, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    n = min(args.n, len(y_test))
    Xs = scaler.transform(X_test[:n])
    preds = model.predict(Xs)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
    else:
        proba = None

    print("=" * 60)
    print(f"Loaded best model: {meta['best_model']}")
    print(f"Scoring {n} held-out test rows")
    print("=" * 60)
    for i in range(n):
        true_y = int(y_test[i])
        pred_y = int(preds[i])
        line = f"  row {i}: true={true_y}  pred={pred_y}"
        if proba is not None:
            probs = ", ".join(f"p{c}={proba[i, c]:.3f}" for c in range(proba.shape[1]))
            line += f"  [{probs}]"
        mark = "✓" if true_y == pred_y else "✗"
        print(f"{line}  {mark}")

    acc = float(np.mean(preds == y_test[:n]))
    print(f"\nSample accuracy ({n} rows): {acc:.4f}")
    print(f"Full test metrics are in {meta_path}")


if __name__ == "__main__":
    main()
