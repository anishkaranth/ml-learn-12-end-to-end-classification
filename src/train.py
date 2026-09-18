#!/usr/bin/env python3
"""
End-to-end supervised classification pipeline.

Loads bundled wine CSV → light EDA → train/val/test split → standardize
(fit on train only) → train LogisticRegression + MLPClassifier → evaluate
→ persist best model + scaler under artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wine.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
TARGET_COL = "target"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def eda(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("LIGHT EDA")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("\nClass balance (target):")
    counts = df[TARGET_COL].value_counts().sort_index()
    for cls, n in counts.items():
        print(f"  class {cls}: {n} ({100.0 * n / len(df):.1f}%)")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    print("\nBasic stats (features):")
    print(df[feature_cols].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print()


def metrics_bundle(y_true, y_pred, label: str) -> dict:
    out = {
        "split": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n--- Metrics: {label} ---")
    print(f"  accuracy         : {out['accuracy']:.4f}")
    print(f"  precision (macro): {out['precision_macro']:.4f}")
    print(f"  recall (macro)   : {out['recall_macro']:.4f}")
    print(f"  f1 (macro)       : {out['f1_macro']:.4f}")
    print(f"  confusion matrix :\n{np.array(out['confusion_matrix'])}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return out


def main() -> None:
    np.random.seed(RANDOM_SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    eda(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # 60% train / 20% val / 20% test (stratified), fixed seed
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"Split sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # Preprocess: fit scaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="lbfgs",  # better for small tabular datasets
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)
        model.fit(X_train_s, y_train)
        trained[name] = model
        val_pred = model.predict(X_val_s)
        test_pred = model.predict(X_test_s)
        results[name] = {
            "val": metrics_bundle(y_val, val_pred, f"{name} / val"),
            "test": metrics_bundle(y_test, test_pred, f"{name} / test"),
        }

    # Select best by validation macro-F1
    best_name = max(results, key=lambda n: results[n]["val"]["f1_macro"])
    best_model = trained[best_name]
    print("=" * 60)
    print(f"BEST MODEL (by val macro-F1): {best_name}")
    print(f"  val F1={results[best_name]['val']['f1_macro']:.4f}  "
          f"test F1={results[best_name]['test']['f1_macro']:.4f}  "
          f"test acc={results[best_name]['test']['accuracy']:.4f}")
    print("=" * 60)

    # Persist artifacts
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    meta_path = ARTIFACTS_DIR / "meta.json"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "best_model": best_name,
        "random_seed": RANDOM_SEED,
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "class_labels": sorted(int(c) for c in np.unique(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "metrics": results,
        "dataset": "sklearn wine (bundled as data/wine.csv)",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {model_path}")
    print(f"Wrote {scaler_path}")
    print(f"Wrote {meta_path}")
    print("\nDone. Run: python src/predict.py")


if __name__ == "__main__":
    main()
