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
import matplotlib.pyplot as plt
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
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
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


def save_class_balance_plot(y: np.ndarray, out_path: Path) -> None:
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xlabel("Cultivar class")
    ax.set_ylabel("Count")
    ax.set_title("Wine class balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix_plot(cm: list, title: str, out_path: Path) -> None:
    mat = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_lr_coefficients_plot(model: LogisticRegression, feature_cols: list[str], out_path: Path) -> None:
    # Mean absolute coefficient across classes as a simple importance proxy
    coef = np.mean(np.abs(model.coef_), axis=0)
    order = np.argsort(coef)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(feature_cols)[order], coef[order], color="#4C72B0")
    ax.set_xlabel("Mean |coefficient| across classes")
    ax.set_title("LogisticRegression feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_results_package(
    best_name: str,
    results: dict,
    feature_cols: list[str],
    trained: dict,
    y: np.ndarray,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": "sklearn wine (bundled as data/wine.csv)",
        "random_seed": RANDOM_SEED,
        "split": "stratified 60/20/20 train/val/test",
        "models_compared": list(results.keys()),
        "best_model": best_name,
        "selection_criterion": "validation macro-F1",
        "metrics": results,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    # Plots
    save_class_balance_plot(y, RESULTS_DIR / "class_balance.png")
    best_cm = results[best_name]["test"]["confusion_matrix"]
    save_confusion_matrix_plot(
        best_cm,
        f"Confusion matrix — {best_name} (test)",
        RESULTS_DIR / "confusion_matrix_test.png",
    )
    if "logistic_regression" in trained:
        save_lr_coefficients_plot(
            trained["logistic_regression"],
            feature_cols,
            RESULTS_DIR / "lr_coefficients.png",
        )

    best = results[best_name]
    lines = [
        "# Results — End-to-End Wine Classification",
        "",
        "## What was run",
        "",
        f"- **Dataset:** UCI Wine via `data/wine.csv` (178 rows, 13 features, 3 cultivar classes)",
        f"- **Seed:** `{RANDOM_SEED}`",
        f"- **Split:** stratified 60% / 20% / 20% train / val / test",
        f"- **Preprocess:** `StandardScaler` fit on train only",
        f"- **Models compared:** `logistic_regression`, `mlp` (hidden `(64, 32)`, `lbfgs`)",
        f"- **Selection:** best by **validation macro-F1** → **`{best_name}`**",
        "",
        "## Headline metrics",
        "",
        f"| Model | Val Acc | Val macro-F1 | Test Acc | Test macro-F1 |",
        f"|---|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        mark = " **(selected)**" if name == best_name else ""
        lines.append(
            f"| `{name}`{mark} | {m['val']['accuracy']:.4f} | {m['val']['f1_macro']:.4f} "
            f"| {m['test']['accuracy']:.4f} | {m['test']['f1_macro']:.4f} |"
        )
    lines += [
        "",
        f"**Selected model (`{best_name}`) test:** "
        f"accuracy={best['test']['accuracy']:.4f}, "
        f"macro-F1={best['test']['f1_macro']:.4f}, "
        f"macro-P={best['test']['precision_macro']:.4f}, "
        f"macro-R={best['test']['recall_macro']:.4f}",
        "",
        "## Plots",
        "",
        "- `class_balance.png` — target class counts",
        "- `confusion_matrix_test.png` — test-set confusion matrix for the selected model",
        "- `lr_coefficients.png` — mean |coef| feature importance for LogisticRegression",
        "",
        "## Files",
        "",
        "- `metrics.json` — structured per-model val/test metrics + best model name",
        "",
    ]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {RESULTS_DIR / 'RESULTS.md'}")
    print(f"Wrote plots under {RESULTS_DIR}/")



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

    write_results_package(best_name, results, feature_cols, trained, y)

    print("\nDone. Run: python src/predict.py")


if __name__ == "__main__":
    main()
