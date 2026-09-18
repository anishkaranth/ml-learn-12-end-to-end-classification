# Results — End-to-End Wine Classification

## What was run

- **Dataset:** UCI Wine via `data/wine.csv` (178 rows, 13 features, 3 cultivar classes)
- **Seed:** `42`
- **Split:** stratified 60% / 20% / 20% train / val / test
- **Preprocess:** `StandardScaler` fit on train only
- **Models compared:** `logistic_regression`, `mlp` (hidden `(64, 32)`, `lbfgs`)
- **Selection:** best by **validation macro-F1** → **`logistic_regression`**

## Headline metrics

| Model | Val Acc | Val macro-F1 | Test Acc | Test macro-F1 |
|---|---:|---:|---:|---:|
| `logistic_regression` **(selected)** | 1.0000 | 1.0000 | 0.9722 | 0.9752 |
| `mlp` | 0.9444 | 0.9453 | 1.0000 | 1.0000 |

**Selected model (`logistic_regression`) test:** accuracy=0.9722, macro-F1=0.9752, macro-P=0.9744, macro-R=0.9778

## Plots

- `class_balance.png` — target class counts
- `confusion_matrix_test.png` — test-set confusion matrix for the selected model
- `lr_coefficients.png` — mean |coef| feature importance for LogisticRegression

## Files

- `metrics.json` — structured per-model val/test metrics + best model name
