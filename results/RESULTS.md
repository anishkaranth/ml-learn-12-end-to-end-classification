# Results — End-to-End Wine Classification

## What was run

- **Dataset:** UCI Wine via `data/wine.csv`
- **Split:** stratified 60/20/20, StandardScaler on train
- **Models:** logistic_regression vs mlp; select by val macro-F1
- **Command:** `python train.py`

## Headline metrics

**Selected (`logistic_regression`) test:** accuracy=0.9722, macro-F1=0.9752

## Plots

- `lr_coefficients.png`
- `confusion_matrix_test.png`
- `class_balance.png`

## Files

- `metrics.json` — structured metrics from the smoke run
- `JSON.shot` — run snapshot (timestamp, repo, command, metrics) as valid JSON
