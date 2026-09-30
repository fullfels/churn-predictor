# Churn Predictor

A compact, reproducible binary-classification project that predicts customer churn from tabular data. It demonstrates a practical scikit-learn workflow: synthetic data generation, preprocessing, cross-validation, model evaluation, and artifact export.

## Highlights

- Mixed numeric and categorical features
- Leakage-safe preprocessing with `ColumnTransformer`
- Logistic regression with class balancing
- ROC AUC, F1, precision, recall, and confusion matrix
- Deterministic training and a small test suite

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python train.py --output-dir artifacts
pytest -q
```

The training command writes `metrics.json`, `model.joblib`, and `confusion_matrix.png` to the output directory.

## Project structure

```text
train.py              training pipeline and CLI
tests/test_train.py   smoke tests for data and model quality
requirements.txt      runtime and test dependencies
```

## Example result

With the default random seed, the model should achieve ROC AUC above 0.75 on the generated holdout set. Exact values can vary slightly between library versions.

## License

MIT
