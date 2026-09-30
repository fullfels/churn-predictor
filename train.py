"""Train a customer-churn classifier on realistic synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["tenure_months", "monthly_charge", "support_calls", "usage_hours"]
CATEGORICAL_FEATURES = ["contract", "payment_method", "region"]


def make_customer_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Create a deterministic customer table with a learnable churn signal."""
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_samples)
    monthly_charge = rng.normal(68, 22, n_samples).clip(15, 150)
    support_calls = rng.poisson(1.8, n_samples)
    usage_hours = rng.gamma(3.2, 8, n_samples)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_samples, p=[0.55, 0.27, 0.18])
    payment = rng.choice(["card", "bank-transfer", "electronic-check"], n_samples)
    region = rng.choice(["north", "south", "east", "west"], n_samples)

    logit = (
        -0.035 * tenure
        + 0.025 * (monthly_charge - 65)
        + 0.48 * support_calls
        - 0.018 * usage_hours
        + 1.05 * (contract == "month-to-month")
        + 0.35 * (payment == "electronic-check")
        - 1.2
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly_charge.round(2),
            "support_calls": support_calls,
            "usage_hours": usage_hours.round(1),
            "contract": contract,
            "payment_method": payment,
            "region": region,
            "churn": churn,
        }
    )
    for column in ["monthly_charge", "payment_method"]:
        missing = rng.choice(n_samples, size=max(1, n_samples // 50), replace=False)
        frame.loc[missing, column] = np.nan
    return frame


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline(
        [("preprocess", preprocessing), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )


def train(random_state: int = 42) -> tuple[Pipeline, dict[str, float], np.ndarray, np.ndarray]:
    data = make_customer_data(random_state=random_state)
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    pipeline = build_pipeline().fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "f1": round(float(report["1"]["f1-score"]), 4),
        "precision": round(float(report["1"]["precision"]), 4),
        "recall": round(float(report["1"]["recall"]), 4),
    }
    return pipeline, metrics, y_test.to_numpy(), predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, metrics, y_test, predictions = train(args.seed)
    joblib.dump(model, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, colorbar=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
