"""
Learning Curves Diagnostic — Telecom Churn
Run: python learning_curve.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# =========================
# Load Data
# =========================
def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    # clean column names
    df.columns = df.columns.str.strip().str.lower()

    # drop id if exists
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # target
    y = df["churned"]
    X = df.drop(columns=["churned"])

    return X, y


# =========================
# Features
# =========================
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents"
]

CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]


# =========================
# Pipeline
# =========================
def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# =========================
# Learning Curve Plot
# =========================
def plot_learning_curve(X, y):
    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1",                        # مهم بسبب class imbalance
        train_sizes=np.linspace(0.1, 1.0, 5),# على الأقل 5 قيم
        n_jobs=-1
    )

    # mean & std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # plot
    plt.figure(figsize=(8, 6))

    # training
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2)

    # validation
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2)

    plt.title("Learning Curve — Logistic Regression")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()

    plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X, y = load_data()
    plot_learning_curve(X, y)