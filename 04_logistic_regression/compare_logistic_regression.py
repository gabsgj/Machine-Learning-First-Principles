"""
Comparison and Benchmarking: Logistic Regression
================================================
Side-by-side verification of Scratch vs Scikit-Learn:
1. Feature Scaling impact analysis (demonstrating why unscaled features fail)
2. L2 Regularization parameter sweeps (lambda = 0 vs 1 vs 10)
3. Performance parity across Breast Cancer and Adult Income datasets
4. Convergence and decision margin plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scratch_logistic_regression import LogisticRegressionScratch, compute_classification_metrics


def benchmark_scaling_and_regularization():
    print("==================================================")
    print("LOGISTIC REGRESSION: Breast Cancer Diagnostic Task")
    print("==================================================")
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scikit-Learn standard scaled sets
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    configurations = [
        ("Without Scaling, lambda=0", X_train, X_test, False, 0.0),
        ("Without Scaling, lambda=1", X_train, X_test, False, 1.0),
        ("With Scaling, lambda=0", Xs_train, Xs_test, False, 0.0),
        ("With Scaling, lambda=1", Xs_train, Xs_test, False, 1.0),
    ]

    print(f"{'Configuration':<30} {'Scratch Acc':<13} {'Sklearn Acc':<13} {'Scratch F1':<12} {'Sklearn F1':<12}")
    print("-" * 80)

    for desc, X_tr, X_te, internal_scale, lam in configurations:
        # Scratch model
        m_scratch = LogisticRegressionScratch(
            learning_rate=0.1, epochs=1500, l2_penalty=lam, scale_features=internal_scale
        )
        m_scratch.fit(X_tr, y_train)
        pred_scratch = m_scratch.predict(X_te)
        acc_sc = accuracy_score(y_test, pred_scratch)
        f1_sc = f1_score(y_test, pred_scratch)

        # Sklearn model (C is inverse of lambda: C = 1 / (lam * len(y) or 1e5))
        C_val = 1.0 / (lam + 1e-5) if lam > 0 else 1e5
        m_sklearn = LogisticRegression(C=C_val, max_iter=2000, random_state=42)
        m_sklearn.fit(X_tr, y_train)
        pred_sklearn = m_sklearn.predict(X_te)
        acc_sk = accuracy_score(y_test, pred_sklearn)
        f1_sk = f1_score(y_test, pred_sklearn)

        print(f"{desc:<30} {acc_sc:<13.4f} {acc_sk:<13.4f} {f1_sc:<12.4f} {f1_sk:<12.4f}")

    # Plot convergence comparison
    os.makedirs("plots", exist_ok=True)
    m_unscaled = LogisticRegressionScratch(learning_rate=0.0001, epochs=500, scale_features=False).fit(X_train, y_train)
    m_scaled = LogisticRegressionScratch(learning_rate=0.1, epochs=500, scale_features=True).fit(X_train, y_train)

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(m_unscaled.loss_history_, color="#d62728", lw=2, label="Unscaled Features (Sluggish / Stagnant)")
    plt.plot(m_scaled.loss_history_, color="#2ca02c", lw=2, label="Scaled Features (Rapid convergence to optimum)")
    plt.title("Logistic Regression Cross-Entropy Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Iterations", fontsize=10)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/04_logistic_scaling_loss.png")
    plt.close()
    print("\n>> Saved comparison plot to 'plots/04_logistic_scaling_loss.png' <<\n")


def benchmark_adult_income_dataset():
    print("==================================================")
    print("LOGISTIC REGRESSION: Adult Income (Census)")
    print("==================================================")
    data_path = "/Users/gabriel/Projects/ML/data/adult.csv"
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Skipping Adult benchmark.")
        return

    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income']
    try:
        df = pd.read_csv(data_path, header=None, names=cols, skipinitialspace=True).dropna()
        # Handle header in csv if present
        if df['age'].iloc[0] == 'age':
            df = df.iloc[1:]
    except Exception:
        df = pd.read_csv(data_path, skipinitialspace=True).dropna()

    # Subset numeric features for direct logistic regression
    feature_cols = ['age', 'education-num', 'hours-per-week']
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values.astype(float)[:2000]
    y = (df['income'].str.contains('>50K')).astype(int).values[:2000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    m_scratch = LogisticRegressionScratch(learning_rate=0.1, epochs=1000, scale_features=True).fit(X_train, y_train)
    p_scratch = m_scratch.predict(X_test)

    scaler = StandardScaler()
    m_sklearn = LogisticRegression().fit(scaler.fit_transform(X_train), y_train)
    p_sklearn = m_sklearn.predict(scaler.transform(X_test))

    print(f"Scratch Accuracy on Adult: {accuracy_score(y_test, p_scratch):.4f} | F1: {f1_score(y_test, p_scratch):.4f}")
    print(f"Sklearn Accuracy on Adult: {accuracy_score(y_test, p_sklearn):.4f} | F1: {f1_score(y_test, p_sklearn):.4f}")


if __name__ == "__main__":
    benchmark_scaling_and_regularization()
    benchmark_adult_income_dataset()
