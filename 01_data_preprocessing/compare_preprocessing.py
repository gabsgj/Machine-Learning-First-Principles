"""
Verification and Comparison: Preprocessing (Scratch vs. Library)
===============================================================
Demonstrates:
1. Exact mathematical parity between scratch and scikit-learn transformers.
2. The catastrophic failure of unscaled gradient descent vs. standardized data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from scratch_preprocessing import (
    MissingValueHandlerScratch,
    FeatureScalerScratch,
)


def verify_numerical_parity():
    print("--- 1. Testing Numerical Parity on Real Dataset ---")
    data = load_breast_cancer()
    X = data.data[:50, :5].copy()  # first 50 rows, 5 features

    # Inject 15 random NaNs
    np.random.seed(42)
    for _ in range(15):
        r = np.random.randint(0, X.shape[0])
        c = np.random.randint(0, X.shape[1])
        X[r, c] = np.nan

    # Imputation: Mean
    imputer_scratch = MissingValueHandlerScratch(strategy="mean")
    X_imputed_scratch = imputer_scratch.fit_transform(X)

    imputer_sklearn = SimpleImputer(strategy="mean")
    X_imputed_sklearn = imputer_sklearn.fit_transform(X)

    diff_impute = np.max(np.abs(X_imputed_scratch - X_imputed_sklearn))
    print(f"Max absolute difference in Mean Imputation: {diff_impute:.2e}")
    assert diff_impute < 1e-12, "Imputation mismatch!"

    # Scaling: StandardScaler
    scaler_scratch = FeatureScalerScratch(method="standard")
    X_scaled_scratch = scaler_scratch.fit_transform(X_imputed_scratch)

    scaler_sklearn = StandardScaler()
    X_scaled_sklearn = scaler_sklearn.fit_transform(X_imputed_sklearn)

    diff_scale = np.max(np.abs(X_scaled_scratch - X_scaled_sklearn))
    print(f"Max absolute difference in StandardScaler:   {diff_scale:.2e}")
    assert diff_scale < 1e-12, "StandardScaler mismatch!"
    print(">> Parity check PASSED perfectly! <<\n")


def plot_scaling_impact_on_gradient_descent():
    print("--- 2. Visualizing Impact of Feature Scaling on Optimization ---")
    data = load_breast_cancer()
    X = data.data[:, [0, 1]]  # mean radius, mean texture (widely different scales)
    y = data.target

    # Unscaled vs Scaled
    scaler = FeatureScalerScratch(method="standard")
    X_scaled = scaler.fit_transform(X)

    # Simple Logistic Regression loss tracker
    def run_gd(X_in, lr, steps=100):
        X_design = np.c_[np.ones(len(X_in)), X_in]
        w = np.zeros(X_design.shape[1])
        losses = []
        for _ in range(steps):
            z = np.clip(X_design @ w, -250, 250)
            p = 1 / (1 + np.exp(-z))
            # Binary Cross Entropy Loss
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            losses.append(loss)
            grad = X_design.T @ (p - y) / len(y)
            w -= lr * grad
        return losses

    losses_unscaled = run_gd(X, lr=0.0001, steps=80)
    losses_scaled = run_gd(X_scaled, lr=0.5, steps=80)

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(losses_unscaled, label="Unscaled Features (lr=0.0001, sluggish progress)", color="#d62728", lw=2)
    plt.plot(losses_scaled, label="Standardized Features (lr=0.5, rapid convergence)", color="#2ca02c", lw=2)
    plt.title("Optimization Trajectory: Scaled vs. Unscaled Gradient Descent", fontsize=12, fontweight="bold")
    plt.xlabel("Iterations", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/01_scaling_impact.png")
    plt.close()
    print("Saved plot to 'plots/01_scaling_impact.png'.")


if __name__ == "__main__":
    verify_numerical_parity()
    plot_scaling_impact_on_gradient_descent()
