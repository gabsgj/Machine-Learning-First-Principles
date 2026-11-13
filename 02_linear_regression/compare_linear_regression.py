"""
Comprehensive Comparison: Linear Regression (Scratch vs. Library)
================================================================
Benchmarks:
1. Parameter convergence: Normal Equation vs. Gradient Descent vs. Sklearn OLS
2. Regularization comparison: OLS vs. Ridge vs. Lasso on multi-feature Diabetes
3. Publication plots: Fit lines and Gradient Descent Loss curve
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from scratch_linear_regression import LinearRegressionScratch, evaluate_regression


def run_single_feature_benchmark():
    print("==================================================")
    print("1. SINGLE FEATURE BENCHMARK: Diabetes (BMI -> Target)")
    print("==================================================")
    data = load_diabetes(as_frame=True)
    df = data.frame
    X = df[["bmi"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize using training statistics
    mu, sigma = X_train.mean(), X_train.std()
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    # A. Scratch Normal Equation
    ne = LinearRegressionScratch(method="normal_equation")
    ne.fit(X_train_std, y_train)
    pred_ne = ne.predict(X_test_std)
    m_ne = evaluate_regression(y_test, pred_ne)

    # B. Scratch Gradient Descent with decay
    gd = LinearRegressionScratch(method="gradient_descent", learning_rate=0.1, decay_rate=0.01, epochs=1000)
    gd.fit(X_train_std, y_train)
    pred_gd = gd.predict(X_test_std)
    m_gd = evaluate_regression(y_test, pred_gd)

    # C. Sklearn LinearRegression
    sk = LinearRegression()
    sk.fit(X_train_std, y_train)
    pred_sk = sk.predict(X_test_std)
    m_sk = {"MSE": mean_squared_error(y_test, pred_sk), "R2": r2_score(y_test, pred_sk)}

    print(f"{'Method':<25} {'Intercept':<12} {'Weight':<12} {'Test MSE':<12} {'Test R²':<10}")
    print("-" * 72)
    print(f"{'Scratch Normal Eq':<25} {ne.bias_:<12.4f} {ne.weights_[0]:<12.4f} {m_ne['MSE']:<12.3f} {m_ne['R2']:<10.4f}")
    print(f"{'Scratch Grad Descent':<25} {gd.bias_:<12.4f} {gd.weights_[0]:<12.4f} {m_gd['MSE']:<12.3f} {m_gd['R2']:<10.4f}")
    print(f"{'Sklearn OLS':<25} {sk.intercept_:<12.4f} {sk.coef_[0]:<12.4f} {m_sk['MSE']:<12.3f} {m_sk['R2']:<10.4f}")

    # Plot 1: Regression Line Comparison
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(X_test, y_test, color="#7f7f7f", alpha=0.5, label="Test Samples")
    
    x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    x_grid_std = (x_grid - mu) / sigma
    
    plt.plot(x_grid, ne.predict(x_grid_std), color="#1f77b4", lw=2, label="Scratch Normal Eq")
    plt.plot(x_grid, gd.predict(x_grid_std), color="#2ca02c", lw=2, linestyle="--", label="Scratch Grad Descent")
    plt.plot(x_grid, sk.predict(x_grid_std), color="#d62728", lw=1.5, linestyle=":", label="Sklearn LinearRegression")

    plt.title("Linear Fit Comparison: Diabetes (BMI vs. Disease Progression)", fontsize=11, fontweight="bold")
    plt.xlabel("BMI", fontsize=10)
    plt.ylabel("Progression Target", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/02_linear_regression_fit.png")
    plt.close()

    # Plot 2: Cost vs Epochs
    plt.figure(figsize=(7, 4), dpi=150)
    plt.plot(gd.cost_history_, color="#2ca02c", lw=2)
    plt.title("Cost Convergence across Epochs (Learning Rate Decay)", fontsize=11, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("MSE Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/02_gradient_descent_loss.png")
    plt.close()
    print(">> Saved comparison plots to 'plots/' <<\n")


def run_regularization_comparison():
    print("==================================================")
    print("2. MULTIVARIATE REGULARIZATION: OLS vs Ridge vs Lasso")
    print("==================================================")
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    models = {
        "OLS (Normal Eq Scratch)": LinearRegressionScratch(method="normal_equation").fit(X_train_std, y_train),
        "Ridge Scratch (lambda=10.0)": LinearRegressionScratch(method="normal_equation", l2_penalty=10.0).fit(X_train_std, y_train),
        "Ridge Sklearn (alpha=10.0)": Ridge(alpha=10.0).fit(X_train_std, y_train),
        "Lasso Sklearn (alpha=1.0)": Lasso(alpha=1.0).fit(X_train_std, y_train),
    }

    print(f"{'Model':<28} {'Test MSE':<12} {'Test R²':<10} {'Non-Zero Coefs':<15}")
    print("-" * 68)
    for name, m in models.items():
        if isinstance(m, LinearRegressionScratch):
            pred = m.predict(X_test_std)
            coefs = m.weights_
        else:
            pred = m.predict(X_test_std)
            coefs = m.coef_
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        non_zero = int(np.sum(np.abs(coefs) > 1e-4))
        print(f"{name:<28} {mse:<12.3f} {r2:<10.4f} {non_zero:<15}")


if __name__ == "__main__":
    run_single_feature_benchmark()
    run_regularization_comparison()
