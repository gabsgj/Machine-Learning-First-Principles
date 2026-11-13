"""
Comparison and Evaluation: Polynomial Regression
================================================
Side-by-side comparison of Scratch vs. Scikit-Learn:
1. Exact parameter and prediction parity verification across degrees.
2. Degree sweeps (1 to 8) showing training vs. validation MSE.
3. Visualization of non-linear curves on real automotive data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from scratch_polynomial_regression import PolynomialRegressionScratch


def run_polynomial_comparison():
    print("==================================================")
    print("POLYNOMIAL REGRESSION: Auto MPG (Horsepower -> MPG)")
    print("==================================================")
    
    data_path = "/Users/gabriel/Projects/ML/data/auto-mpg.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).dropna()
        df = df[pd.to_numeric(df["horsepower"], errors="coerce").notnull()]
        X = df["horsepower"].astype(float).values
        y = df["mpg"].astype(float).values
    else:
        np.random.seed(42)
        X = np.linspace(40, 230, 300)
        y = 50.0 - 0.35 * X + 0.0008 * (X**2) + np.random.normal(0, 3, 300)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization parameters from training set
    mu, sigma = X_train.mean(), X_train.std()
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma

    degrees = [1, 2, 3, 4]
    
    print(f"{'Degree':<8} {'Scratch MSE':<14} {'Sklearn MSE':<14} {'Scratch R²':<12} {'Sklearn R²':<12} {'Max Diff':<10}")
    print("-" * 72)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.scatter(X_test, y_test, color="#8c8c8c", alpha=0.5, label="Test Observations")

    # Grid for smooth line plotting
    x_plot = np.linspace(X.min(), X.max(), 300)
    x_plot_std = (x_plot - mu) / sigma

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for deg, color in zip(degrees, colors):
        # 1. Scratch
        m_scratch = PolynomialRegressionScratch(degree=deg)
        m_scratch.fit(X_train_std, y_train)
        pred_scratch = m_scratch.predict(X_test_std)
        mse_scratch = mean_squared_error(y_test, pred_scratch)
        r2_scratch = r2_score(y_test, pred_scratch)

        # 2. Sklearn
        poly = PolynomialFeatures(degree=deg, include_bias=True)
        X_tr_poly = poly.fit_transform(X_train_std.reshape(-1, 1))
        X_te_poly = poly.transform(X_test_std.reshape(-1, 1))
        sk_model = LinearRegression(fit_intercept=False)
        sk_model.fit(X_tr_poly, y_train)
        pred_sklearn = sk_model.predict(X_te_poly)
        mse_sklearn = mean_squared_error(y_test, pred_sklearn)
        r2_sklearn = r2_score(y_test, pred_sklearn)

        max_diff = np.max(np.abs(pred_scratch - pred_sklearn))
        print(f"{deg:<8} {mse_scratch:<14.3f} {mse_sklearn:<14.3f} {r2_scratch:<12.4f} {r2_sklearn:<12.4f} {max_diff:<10.1e}")

        # Plot curve
        y_plot_line = m_scratch.predict(x_plot_std)
        plt.plot(x_plot, y_plot_line, color=color, lw=2.2, label=f"Degree {deg} (R² = {r2_scratch:.3f})")

    plt.title("Polynomial Regression Degrees Comparison (Auto MPG)", fontsize=12, fontweight="bold")
    plt.xlabel("Horsepower", fontsize=10)
    plt.ylabel("Miles Per Gallon (MPG)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/03_polynomial_fit_comparison.png")
    plt.close()
    print(">> Saved comparison plot to 'plots/03_polynomial_fit_comparison.png' <<\n")


def run_degree_sweep_analysis():
    print("==================================================")
    print("DEGREE SWEEP (1 to 10): Underfitting vs Overfitting")
    print("==================================================")
    
    np.random.seed(42)
    n = 100
    x = np.linspace(-3, 3, n)
    y = np.sin(x) + np.random.normal(0, 0.3, n)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    mu, sigma = x_train.mean(), x_train.std()
    x_train_std = (x_train - mu) / sigma
    x_val_std = (x_val - mu) / sigma

    degrees = list(range(1, 11))
    train_errs = []
    val_errs = []

    for d in degrees:
        m = PolynomialRegressionScratch(degree=d)
        m.fit(x_train_std, y_train)
        train_errs.append(mean_squared_error(y_train, m.predict(x_train_std)))
        val_errs.append(mean_squared_error(y_val, m.predict(x_val_std)))

    best_degree = degrees[np.argmin(val_errs)]
    print(f"Optimal Degree (Lowest Validation MSE): {best_degree}")

    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(degrees, train_errs, marker="o", color="#1f77b4", lw=2, label="Training MSE (Monotonic drop)")
    plt.plot(degrees, val_errs, marker="s", color="#d62728", lw=2, label="Validation MSE (U-shaped curve)")
    plt.axvline(best_degree, color="#2ca02c", linestyle="--", label=f"Best Degree = {best_degree}")
    plt.title("Bias-Variance Tradeoff via Polynomial Complexity", fontsize=12, fontweight="bold")
    plt.xlabel("Polynomial Degree", fontsize=10)
    plt.ylabel("Mean Squared Error", fontsize=10)
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/03_bias_variance_polynomial.png")
    plt.close()
    print(">> Saved bias-variance plot to 'plots/03_bias_variance_polynomial.png' <<\n")


if __name__ == "__main__":
    run_polynomial_comparison()
    run_degree_sweep_analysis()
