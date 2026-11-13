"""
Syllabus Experiment 20: Bias-Variance Tradeoff Analysis
======================================================
Direct execution of Experiment 20:
1. Polynomial regression complexity sweeps across degrees 1 to 10 on Diabetes BMI.
2. Tracking Training Error (Bias reduction) vs. Validation Error (Variance explosion).
3. Automated identification of the optimal complexity sweet-spot.
4. Publication-ready U-curve error plot.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def run_experiment_20():
    print("==================================================")
    print("EXP 20: Bias-Variance Tradeoff (Diabetes BMI)")
    print("==================================================")
    # 1. Load data
    data = load_diabetes(as_frame=True)
    X = data.frame[['bmi']]
    y = data.frame['target']

    # 2. Split data (70:30)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_err = []
    val_err = []
    degrees = list(range(1, 11))

    print(f"{'Degree':<8} {'Training MSE':<18} {'Validation MSE':<18}")
    print("-" * 46)

    # 3. Evaluate polynomial degrees 1 to 10
    for d in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=d),
            LinearRegression()
        )
        model.fit(X_train, y_train)

        tr_mse = mean_squared_error(y_train, model.predict(X_train))
        va_mse = mean_squared_error(y_val, model.predict(X_val))

        train_err.append(tr_mse)
        val_err.append(va_mse)

        print(f"{d:<8} {tr_mse:<18.2f} {va_mse:<18.2f}")

    # 4. Find best degree
    best = int(np.argmin(val_err)) + 1
    print(f"\n>> Optimal Polynomial Degree: {best} (Minimum Validation MSE: {val_err[best-1]:.2f}) <<")

    # 5. Plot Bias-Variance Tradeoff (reproducing exact WhatsApp EXP 20 visual)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(degrees, train_err, marker='o', lw=2, color="#1f77b4", label='Training Error (Bias^2 drop)')
    plt.plot(degrees, val_err, marker='s', lw=2, color="#d62728", label='Validation Error (Total MSE)')
    plt.axvline(best, color='#2ca02c', linestyle='--', lw=1.8, label=f'Optimal Complexity (Degree {best})')

    plt.xlabel('Polynomial Degree (Model Complexity)', fontsize=10, fontweight="bold")
    plt.ylabel('Mean Squared Error (MSE)', fontsize=10, fontweight="bold")
    plt.title('Experiment 20: Bias-Variance Tradeoff Curve', fontsize=12, fontweight="bold")
    plt.xticks(degrees)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/14_bias_variance_curve.png')
    plt.close()
    print(">> Saved Bias-Variance curve plot to 'plots/14_bias_variance_curve.png' <<\n")


if __name__ == "__main__":
    run_experiment_20()
