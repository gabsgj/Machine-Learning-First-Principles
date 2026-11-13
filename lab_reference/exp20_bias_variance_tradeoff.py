"""
Experiment 20: Bias-Variance Tradeoff Analysis with Polynomial Complexity
=========================================================================
Dataset: Diabetes (BMI Feature)
Demonstrates underfitting vs. overfitting via the U-shaped validation error curve.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 1. Dataset Ingestion & Partitioning (70:30)
X, y = load_diabetes(return_X_y=True, as_frame=True)
X = X[['bmi']]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Complexity Sweeps across Degrees 1 to 10
train_err = []
val_err = []
degrees = list(range(1, 11))

print("=== Experiment 20: Bias-Variance Tradeoff Analysis ===")
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
    print(f"Degree {d:<2} | Training MSE: {tr_mse:.2f} | Validation MSE: {va_mse:.2f}")

# 3. Sweet Spot Identification
best_degree = int(np.argmin(val_err)) + 1
print(f"\nOptimal Polynomial Complexity: Degree {best_degree} (Lowest Validation Error: {val_err[best_degree-1]:.2f})")

# 4. Error Trajectory Plot
plt.figure(figsize=(7.5, 4.8), dpi=150)
plt.plot(degrees, train_err, marker='o', lw=2, color='#1f77b4', label='Training Error (Bias^2 drop)')
plt.plot(degrees, val_err, marker='s', lw=2, color='#d62728', label='Validation Error (Variance jump)')
plt.axvline(best_degree, color='#2ca02c', linestyle='--', lw=1.8, label=f'Optimal Complexity (Degree {best_degree})')
plt.xlabel('Polynomial Degree (Model Complexity)')
plt.ylabel('Mean Squared Error')
plt.title('Experiment 20: Bias-Variance Tradeoff Curve')
plt.xticks(degrees)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('exp20_bias_variance.png')
print("Saved visualization to exp20_bias_variance.png")
