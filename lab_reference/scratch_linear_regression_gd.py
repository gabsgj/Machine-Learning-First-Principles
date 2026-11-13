"""
Core Scratch: Univariate Linear Regression with Gradient Descent & LR Decay
===========================================================================
First-principles implementation of Ordinary Least Squares via:
1. Batch Gradient Descent with hyperbolic learning rate decay
2. Matrix Closed-Form Normal Equation: theta = (X^T X)^(-1) X^T y
3. Generalization metrics: Mean Squared Error (MSE) and R^2 score
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. Dataset Ingestion
data = load_diabetes(as_frame=True)
df = data.frame
X = df[['bmi']].values
y = df['target'].values

# 2. Partitioning and Feature Standardization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 3. Augmented Design Matrix (Prepending Bias Column)
Xg_train = np.c_[np.ones(len(X_train)), X_train]
Xg_test = np.c_[np.ones(len(X_test)), X_test]

# 4. Batch Gradient Descent Optimization
w = np.zeros(2)
epochs = 1000
costs = []

for i in range(epochs):
    lr = 0.1 / (1 + 0.01 * i)  # Learning rate schedule
    err = Xg_train @ w - y_train
    grad = Xg_train.T @ err / len(y_train)
    w -= lr * grad
    costs.append(np.mean((Xg_train @ w - y_train) ** 2))

# 5. Analytical Normal Equation Verification
theta_ne = np.linalg.inv(Xg_train.T @ Xg_train) @ Xg_train.T @ y_train

# 6. Out-of-Sample Evaluation
y_pred = Xg_test @ w
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print("=== Gradient Descent Solution ===")
print("Bias (w0):  ", round(w[0], 4))
print("Slope (w1): ", round(w[1], 4))
print("Test MSE:   ", round(mse, 4))
print("R2 Score:   ", round(r2, 4))

print("\n=== Normal Equation Solution ===")
print("Bias (theta0): ", round(theta_ne[0], 4))
print("Slope (theta1):", round(theta_ne[1], 4))

# 7. Loss Convergence Plot
plt.figure(figsize=(7, 4), dpi=150)
plt.plot(costs, color='#1f77b4', lw=2)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Linear Regression: Cost Convergence under LR Decay")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("linear_regression_cost_convergence.png")
print("\nGenerated plot: linear_regression_cost_convergence.png")
