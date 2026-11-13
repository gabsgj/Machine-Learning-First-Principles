"""
Experiment 01: Linear Regression with One Variable
==================================================
Dataset: California Housing (AveRooms -> MedHouseVal)
Compares:
1. Batch Gradient Descent from scratch
2. Matrix Normal Equation from scratch
3. Production Scikit-Learn LinearRegression
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Ingestion & Outlier Filtering
df = fetch_california_housing(as_frame=True).frame
df = df[df['AveRooms'] < 10]

X = df[['AveRooms']]
y = df['MedHouseVal']

# 2. Partitioning (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x = X_train['AveRooms'].to_numpy()
t = y_train.to_numpy()
n = len(x)

# 3a. Gradient Descent Optimization
w, b = 0.0, 0.0
lr = 0.01
for _ in range(5000):
    error = w * x + b - t
    w -= lr * (2 / n) * np.sum(error * x)
    b -= lr * (2 / n) * np.sum(error)

print("=== Experiment 01 Results ===")
print(f"Gradient Descent -> Intercept: {b:.4f} | Slope: {w:.4f}")

# 3b. Normal Equation: theta = (X^T X)^(-1) X^T y
X_b = np.c_[np.ones(n), x]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ t
print(f"Normal Equation  -> Intercept: {theta[0]:.4f} | Slope: {theta[1]:.4f}")

# 4. Scikit-Learn Estimator & Metrics
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Scikit-Learn OLS -> Intercept: {model.intercept_:.4f} | Slope: {model.coef_[0]:.4f}")
print(f"Test MSE: {mean_squared_error(y_test, pred):.4f}")
print(f"Test R2:  {r2_score(y_test, pred):.4f}")

# 5. Visualization
plt.figure(figsize=(7, 4.5), dpi=150)
plt.scatter(x[:250], t[:250], alpha=0.3, label='Observed Data')
xs = np.linspace(x.min(), x.max(), 100)
plt.plot(xs, model.intercept_ + model.coef_[0] * xs, color='#d62728', lw=2, label='Fitted Line')
plt.xlabel('Average Rooms')
plt.ylabel('Median House Value ($100k)')
plt.title('Experiment 01: Univariate Linear Regression (California Housing)')
plt.legend()
plt.tight_layout()
plt.savefig('exp01_linear_regression.png')
print("Saved visualization to exp01_linear_regression.png")
