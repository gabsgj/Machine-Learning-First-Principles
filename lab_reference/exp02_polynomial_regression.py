"""
Experiment 02: Polynomial Regression and Complexity Comparison
==============================================================
Dataset: California Housing (MedInc -> MedHouseVal)
Compares polynomial basis expansions across degrees 1, 2, and 3.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Ingestion
df = fetch_california_housing(as_frame=True).frame
X = df[['MedInc']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Polynomial Degree Sweeps (1, 2, 3)
print("=== Experiment 02: Degree Complexity Analysis ===")
for d in [1, 2, 3]:
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"Polynomial Degree {d} | Test MSE: {mse:.4f} | Test R2: {r2:.4f}")

# 3. Non-Linear Curve Visualization
best_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
best_model.fit(X_train, y_train)

xs = pd.DataFrame({'MedInc': np.linspace(X['MedInc'].min(), X['MedInc'].max(), 100)})
plt.figure(figsize=(7, 4.5), dpi=150)
plt.scatter(X_test['MedInc'][:300], y_test[:300], alpha=0.3, label='Observed Data')
plt.plot(xs['MedInc'], best_model.predict(xs), color='#d62728', lw=2.2, label='Degree 2 Polynomial Fit')
plt.xlabel('Median Income')
plt.ylabel('Median House Value ($100k)')
plt.title('Experiment 02: Polynomial Regression Analysis')
plt.legend()
plt.tight_layout()
plt.savefig('exp02_polynomial_regression.png')
print("Saved visualization to exp02_polynomial_regression.png")
