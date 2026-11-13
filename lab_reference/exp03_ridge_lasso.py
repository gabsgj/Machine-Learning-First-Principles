"""
Experiment 03: Ridge (L2) and Lasso (L1) Regularization
======================================================
Dataset: Scikit-Learn Diabetes (All Features)
Demonstrates:
1. Feature shrinkage via Ridge penalty (L2)
2. Sparsity induction and feature selection via Lasso penalty (L1)
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Ingestion & Partitioning
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline Configuration with Feature Standardization
models = {
    'OLS Linear': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge (L2)': make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)),
    'Lasso (L1)': make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=10000)),
}

# 3. Model Fitting and Coefficient Sparsity Inspection
print("=== Experiment 03: Regularization Comparison ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    coef = model[-1].coef_
    zero_weights = (abs(coef) < 1e-4).sum()
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:<12} | Test MSE: {mse:.3f} | R2: {r2:.4f} | Zero Coefficients: {zero_weights}")
