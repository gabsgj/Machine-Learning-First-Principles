"""
Polynomial Regression with Scikit-Learn
=======================================
Pipeline-based polynomial regression with automated degree selection via cross-validation:
- PolynomialFeatures
- LinearRegression and Ridge
- Pipeline encapsulation
- GridSearchCV for hyperparameter optimization
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def build_polynomial_pipeline(degree: int = 2, alpha: float = 0.0):
    """Creates a standardized polynomial regression pipeline."""
    if alpha > 0.0:
        reg = Ridge(alpha=alpha)
    else:
        reg = LinearRegression()
        
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        reg
    )


def find_best_degree_cv(X_train: np.ndarray, y_train: np.ndarray, max_degree: int = 6):
    """Performs K-Fold cross validation to determine optimal polynomial degree."""
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )
    param_grid = {"polynomialfeatures__degree": list(range(1, max_degree + 1))}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)
    return grid.best_params_["polynomialfeatures__degree"], -grid.best_score_


if __name__ == "__main__":
    import os
    print("=== Demo: Scikit-Learn Polynomial Regression with Cross-Validation ===")
    
    data_path = "/Users/gabriel/Projects/ML/data/auto-mpg.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).dropna()
        df = df[pd.to_numeric(df["horsepower"], errors="coerce").notnull()]
        X = df[["horsepower"]].astype(float).values
        y = df["mpg"].astype(float).values
    else:
        np.random.seed(42)
        X = np.linspace(-3, 3, 200).reshape(-1, 1)
        y = 2.0 - 1.5 * X.ravel() + 0.8 * (X.ravel()**2) + np.random.normal(0, 1, 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_deg, cv_mse = find_best_degree_cv(X_train, y_train, max_degree=6)
    print(f"Optimal Polynomial Degree identified by 5-Fold CV: {best_deg} (CV MSE: {cv_mse:.3f})")

    # Fit best model
    best_model = build_polynomial_pipeline(degree=best_deg)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"Test R²:  {r2_score(y_test, y_pred):.3f}")
