"""
Linear Regression with Scikit-Learn
===================================
Production implementations of linear models:
1. Ordinary Least Squares (LinearRegression)
2. L2 Regularization (Ridge)
3. L1 Regularization (Lasso)
4. Stochastic Gradient Descent (SGDRegressor)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def build_and_evaluate_linear_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Fits and compares Scikit-Learn OLS, Ridge, Lasso, and SGDRegressor.
    """
    models = {
        "OLS LinearRegression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=0.1)": Lasso(alpha=0.1),
        "SGDRegressor (lr=0.01)": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        reg = pipe.named_steps["regressor"]
        intercept = reg.intercept_
        coef = reg.coef_

        results[name] = {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mae,
            "R2": r2,
            "Intercept": intercept if np.isscalar(intercept) else intercept[0],
            "Coefficients": coef,
        }

    return results


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    print("=== Demo: Scikit-Learn Linear Models on Diabetes BMI ===")
    data = load_diabetes(as_frame=True)
    X = data.frame[["bmi"]].values
    y = data.frame["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = build_and_evaluate_linear_models(X_train, y_train, X_test, y_test)
    for model_name, res in metrics.items():
        print(f"\n--- {model_name} ---")
        print(f"Intercept: {res['Intercept']:.3f}")
        print(f"Coef:      {res['Coefficients'][0]:.3f}")
        print(f"MSE:       {res['MSE']:.3f}")
        print(f"R²:        {res['R2']:.3f}")
