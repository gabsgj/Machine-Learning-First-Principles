"""
Polynomial Regression from Scratch
==================================
Mathematical Formulation:
1. Non-linear basis expansion: phi(x) = [1, x, x^2, ..., x^d]
2. Closed-form pseudo-inverse optimization via Normal Equation: w = (Phi^T Phi)^(-1) Phi^T y
3. Generalization analysis across polynomial degrees 1 through 5
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class PolynomialRegressionScratch:
    """
    Polynomial Regression implemented from first principles.
    
    Transforms 1D or multi-dimensional features into polynomial basis representations
    and fits an Ordinary Least Squares estimator via the Normal Equation.
    """
    def __init__(self, degree: int = 2, l2_penalty: float = 0.0):
        self.degree = degree
        self.l2_penalty = l2_penalty
        self.theta_: Optional[np.ndarray] = None

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Expands input feature vectors into polynomial combinations.
        For 1D input x: [1, x, x^2, ..., x^degree].
        """
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            
        n_samples, n_features = arr.shape
        if n_features == 1:
            x_col = arr[:, 0]
            design_cols = [np.ones(n_samples)]
            for d in range(1, self.degree + 1):
                design_cols.append(x_col ** d)
            return np.column_stack(design_cols)
        else:
            # Multi-feature polynomial expansion up to power d for each feature
            design_cols = [np.ones(n_samples)]
            for d in range(1, self.degree + 1):
                for f in range(n_features):
                    design_cols.append(arr[:, f] ** d)
            return np.column_stack(design_cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressionScratch":
        Phi = self._transform_features(X)
        y_vec = np.asarray(y, dtype=float).ravel()
        
        # Regularization matrix (do not regularize bias at index 0)
        n_params = Phi.shape[1]
        reg = np.eye(n_params) * self.l2_penalty
        reg[0, 0] = 0.0

        # Normal Equation with pseudo-inverse for numerical stability
        self.theta_ = np.linalg.pinv(Phi.T @ Phi + reg) @ (Phi.T @ y_vec)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model has not been fitted.")
        Phi = self._transform_features(X)
        return Phi @ self.theta_


if __name__ == "__main__":
    import os
    print("=== Demo: Polynomial Regression from Scratch on Auto MPG ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "auto-mpg.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).dropna()
        # Ensure horsepower is numeric
        df = df[pd.to_numeric(df["horsepower"], errors="coerce").notnull()]
        x = df["horsepower"].astype(float).values
        y = df["mpg"].astype(float).values
    else:
        # Fallback synthetic non-linear data
        np.random.seed(42)
        x = np.linspace(-3, 3, 200)
        y = 2.0 - 1.5 * x + 0.8 * (x**2) - 0.2 * (x**3) + np.random.normal(0, 1, 200)

    # Train-test split (80/20)
    indices = np.random.RandomState(42).permutation(len(x))
    split = int(0.8 * len(x))
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize
    mu, sigma = x_train.mean(), x_train.std()
    x_train_std = (x_train - mu) / sigma
    x_test_std = (x_test - mu) / sigma

    for deg in [1, 2, 3, 4]:
        model = PolynomialRegressionScratch(degree=deg)
        model.fit(x_train_std, y_train)
        pred = model.predict(x_test_std)
        mse = np.mean((y_test - pred) ** 2)
        r2 = 1.0 - (np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2))
        print(f"Degree {deg} | Test MSE: {mse:.3f} | Test R²: {r2:.3f}")
