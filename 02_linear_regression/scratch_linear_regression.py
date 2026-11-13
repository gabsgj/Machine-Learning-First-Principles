"""
Linear Regression from Scratch
==============================
Mathematical Derivations and Numerical Implementation:
1. Gradient Descent with Learning Rate Decay
2. Ordinary Least Squares (OLS) via Normal Equation: w = (X^T X)^(-1) X^T y
3. Ridge Regression (L2 Regularization) Analytical Solution
4. Performance Metrics: MSE, RMSE, MAE, R² Score
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


class LinearRegressionScratch:
    """
    Linear Regression implemented from first mathematical principles.
    
    Supports:
    - Optimization via Batch Gradient Descent with learning rate decay
    - Exact analytical solution via the Normal Equation
    - L2 Regularization (Ridge)
    """
    def __init__(
        self,
        method: str = "gradient_descent",
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        epochs: int = 1000,
        l2_penalty: float = 0.0,
        random_state: Optional[int] = 42,
    ):
        assert method in ["gradient_descent", "normal_equation"], f"Unknown method {method}"
        self.method = method
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.random_state = random_state
        
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.cost_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_arr.shape

        # Construct Design Matrix with Bias Column: Xg = [1, X]
        X_design = np.c_[np.ones(n_samples), X_arr]

        if self.method == "normal_equation":
            # Normal Equation: w = (X^T X + lambda * I)^(-1) X^T y
            # We do not regularize the intercept term (index 0)
            reg_matrix = np.eye(n_features + 1) * self.l2_penalty
            reg_matrix[0, 0] = 0.0
            
            theta = np.linalg.pinv(X_design.T @ X_design + reg_matrix) @ (X_design.T @ y_arr)
            self.bias_ = float(theta[0])
            self.weights_ = theta[1:]
            
            # Record final cost
            y_pred = X_design @ theta
            mse = np.mean((y_arr - y_pred) ** 2)
            self.cost_history_ = [mse]

        elif self.method == "gradient_descent":
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            theta = np.zeros(n_features + 1)
            self.cost_history_ = []

            for epoch in range(self.epochs):
                # Learning rate schedule with decay: lr = lr0 / (1 + decay * epoch)
                current_lr = self.learning_rate / (1.0 + self.decay_rate * epoch)
                
                # Predictions and residuals
                y_hat = X_design @ theta
                error = y_hat - y_arr
                
                # Gradient of MSE + L2 penalty: grad = (2 / n) * X_design^T (X_design @ theta - y)
                grad = (2.0 / n_samples) * (X_design.T @ error)
                if self.l2_penalty > 0.0:
                    reg_grad = 2.0 * self.l2_penalty * theta
                    reg_grad[0] = 0.0  # Do not regularize bias
                    grad += reg_grad
                
                # Gradient descent step
                theta -= current_lr * grad
                
                # Calculate current MSE cost
                current_cost = np.mean((X_design @ theta - y_arr) ** 2)
                self.cost_history_.append(current_cost)

            self.bias_ = float(theta[0])
            self.weights_ = theta[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_arr = np.asarray(X, dtype=float)
        return (X_arr @ self.weights_) + self.bias_


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates regression performance metrics from first principles."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    print("=== Demo: Linear Regression from Scratch on Diabetes BMI ===")
    data = load_diabetes(as_frame=True)
    df = data.frame
    X = df[["bmi"]].values
    y = df["target"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize
    mean_x, std_x = X_train.mean(), X_train.std()
    X_train_std = (X_train - mean_x) / std_x
    X_test_std = (X_test - mean_x) / std_x

    # 1. Gradient Descent with Learning Rate Decay
    gd_model = LinearRegressionScratch(
        method="gradient_descent", learning_rate=0.1, decay_rate=0.01, epochs=1000
    )
    gd_model.fit(X_train_std, y_train)
    y_pred_gd = gd_model.predict(X_test_std)
    metrics_gd = evaluate_regression(y_test, y_pred_gd)

    print(f"\n[Gradient Descent]")
    print(f"Bias (Intercept): {gd_model.bias_:.3f}")
    print(f"Weight (Slope):   {gd_model.weights_[0]:.3f}")
    print(f"MSE:  {metrics_gd['MSE']:.3f}")
    print(f"RMSE: {metrics_gd['RMSE']:.3f}")
    print(f"R²:   {metrics_gd['R2']:.3f}")

    # 2. Normal Equation (Exact Solution)
    ne_model = LinearRegressionScratch(method="normal_equation")
    ne_model.fit(X_train_std, y_train)
    y_pred_ne = ne_model.predict(X_test_std)
    metrics_ne = evaluate_regression(y_test, y_pred_ne)

    print(f"\n[Normal Equation]")
    print(f"Bias (Intercept): {ne_model.bias_:.3f}")
    print(f"Weight (Slope):   {ne_model.weights_[0]:.3f}")
    print(f"MSE:  {metrics_ne['MSE']:.3f}")
    print(f"RMSE: {metrics_ne['RMSE']:.3f}")
    print(f"R²:   {metrics_ne['R2']:.3f}")
