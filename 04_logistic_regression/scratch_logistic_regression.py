"""
Logistic Regression from Scratch
================================
Mathematical Derivations and Numerical Implementation:
1. Sigmoid Activation: sigma(z) = 1 / (1 + exp(-z))
2. Binary Cross-Entropy Loss (Log-Loss)
3. Analytical Gradient: grad = (1 / n) * X^T (p - y)
4. L2 Regularization (Ridge / MAP Gaussian Prior)
5. Numerical Safeguards (Clipping to prevent exp overflow)
6. Complete Confusion Matrix and Classification Metrics from first principles
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union


def sigmoid_scratch(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    Clips inputs to [-500, 500] to prevent floating point overflow in np.exp.
    """
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


class LogisticRegressionScratch:
    """
    Binary Logistic Regression classifier with L2 regularization and gradient descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size taken in opposite direction of gradient.
    epochs : int, default=1000
        Maximum training iterations.
    l2_penalty : float, default=0.0
        Regularization coefficient lambda (MAP Gaussian prior parameter).
    scale_features : bool, default=True
        Whether to standardize features internally prior to fitting.
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        l2_penalty: float = 0.0,
        scale_features: bool = True,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.scale_features = scale_features
        
        self.weights_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X_mat = np.asarray(X, dtype=float).copy()
        y_vec = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X_mat.shape

        # Standardize features if enabled
        if self.scale_features:
            self.mean_ = np.mean(X_mat, axis=0)
            self.std_ = np.std(X_mat, axis=0)
            self.std_[self.std_ == 0.0] = 1.0
            X_mat = (X_mat - self.mean_) / self.std_

        # Add bias column (Xg = [1, X])
        X_design = np.c_[np.ones(n_samples), X_mat]
        w = np.zeros(n_features + 1)
        self.loss_history_ = []

        for _ in range(self.epochs):
            # Forward pass: log-odds linear projection -> sigmoid probability
            z = X_design @ w
            p = sigmoid_scratch(z)
            
            # Loss: Binary Cross-Entropy + L2 penalty
            bce_loss = -np.mean(y_vec * np.log(p + 1e-15) + (1.0 - y_vec) * np.log(1.0 - p + 1e-15))
            if self.l2_penalty > 0.0:
                bce_loss += (self.l2_penalty / (2.0 * n_samples)) * np.sum(w[1:] ** 2)
            self.loss_history_.append(bce_loss)

            # Gradient computation: grad = (1 / n) * X^T (p - y)
            error = p - y_vec
            grad = (X_design.T @ error) / n_samples
            
            # Regularization penalty (do not penalize intercept at index 0)
            if self.l2_penalty > 0.0:
                grad[1:] += (self.l2_penalty / n_samples) * w[1:]

            w -= self.learning_rate * grad

        self.weights_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        X_mat = np.asarray(X, dtype=float).copy()
        if self.scale_features and self.mean_ is not None and self.std_ is not None:
            X_mat = (X_mat - self.mean_) / self.std_
        X_design = np.c_[np.ones(len(X_mat)), X_mat]
        return sigmoid_scratch(X_design @ self.weights_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes TP, TN, FP, FN, Accuracy, Precision, Recall, and F1 Score from scratch."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = len(yt)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("=== Demo: Logistic Regression from Scratch on Breast Cancer ===")
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for scale in [False, True]:
        model = LogisticRegressionScratch(
            learning_rate=0.1, epochs=1500, l2_penalty=1.0, scale_features=scale
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = compute_classification_metrics(y_test, preds)
        
        print(f"\n[Feature Scaling = {scale}]")
        print(f"Accuracy:  {m['Accuracy']:.4f}")
        print(f"Precision: {m['Precision']:.4f}")
        print(f"Recall:    {m['Recall']:.4f}")
        print(f"F1 Score:  {m['F1_Score']:.4f}")
        print(f"Confusion Matrix: [TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}]")
