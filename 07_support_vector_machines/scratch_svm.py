"""
Support Vector Machine (SVM) from Scratch
=========================================
Mathematical Implementation:
1. Primal Soft-Margin Objective with Hinge Loss:
   min (1/2)||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
2. Subgradient Descent Optimization (Pegasos Formulation)
3. Support Vector Identification: samples lying on or inside the margin
4. Evaluation and Margin calculation: Margin = 2 / ||w||
"""

import numpy as np
from typing import Optional, Tuple


class LinearSVMScratch:
    """
    Linear Support Vector Classifier solved via Hinge-Loss Subgradient Descent.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization penalty parameter (trades off margin maximization against training error).
    learning_rate : float, default=0.001
        Step size for subgradient descent updates.
    epochs : int, default=1000
        Number of training iterations through the dataset.
    """
    def __init__(self, C: float = 1.0, learning_rate: float = 0.001, epochs: int = 1000):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_vectors_: Optional[np.ndarray] = None
        self.loss_history_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        X_arr = np.asarray(X, dtype=float)
        # Convert {0, 1} binary labels to {-1, +1} SVM target format
        y_arr = np.asarray(y, dtype=float).ravel()
        y_svm = np.where(y_arr <= 0, -1.0, 1.0)
        
        n_samples, n_features = X_arr.shape
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        for epoch in range(1, self.epochs + 1):
            # Dynamic decaying learning rate: lr = lr0 / sqrt(t)
            eta = self.learning_rate / np.sqrt(epoch)
            
            # Stochastic Subgradient Descent across samples
            indices = np.random.permutation(n_samples)
            for idx in indices:
                xi = X_arr[idx]
                yi = y_svm[idx]
                
                # Check functional margin condition: y_i * (w^T x_i + b) >= 1
                condition = yi * (np.dot(self.w_, xi) + self.b_) >= 1.0
                
                if condition:
                    # Point is strictly outside margin on the correct side
                    # Gradient only comes from the weight decay (1/2 ||w||^2)
                    self.w_ -= eta * (self.w_ / n_samples)
                else:
                    # Point violates margin or is misclassified: Hinge loss is active
                    self.w_ -= eta * ((self.w_ / n_samples) - self.C * yi * xi)
                    self.b_ -= eta * (-self.C * yi)

            # Record total loss
            margins = y_svm * (X_arr @ self.w_ + self.b_)
            hinge_loss = np.sum(np.maximum(0, 1.0 - margins))
            total_loss = 0.5 * np.dot(self.w_, self.w_) + self.C * hinge_loss
            self.loss_history_.append(total_loss)

        # Identify Support Vectors: instances satisfying y_i * (w^T x_i + b) <= 1 + eps
        functional_margins = y_svm * (X_arr @ self.w_ + self.b_)
        sv_mask = functional_margins <= 1.05
        self.support_vectors_ = X_arr[sv_mask]
        
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model is not fitted.")
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=== Demo: Linear SVM from Scratch on Iris (Setosa vs Non-Setosa) ===")
    iris = load_iris()
    # Features: Petal length (col 2), Petal width (col 3)
    X = iris.data[:, [2, 3]]
    y = (iris.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    svm = LinearSVMScratch(C=10.0, learning_rate=0.01, epochs=800)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)

    acc = np.mean(preds == y_test) * 100
    norm_w = np.linalg.norm(svm.w_)
    margin_width = 2.0 / norm_w if norm_w > 0 else 0.0

    print(f"Learned Weights w:   {np.round(svm.w_, 4)}")
    print(f"Learned Bias b:      {svm.b_:.4f}")
    print(f"Margin Width (2/|w|):{margin_width:.4f}")
    print(f"Support Vectors:     {len(svm.support_vectors_)} found")
    print(f"Testing Accuracy:    {acc:.2f}%")
