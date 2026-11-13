"""
K-Nearest Neighbors (KNN) from Scratch
======================================
Vectorized distance metrics and neighbor aggregation:
1. Distance functions: Euclidean (L2), Manhattan (L1), and Minkowski (Lp)
2. Uniform and Inverse-Distance Weighted Majority Voting
3. Multi-Class classification with np.bincount tie-breaking
4. Full support for raw NumPy arrays
"""

import numpy as np
from typing import Callable, Optional, Union, List


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance (L2 norm): sqrt(sum((a - b)^2))"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan distance (L1 norm): sum(|a - b|)"""
    return float(np.sum(np.abs(a - b)))


def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3.0) -> float:
    """Minkowski distance: (sum(|a - b|^p))^(1/p)"""
    return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))


class KNNClassifierScratch:
    """
    K-Nearest Neighbors classifier implemented from scratch.
    
    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors to query.
    metric : str or callable, default='euclidean'
        Distance metric ('euclidean', 'manhattan', or custom function).
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance').
    """
    def __init__(
        self,
        k: int = 3,
        metric: Union[str, Callable] = "euclidean",
        weights: str = "uniform",
    ):
        self.k = k
        self.weights = weights
        if isinstance(metric, str):
            if metric == "euclidean":
                self.metric_func = euclidean_distance
            elif metric == "manhattan":
                self.metric_func = manhattan_distance
            else:
                raise ValueError(f"Unknown metric {metric}")
        else:
            self.metric_func = metric
            
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierScratch":
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int).ravel()
        return self

    def _predict_single(self, x: np.ndarray) -> int:
        # Compute distance to all training samples
        distances = np.array([self.metric_func(x, t) for t in self.X_train_])
        
        # Identify top-k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train_[k_indices]
        
        if self.weights == "uniform":
            # Majority vote
            return int(np.bincount(k_labels).argmax())
        elif self.weights == "distance":
            k_dists = distances[k_indices]
            weights = 1.0 / (k_dists + 1e-10)
            unique_classes = np.unique(self.y_train_)
            weighted_votes = np.zeros(np.max(unique_classes) + 1)
            for label, w in zip(k_labels, weights):
                weighted_votes[label] += w
            return int(np.argmax(weighted_votes))
        else:
            raise ValueError(f"Unknown weights option {self.weights}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X_arr])


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import pandas as pd

    print("=== Demo: KNN from Scratch with Missing Values (Digits Dataset) ===")
    data = load_digits()
    df = pd.DataFrame(data.data)
    y = data.target

    # Inject 20 random NaNs as specified in lab exam
    np.random.seed(42)
    for _ in range(20):
        r = np.random.randint(len(df))
        c = np.random.randint(df.shape[1])
        df.iat[r, c] = np.nan

    # Impute missing values with column mean
    df = df.fillna(df.mean())
    X = df.values

    # Train-test split (80/20) from scratch
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    for metric_name, m_func in [("Euclidean", euclidean_distance), ("Manhattan", manhattan_distance)]:
        for k in [1, 3, 5]:
            clf = KNNClassifierScratch(k=k, metric=m_func)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = float(np.mean(preds == y_test) * 100)
            print(f"Metric: {metric_name:<10} | K = {k} | Accuracy = {acc:.2f}%")
