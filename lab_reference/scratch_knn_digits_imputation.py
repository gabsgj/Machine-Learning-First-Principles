"""
Core Scratch: K-Nearest Neighbors Classifier with Missing Value Imputation
==========================================================================
First-principles implementation featuring:
1. Synthetic missing value generation and column-wise imputation
2. Permutation-based uniform train-test split from scratch
3. Euclidean (L2) and Manhattan (L1) metric functions
4. Multi-class majority voting across varying neighbor counts K
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

# 1. Dataset Ingestion
data = load_digits()
df = pd.DataFrame(data.data)
y = data.target

# 2. Random Missingness Simulation
np.random.seed(42)
for _ in range(20):
    r = np.random.randint(len(df))
    c = np.random.randint(df.shape[1])
    df.iat[r, c] = np.nan

# 3. Univariate Imputation
df = df.fillna(df.mean())
X = df.values

# 4. Train-Test Partitioning from Scratch (80:20)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# 5. Distance Metric Formulations
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

# 6. KNN Inference Engine
def knn_predict(X_tr, y_tr, X_te, k, distance_metric):
    predictions = []
    for x in X_te:
        distances = [distance_metric(x, t) for t in X_tr]
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_tr[nearest_idx]
        predictions.append(np.bincount(nearest_labels).argmax())
    return np.array(predictions)

# 7. Benchmarking across Metrics and K Values
print("=== KNN Scratch Evaluation on Digits ===")
for metric in [euclidean, manhattan]:
    for k in [1, 3, 5]:
        pred = knn_predict(X_train, y_train, X_test, k, metric)
        acc = np.mean(pred == y_test) * 100
        print(f"Metric: {metric.__name__.capitalize():<10} | K = {k} | Accuracy = {acc:.2f}%")
