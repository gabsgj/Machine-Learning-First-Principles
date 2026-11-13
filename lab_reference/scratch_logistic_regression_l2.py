"""
Core Scratch: Binary Logistic Regression with L2 Regularization & Diagnostic Metrics
===================================================================================
First-principles implementation featuring:
1. Numerically stabilized Sigmoid activation: np.clip(z, -500, 500)
2. Gradient Descent with L2 Ridge Penalty (Gaussian MAP Prior)
3. Confusion matrix metrics from scratch: TP, TN, FP, FN, Accuracy, Precision, Recall, F1
4. Empirical demonstration of feature scaling impact
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. Dataset Ingestion
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop(columns='target').values
y = df['target'].values

# 2. Partitioning from Scratch (80:20)
np.random.seed(42)
idx = np.random.permutation(len(X))
s = int(0.8 * len(X))
X_train, X_test = X[idx[:s]], X[idx[s:]]
y_train, y_test = y[idx[:s]], y[idx[s:]]

# 3. Model Training
def train_logistic(X, y, scale: bool, lam: float = 0.01):
    if scale:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

    X_design = np.c_[np.ones(len(X)), X]
    w = np.zeros(X_design.shape[1])

    for _ in range(2000):
        z = np.clip(X_design @ w, -500.0, 500.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X_design.T @ (p - y) / len(y)
        grad[1:] += (lam * w[1:]) / len(y)  # L2 Regularization
        w -= 0.1 * grad

    return w, (mean if scale else None), (std if scale else None)

# 4. Inference
def predict_logistic(X, w, mean=None, std=None):
    if mean is not None:
        X = (X - mean) / std
    X_design = np.c_[np.ones(len(X)), X]
    p = 1.0 / (1.0 + np.exp(-np.clip(X_design @ w, -500.0, 500.0)))
    return (p >= 0.5).astype(int)

# 5. Performance Evaluation from Scratch
def evaluate_classification(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion: [TP={tp}, TN={tn}, FP={fp}, FN={fn}]")

# 6. Comparative Execution: Scaled vs Unscaled
for scale in [False, True]:
    w, mu, sigma = train_logistic(X_train.copy(), y_train, scale=scale, lam=1.0)
    preds = predict_logistic(X_test.copy(), w, mu, sigma)
    print(f"\n=== Evaluation: Feature Scaling = {scale} ===")
    evaluate_classification(y_test, preds)
