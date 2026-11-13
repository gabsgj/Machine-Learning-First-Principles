"""
Experiment 04: Logistic Regression with MLE and MAP Priors
==========================================================
Dataset: Breast Cancer Wisconsin Diagnostic
Demonstrates:
1. Maximum Likelihood Estimation (MLE, unregularized)
2. Maximum A Posteriori with Gaussian Prior (MAP, L2 penalty)
3. Maximum A Posteriori with Laplace Prior (MAP, L1 penalty)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Dataset Ingestion & Partitioning
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline Configuration
models = {
    'MLE (Unregularized)':     make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000)),
    'MAP Gaussian Prior (L2)': make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1.0, max_iter=5000)),
    'MAP Laplace Prior (L1)':  make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=1.0)),
}

# 3. Model Evaluation
print("=== Experiment 04: Parameter Estimation in Logistic Regression ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    coef = model[-1].coef_[0]
    norm_w = np.linalg.norm(coef)
    zero_weights = (abs(coef) < 1e-4).sum()
    print(f"{name:<25} | Accuracy: {acc*100:.2f}% | ||w||: {norm_w:.3f} | Zero Weights: {zero_weights}")
