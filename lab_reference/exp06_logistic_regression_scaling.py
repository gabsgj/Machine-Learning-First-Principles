"""
Experiment 06: Logistic Regression with and without Feature Standardization
===========================================================================
Dataset: Breast Cancer Wisconsin Diagnostic
Demonstrates why gradient-based optimization requires feature scaling.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Dataset Ingestion & Partitioning
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Standardization
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

# 3. Model Fitting and Comparative Diagnostics
print("=== Experiment 06: Impact of Feature Scaling ===")
for Xtr, Xte, name in [(X_train, X_test, "Unscaled Features"), (Xs_train, Xs_test, "Standardized Features")]:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall:    {recall_score(y_test, pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, pred):.4f}")
