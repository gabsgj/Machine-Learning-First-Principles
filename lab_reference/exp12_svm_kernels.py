"""
Experiment 12: Support Vector Machines with Non-Linear Kernels
==============================================================
Dataset: Optical Recognition of Handwritten Digits
Compares Linear, Polynomial (degree 3), and Radial Basis Function (RBF) kernels.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Dataset Ingestion & Min-Max Pixel Scaling to [0, 1]
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Kernel Formulations
kernels = {
    'Linear Kernel':        SVC(kernel='linear', C=1.0),
    'Polynomial Kernel (3)':SVC(kernel='poly', degree=3, C=1.0),
    'RBF Kernel':           SVC(kernel='rbf', gamma='scale', C=1.0)
}

# 3. Model Fitting & Evaluation
print("=== Experiment 12: SVM Kernel Comparison on Digits ===")
for name, model in kernels.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred) * 100
    n_sv = len(model.support_vectors_)
    print(f"{name:<22} | Accuracy: {acc:.2f}% | Support Vectors: {n_sv}")
