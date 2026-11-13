"""
Support Vector Machines with Scikit-Learn
=========================================
Production implementation comparing Kernel Tricks:
1. Linear Kernel: K(x, z) = x^T z
2. Polynomial Kernel: K(x, z) = (gamma * x^T z + r)^d
3. Radial Basis Function (RBF / Gaussian): K(x, z) = exp(-gamma * ||x - z||^2)
4. Multi-Class classification (One-vs-Rest and One-vs-One)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


def compare_svm_kernels(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Fits and compares Linear, Polynomial, and RBF SVM kernels."""
    kernels = {
        "Linear SVM": SVC(kernel="linear", C=1.0),
        "Polynomial SVM (d=3)": SVC(kernel="poly", degree=3, C=1.0),
        "RBF Kernel SVM": SVC(kernel="rbf", gamma="scale", C=1.0),
    }

    results = {}
    for name, model in kernels.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        n_sv = len(model.support_vectors_)
        results[name] = {
            "model": model,
            "accuracy": acc,
            "n_support_vectors": n_sv,
            "report": classification_report(y_test, preds)
        }
    return results


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=== Demo: Scikit-Learn SVM Kernels on Iris ===")
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    kernel_res = compare_svm_kernels(X_train, y_train, X_test, y_test)
    for k_name, res in kernel_res.items():
        print(f"\n--- {k_name} ---")
        print(f"Accuracy:        {res['accuracy'] * 100:.2f}%")
        print(f"Support Vectors: {res['n_support_vectors']}")
