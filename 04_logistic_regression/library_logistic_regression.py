"""
Logistic Regression with Scikit-Learn
=====================================
Production-grade Logistic Regression pipelines:
- Standardization pipeline
- L1 (Lasso) vs L2 (Ridge) penalties
- Inverse regularization strength (C parameter)
- Class balance weighting
- Comprehensive classification report & ROC-AUC
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def train_sklearn_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight: str = None,
):
    """Fits a standardized LogisticRegression model and prints performance diagnostics."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty=penalty, C=C, solver=solver, class_weight=class_weight, max_iter=2000, random_state=42
        ))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "model": pipe,
        "predictions": y_pred,
        "probabilities": y_prob,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred)
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("=== Demo: Scikit-Learn Logistic Regression Pipeline ===")
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    results = train_sklearn_logistic_regression(X_tr, y_tr, X_te, y_te, C=1.0)
    print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['report'])
