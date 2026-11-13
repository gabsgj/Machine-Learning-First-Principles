"""
K-Nearest Neighbors with Scikit-Learn
=====================================
Production-grade KNN classification:
- KNeighborsClassifier with Euclidean, Manhattan, and Minkowski metrics
- Uniform vs distance weighting
- Cross-validation curve generation to select optimal K
- Stratified evaluation on Digits and Iris datasets
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def tune_knn_hyperparameters(X_train: np.ndarray, y_train: np.ndarray):
    """
    Finds optimal K and distance metric using GridSearchCV.
    """
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "metric": ["euclidean", "manhattan"],
        "weights": ["uniform", "distance"]
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    print("=== Demo: Scikit-Learn KNN Hyperparameter Tuning ===")
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )

    best_params, best_cv = tune_knn_hyperparameters(X_train, y_train)
    print(f"Optimal Hyperparameters: {best_params} (CV Accuracy: {best_cv * 100:.2f}%)")

    # Fit best estimator
    best_clf = KNeighborsClassifier(**best_params)
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
