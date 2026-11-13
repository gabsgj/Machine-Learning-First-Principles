"""
Ensemble Learning with Scikit-Learn
===================================
Production implementations of syllabus Experiment 19:
- Single Decision Tree baseline
- BaggingClassifier (Bootstrap committee)
- AdaBoostClassifier (Adaptive sequential boosting)
- RandomForestClassifier (Feature bagging)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_ensemble_suite(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Fits and benchmarks Single Tree vs. Bagging vs. AdaBoost vs. Random Forest."""
    models = {
        "Single Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }

    return results


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("=== Demo: Scikit-Learn Ensemble Suite on Breast Cancer ===")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    metrics = evaluate_ensemble_suite(X_train, y_train, X_test, y_test)
    for model_name, res in metrics.items():
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:  {res['accuracy'] * 100:.2f}%")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall:    {res['recall']:.4f}")
        print(f"F1 Score:  {res['f1']:.4f}")
