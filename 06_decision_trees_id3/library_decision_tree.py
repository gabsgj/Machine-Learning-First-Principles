"""
Decision Tree Classifier with Scikit-Learn
==========================================
Scikit-Learn CART (Classification and Regression Trees) implementation:
- Splitting criteria: Entropy (Information Gain) vs. Gini Impurity
- Tree regularization: max_depth, min_samples_split, min_samples_leaf
- Feature importance extraction
- Tree visualization rendering
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


def train_sklearn_decision_tree(
    X_train, y_train, X_test, y_test, criterion="entropy", max_depth=3, feature_names=None
):
    """Fits a decision tree, generates feature importances, and exports a visual plot."""
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6), dpi=150)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title(f"Scikit-Learn Decision Tree (Criterion={criterion}, Max Depth={max_depth})", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/06_decision_tree_structure.png")
    plt.close()

    return clf, acc, y_pred


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    print("=== Demo: Scikit-Learn Decision Tree on Breast Cancer ===")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    clf, acc, preds = train_sklearn_decision_tree(
        X_train, y_train, X_test, y_test, criterion="entropy", max_depth=3, feature_names=data.feature_names
    )
    print(f"Decision Tree Accuracy: {acc * 100:.2f}%")
    print(f"Top 3 Features: {pd.Series(clf.feature_importances_, index=data.feature_names).sort_values(ascending=False).head(3)}")
