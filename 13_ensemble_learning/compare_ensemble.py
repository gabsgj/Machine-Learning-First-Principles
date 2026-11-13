"""
Comparison and Benchmarking: Ensemble Learning (EXP 19)
=======================================================
Direct execution of syllabus Experiment 19:
1. Head-to-head evaluation: Single Decision Tree vs. Bagging vs. AdaBoost.
2. Accuracy, Precision, Recall, and F1-score metric tracking.
3. Multi-line diagnostic plot replicating the official syllabus comparison.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_ensemble_benchmark():
    print("==================================================")
    print("EXP 19: Single Tree vs. Bagging vs. AdaBoost (Breast Cancer)")
    print("==================================================")
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Single Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    acc, pre, rec, f1 = [], [], [], []

    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 61)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        a = accuracy_score(y_test, pred)
        p = precision_score(y_test, pred)
        r = recall_score(y_test, pred)
        f = f1_score(y_test, pred)

        acc.append(a)
        pre.append(p)
        rec.append(r)
        f1.append(f)

        print(f"{name:<15} {a:<12.4f} {p:<12.4f} {r:<12.4f} {f:<10.4f}")

    # Plot lines with markers (Replicating exact WhatsApp Exp 19 visual)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(acc, label='Accuracy', marker='o', lw=2)
    plt.plot(pre, label='Precision', marker='s', lw=2)
    plt.plot(rec, label='Recall', marker='^', lw=2)
    plt.plot(f1, label='F1 Score', marker='d', lw=2)

    plt.xticks([0, 1, 2], ['Single Tree', 'Bagging', 'AdaBoost'], fontsize=10, fontweight="bold")
    plt.ylabel('Score Metric', fontsize=10)
    plt.title('Ensemble Model Comparison: Single Tree vs Bagging vs AdaBoost', fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('plots/13_ensemble_comparison.png')
    plt.close()
    print(">> Saved ensemble comparison plot to 'plots/13_ensemble_comparison.png' <<\n")


if __name__ == "__main__":
    run_ensemble_benchmark()
