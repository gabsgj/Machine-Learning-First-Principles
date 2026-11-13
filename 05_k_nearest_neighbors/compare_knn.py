"""
Comparison and Benchmarking: K-Nearest Neighbors
================================================
Side-by-side verification:
1. Exact prediction agreement between Scratch and Scikit-Learn.
2. Distance metric comparison: Euclidean vs. Manhattan across K values.
3. K hyperparameter sweep plotting Training vs. Testing accuracy curve.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scratch_knn import KNNClassifierScratch, euclidean_distance, manhattan_distance


def compare_scratch_vs_sklearn_digits():
    print("==================================================")
    print("KNN BENCHMARK: Digits Dataset (Parity Verification)")
    print("==================================================")
    digits = load_digits()
    X, y = digits.data[:600], digits.target[:600]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"{'Metric':<12} {'K':<5} {'Scratch Acc':<15} {'Sklearn Acc':<15} {'Disagreements':<15}")
    print("-" * 65)

    for metric_str, m_func in [("euclidean", euclidean_distance), ("manhattan", manhattan_distance)]:
        for k in [1, 3, 5, 7]:
            # Scratch
            sc = KNNClassifierScratch(k=k, metric=m_func)
            sc.fit(X_train, y_train)
            pred_sc = sc.predict(X_test)
            acc_sc = accuracy_score(y_test, pred_sc) * 100

            # Sklearn
            sk = KNeighborsClassifier(n_neighbors=k, metric=metric_str)
            sk.fit(X_train, y_train)
            pred_sk = sk.predict(X_test)
            acc_sk = accuracy_score(y_test, pred_sk) * 100

            disagreements = int(np.sum(pred_sc != pred_sk))
            print(f"{metric_str:<12} {k:<5} {acc_sc:<15.2f}% {acc_sk:<15.2f}% {disagreements:<15}")


def plot_k_vs_accuracy_curve_iris():
    print("\n==================================================")
    print("K HYPERPARAMETER SWEEP: Iris Dataset Curve")
    print("==================================================")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    k_range = list(range(1, 26))
    train_accs = []
    test_accs = []

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        train_accs.append(clf.score(X_train, y_train))
        test_accs.append(clf.score(X_test, y_test))

    best_k = k_range[int(np.argmax(test_accs))]
    best_acc = max(test_accs)
    print(f"Optimal K: {best_k} with Test Accuracy: {best_acc * 100:.2f}%")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(k_range, train_accs, marker="s", color="#1f77b4", lw=2, label="Training Accuracy")
    plt.plot(k_range, test_accs, marker="o", color="#2ca02c", lw=2, label="Testing Accuracy")
    plt.axvline(best_k, color="#d62728", linestyle="--", label=f"Optimal K = {best_k} ({best_acc*100:.1f}%)")
    plt.title("KNN Bias-Variance Tradeoff: Accuracy vs. K (Iris Dataset)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Neighbors (K)", fontsize=10)
    plt.ylabel("Accuracy Score", fontsize=10)
    plt.xticks(range(1, 26, 2))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/05_knn_k_accuracy_curve.png")
    plt.close()
    print(">> Saved K-sweep plot to 'plots/05_knn_k_accuracy_curve.png' <<\n")


if __name__ == "__main__":
    compare_scratch_vs_sklearn_digits()
    plot_k_vs_accuracy_curve_iris()
