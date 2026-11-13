"""
Comparison and Visual Diagnostics: Support Vector Machines
==========================================================
Visualizes:
1. Hard/Soft Margin separating hyperplanes with exact margin boundaries: w^T x + b = +- 1.
2. Direct identification and circling of Support Vectors (Scratch vs. Sklearn).
3. Non-linear boundary generation across Linear, Polynomial, and RBF kernels.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scratch_svm import LinearSVMScratch


def plot_svm_decision_boundary_with_margins():
    print("==================================================")
    print("SVM BENCHMARK: Separating Hyperplane and Margins")
    print("==================================================")
    iris = load_iris()
    # Features: Petal length (col 2), Petal width (col 3)
    X = iris.data[:, [2, 3]]
    y = (iris.target == 0).astype(int)  # 1 for Setosa, 0 for Non-Setosa

    # 1. Scratch Linear SVM
    svm_scratch = LinearSVMScratch(C=5.0, learning_rate=0.01, epochs=1000)
    svm_scratch.fit(X, y)

    # 2. Sklearn Linear SVM
    svm_sklearn = SVC(kernel="linear", C=5.0)
    svm_sklearn.fit(X, y)

    print(f"Scratch SVM w: {np.round(svm_scratch.w_, 4)} | b: {svm_scratch.b_:.4f}")
    print(f"Sklearn SVM w: {np.round(svm_sklearn.coef_[0], 4)} | b: {svm_sklearn.intercept_[0]:.4f}")
    print(f"Support Vectors: Scratch={len(svm_scratch.support_vectors_)} | Sklearn={len(svm_sklearn.support_vectors_)}")

    # Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(9, 6), dpi=150)

    # Plot sample points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=50, label="Iris Samples")

    # Highlight support vectors from sklearn
    plt.scatter(
        svm_sklearn.support_vectors_[:, 0],
        svm_sklearn.support_vectors_[:, 1],
        s=160,
        facecolors="none",
        edgecolors="#2ca02c",
        linewidths=2.0,
        label=f"Support Vectors ({len(svm_sklearn.support_vectors_)})"
    )

    # Generate grid for margin contours
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = svm_sklearn.decision_function(grid).reshape(xx.shape)

    # Plot decision boundary (Z=0) and margins (Z=+1, Z=-1)
    plt.contour(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors=["black", "blue", "black"],
                linestyles=["--", "-", "--"], linewidths=[1.5, 2.5, 1.5])

    plt.title("Linear SVM: Optimal Hyperplane & Margins (Iris Setosa vs Non-Setosa)", fontsize=12, fontweight="bold")
    plt.xlabel("Petal Length (cm)", fontsize=10)
    plt.ylabel("Petal Width (cm)", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("plots/07_svm_margin_and_hyperplane.png")
    plt.close()
    print(">> Saved margin plot to 'plots/07_svm_margin_and_hyperplane.png' <<\n")


def plot_kernel_comparisons():
    print("==================================================")
    print("SVM KERNELS COMPARISON: Linear vs Poly vs RBF")
    print("==================================================")
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target  # All 3 classes

    kernels = [
        ("Linear Kernel", SVC(kernel="linear", C=1.0)),
        ("Polynomial Kernel (d=3)", SVC(kernel="poly", degree=3, C=1.0)),
        ("RBF Kernel (gamma=0.7)", SVC(kernel="rbf", gamma=0.7, C=1.0))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, (title, clf) in zip(axes, kernels):
        clf.fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=35)
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, facecolors="none", edgecolors="red", lw=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")

    plt.tight_layout()
    plt.savefig("plots/07_svm_kernels_comparison.png")
    plt.close()
    print(">> Saved multi-kernel plot to 'plots/07_svm_kernels_comparison.png' <<\n")


if __name__ == "__main__":
    plot_svm_decision_boundary_with_margins()
    plot_kernel_comparisons()
