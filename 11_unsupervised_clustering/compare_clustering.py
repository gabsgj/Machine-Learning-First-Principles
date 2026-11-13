"""
Comparison and Visual Diagnostics: Unsupervised Clustering
==========================================================
Benchmarks:
1. Parity validation: Scratch KMeans vs. Scikit-Learn KMeans inertia.
2. Side-by-side visual comparison: Agglomerative vs. KMeans on Iris (Petal features).
3. The Elbow Method: Inertia reduction across K=1..10 on Digits dataset.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

from scratch_kmeans import KMeansScratch, compute_silhouette_scratch


def compare_scratch_vs_sklearn_kmeans():
    print("==================================================")
    print("1. KMEANS PARITY: Scratch vs Sklearn on Iris")
    print("==================================================")
    X = load_iris().data

    # Scratch KMeans
    km_sc = KMeansScratch(n_clusters=3, random_state=42)
    km_sc.fit(X)
    sil_sc = compute_silhouette_scratch(X, km_sc.labels_)

    # Sklearn KMeans
    km_sk = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_sk.fit(X)
    sil_sk = silhouette_score(X, km_sk.labels_)

    print(f"Scratch Inertia: {km_sc.inertia_:.2f} | Silhouette: {sil_sc:.4f}")
    print(f"Sklearn Inertia: {km_sk.inertia_:.2f} | Silhouette: {sil_sk:.4f}")
    diff = abs(km_sc.inertia_ - km_sk.inertia_)
    print(f"Inertia Discrepancy: {diff:.2f}")


def plot_hierarchical_vs_kmeans_scatter():
    print("\n==================================================")
    print("2. VISUAL CLUSTERING: Agglomerative vs KMeans (Iris)")
    print("==================================================")
    X = load_iris().data

    # Dendrogram
    os.makedirs("plots", exist_ok=True)
    Z = linkage(X, method="ward")
    plt.figure(figsize=(8, 4), dpi=150)
    dendrogram(Z, no_labels=True)
    plt.title("Iris Hierarchical Dendrogram (Ward Linkage)", fontsize=11, fontweight="bold")
    plt.ylabel("Cophenetic Distance")
    plt.tight_layout()
    plt.savefig("plots/11_dendrogram.png")
    plt.close()

    # Cluster assignments
    agg_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    km_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

    # Side-by-side 2D scatter plots on Petal Length (col 2) and Petal Width (col 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    axes[0].scatter(X[:, 2], X[:, 3], c=agg_labels, cmap="viridis", edgecolors="k", s=45)
    axes[0].set_title(f"Agglomerative Clustering (Sil: {silhouette_score(X, agg_labels):.3f})", fontweight="bold")
    axes[0].set_xlabel("Petal Length (cm)")
    axes[0].set_ylabel("Petal Width (cm)")

    axes[1].scatter(X[:, 2], X[:, 3], c=km_labels, cmap="viridis", edgecolors="k", s=45)
    axes[1].set_title(f"K-Means Clustering (Sil: {silhouette_score(X, km_labels):.3f})", fontweight="bold")
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].set_ylabel("Petal Width (cm)")

    plt.tight_layout()
    plt.savefig("plots/11_iris_clustering_comparison.png")
    plt.close()
    print(">> Saved side-by-side clustering plot to 'plots/11_iris_clustering_comparison.png' <<\n")


def plot_elbow_method_digits():
    print("==================================================")
    print("3. ELBOW METHOD: Inertia across K=1..10 (Digits)")
    print("==================================================")
    X = load_digits().data[:400]
    k_range = list(range(1, 11))
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7.5, 4.5), dpi=150)
    plt.plot(k_range, inertias, marker="o", color="#1f77b4", lw=2)
    plt.title("The Elbow Method: Optimal K Identification", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Clusters (K)", fontsize=10)
    plt.ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=10)
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/11_kmeans_elbow_inertia.png")
    plt.close()
    print(">> Saved elbow curve to 'plots/11_kmeans_elbow_inertia.png' <<\n")


if __name__ == "__main__":
    compare_scratch_vs_sklearn_kmeans()
    plot_hierarchical_vs_kmeans_scatter()
    plot_elbow_method_digits()
