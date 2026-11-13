"""
Clustering with Scikit-Learn & SciPy
====================================
Production-grade unsupervised clustering:
1. KMeans Clustering with Elbow Inertia evaluation
2. Agglomerative (Hierarchical) Clustering with Ward linkage
3. Dendrogram generation using SciPy
4. Silhouette Score validation
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np


def run_hierarchical_and_kmeans(X: np.ndarray, n_clusters: int = 3):
    """Fits both Agglomerative and KMeans clustering and computes silhouette metrics."""
    # 1. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X)
    sil_agg = silhouette_score(X, labels_agg)

    # 2. KMeans Clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_km = km.fit_predict(X)
    sil_km = silhouette_score(X, labels_km)

    return {
        "agg_labels": labels_agg,
        "km_labels": labels_km,
        "sil_agg": sil_agg,
        "sil_km": sil_km,
        "inertia": km.inertia_
    }


def plot_dendrogram_scipy(X: np.ndarray, save_path: str = "plots/11_dendrogram.png"):
    """Generates and saves hierarchical clustering dendrogram."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Z = linkage(X, method="ward")
    
    plt.figure(figsize=(9, 4.8), dpi=150)
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9)
    plt.title("Hierarchical Clustering Dendrogram (Ward's Minimum Variance Linkage)", fontsize=12, fontweight="bold")
    plt.xlabel("Sample Index or Cluster Size", fontsize=10)
    plt.ylabel("Cophenetic Distance", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print("=== Demo: Scikit-Learn Clustering on Iris ===")
    X = load_iris().data
    results = run_hierarchical_and_kmeans(X, n_clusters=3)
    plot_dendrogram_scipy(X)

    print(f"Agglomerative Silhouette Score: {results['sil_agg']:.4f}")
    print(f"KMeans Silhouette Score:        {results['sil_km']:.4f}")
    print(f"KMeans Inertia:                 {results['inertia']:.2f}")
    print(">> Generated dendrogram plot in 'plots/11_dendrogram.png'.")
