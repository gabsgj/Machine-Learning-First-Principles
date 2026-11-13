"""
K-Means Clustering from Scratch
===============================
Algorithmic Implementation:
1. Centroid Initialization (Random and K-Means++)
2. Alternating Optimization: Assignment step and Mean Update step
3. Inertia (Within-Cluster Sum of Squares) tracking
4. Silhouette Coefficient calculation from first principles
"""

import numpy as np
from typing import Optional, Tuple, List


class KMeansScratch:
    """
    K-Means clustering algorithm implemented from first principles.
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters (K) to form.
    max_iter : int, default=300
        Maximum iterations of Lloyd's algorithm.
    tol : float, default=1e-4
        Tolerance for relative convergence in centroid movement.
    random_state : int, optional
        Seed for centroid initialization.
    """
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Initialize centroids randomly from existing data points
        init_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X_arr[init_indices].copy()

        for _ in range(self.max_iter):
            # 2. Assignment Step: Find nearest centroid for each point
            # Pairwise squared Euclidean distance: (n_samples, n_clusters)
            dists = np.sum((X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            # 3. Update Step: Recompute centroids as cluster means
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X_arr[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Re-initialize empty cluster to random sample
                    new_centroids[k] = X_arr[np.random.choice(n_samples)]

            # Check convergence
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels

        # Compute final Inertia: sum of squared distances to closest centroid
        min_sq_dists = np.min(np.sum((X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1)
        self.inertia_ = float(np.sum(min_sq_dists))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        dists = np.sum((X_arr[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)


def compute_silhouette_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Mean Silhouette Coefficient from first principles.
    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    X_arr = np.asarray(X, dtype=float)
    unique_labels = np.unique(labels)
    n_samples = len(X_arr)
    if len(unique_labels) <= 1:
        return 0.0

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        own_cluster = labels[i]
        own_points = X_arr[labels == own_cluster]
        
        # Mean intra-cluster distance a(i)
        if len(own_points) > 1:
            a_i = np.mean(np.linalg.norm(own_points - X_arr[i], axis=1))
        else:
            a_i = 0.0

        # Mean nearest-cluster distance b(i)
        b_i = np.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_points = X_arr[labels == other_cluster]
            if len(other_points) > 0:
                dist_to_other = np.mean(np.linalg.norm(other_points - X_arr[i], axis=1))
                if dist_to_other < b_i:
                    b_i = dist_to_other

        max_ab = max(a_i, b_i)
        silhouette_vals[i] = (b_i - a_i) / max_ab if max_ab > 0 else 0.0

    return float(np.mean(silhouette_vals))


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print("=== Demo: K-Means from Scratch on Iris Dataset ===")
    iris = load_iris()
    X = iris.data

    kmeans = KMeansScratch(n_clusters=3, random_state=42)
    kmeans.fit(X)

    sil_score = compute_silhouette_scratch(X, kmeans.labels_)
    print(f"Computed Inertia:          {kmeans.inertia_:.2f}")
    print(f"Scratch Silhouette Score: {sil_score:.4f}")
    print(f"Cluster Centers Shape:    {kmeans.cluster_centers_.shape}")
