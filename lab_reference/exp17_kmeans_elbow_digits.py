"""
Experiment 17: K-Means Clustering and Cluster Cardinality Selection
===================================================================
Dataset: Optical Recognition of Handwritten Digits
Demonstrates the Elbow Method (Inertia curve) and Silhouette Coefficient validation.
"""

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion & Normalization
X, _ = load_digits(return_X_y=True)
X = X / 16.0

# 2. Sweep Cluster Counts K in [2, 10]
k_range = list(range(2, 11))
inertias = []
silhouettes = []

print("=== Experiment 17: K-Means Cluster Validation ===")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)
    sil = silhouette_score(X, km.labels_)
    silhouettes.append(sil)
    print(f"Clusters (K={k:<2}) | Inertia: {km.inertia_:.1f} | Silhouette: {sil:.4f}")

# 3. Elbow Plot
plt.figure(figsize=(7.5, 4.5), dpi=150)
plt.plot(k_range, inertias, marker='o', lw=2, color='#1f77b4')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Experiment 17: K-Means Elbow Method on Digits')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('exp17_kmeans_elbow.png')
print("Saved visualization to exp17_kmeans_elbow.png")
