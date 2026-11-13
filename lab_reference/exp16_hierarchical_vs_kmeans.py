"""
Experiment 16: Hierarchical vs. K-Means Clustering Analysis
===========================================================
Dataset: Iris (Petal Dimensions)
Compares Ward Linkage Dendrogram against Lloyd's K-Means clustering.
"""

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion
X = load_iris().data

# 2. Hierarchical Dendrogram (Ward's Minimum Variance Linkage)
Z = linkage(X, method='ward')
plt.figure(figsize=(7.5, 4), dpi=150)
dendrogram(Z, no_labels=True)
plt.title('Experiment 16: Hierarchical Dendrogram (Ward Linkage)')
plt.ylabel('Cophenetic Distance')
plt.tight_layout()
plt.savefig('exp16_dendrogram.png')
plt.close()

# 3. Model Induction (K=3)
agg = AgglomerativeClustering(n_clusters=3).fit_predict(X)
km_model = KMeans(n_clusters=3, random_state=42, n_init=10)
km = km_model.fit_predict(X)

# 4. Metric Diagnostics
print("=== Experiment 16: Clustering Performance ===")
print(f"Agglomerative Silhouette Score: {silhouette_score(X, agg):.4f}")
print(f"K-Means Silhouette Score:       {silhouette_score(X, km):.4f}")
print(f"K-Means Inertia (WCSS):         {km_model.inertia_:.2f}")

# 5. Comparative 2D Scatter Visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
ax[0].scatter(X[:, 2], X[:, 3], c=agg, cmap='viridis', edgecolors='k', s=45)
ax[0].set_title('Agglomerative Clustering')
ax[0].set_xlabel('Petal Length (cm)')
ax[0].set_ylabel('Petal Width (cm)')

ax[1].scatter(X[:, 2], X[:, 3], c=km, cmap='viridis', edgecolors='k', s=45)
ax[1].set_title('K-Means Clustering')
ax[1].set_xlabel('Petal Length (cm)')
ax[1].set_ylabel('Petal Width (cm)')

plt.tight_layout()
plt.savefig('exp16_cluster_scatter.png')
print("Saved visualizations to exp16_dendrogram.png and exp16_cluster_scatter.png")
