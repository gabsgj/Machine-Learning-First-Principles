# Module 11: Unsupervised Clustering & Hierarchical Analysis

## Mathematical Foundations & Principles

Clustering partitions unlabeled observations $\{x_1, \dots, x_n\} \subset \mathbb{R}^d$ into distinct subsets such that intra-cluster similarity is maximized and inter-cluster similarity is minimized.

---

### 1. K-Means & Lloyd's Optimization
K-Means solves the hard-assignment partitioning problem minimizing Within-Cluster Sum of Squares (WCSS / Inertia):
$$J(C_1, \dots, C_K) = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$

Lloyd's algorithm alternates two convex optimization steps:
1. **Assignment Step:**
   $$C_k^{(t)} = \{x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2, \forall j\}$$
2. **Centroid Update Step:**
   $$\mu_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{x_i \in C_k^{(t)}} x_i$$

Because each step monotonically decreases $J$ on a finite partition space, Lloyd's algorithm is guaranteed to converge to a local minimum.

---

### 2. Validation Metrics: The Silhouette Coefficient
For observation $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
* $a(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, j \ne i} \|x_i - x_j\|$ (Mean intra-cluster distance).
* $b(i) = \min_{J \ne I} \frac{1}{|C_J|} \sum_{j \in C_J} \|x_i - x_j\|$ (Mean distance to nearest neighboring cluster).

---

### 3. Hierarchical Agglomerative Clustering & Linkage
Starts with $n$ singleton clusters and iteratively merges the closest pair.
* **Ward's Minimum Variance Linkage:**
  $$\Delta \text{ESS}_{AB} = \frac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|_2^2$$
  Minimizes the increase in total within-cluster variance at each merge.

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_kmeans.py` | Full Lloyd's K-Means clustering algorithm and Silhouette coefficient from scratch. |
| `library_clustering.py` | Scikit-Learn `KMeans`, `AgglomerativeClustering`, and SciPy hierarchical dendrogram generator. |
| `compare_clustering.py` | Numerical parity checks, side-by-side 2D cluster scatter comparison, and the Elbow method curve. |

## Quick Execution

```bash
python3 scratch_kmeans.py
python3 library_clustering.py
python3 compare_clustering.py
```
