# Module 05: K-Nearest Neighbors (KNN)

## Mathematical Foundations & Principles

The K-Nearest Neighbors algorithm is a non-parametric, instance-based (lazy) learning algorithm that estimates conditional class probabilities based on local proximity in the feature space $\mathbb{R}^d$.

---

### 1. Distance Metric Formulations
For query vector $x \in \mathbb{R}^d$ and training vector $t \in \mathbb{R}^d$:

* **Euclidean Distance ($L_2$ Norm):**
  $$d_{\text{Euclidean}}(x, t) = \|x - t\|_2 = \sqrt{\sum_{j=1}^d (x_j - t_j)^2}$$
* **Manhattan Distance ($L_1$ Norm):**
  $$d_{\text{Manhattan}}(x, t) = \|x - t\|_1 = \sum_{j=1}^d |x_j - t_j|$$
* **Minkowski Distance ($L_p$ Metric):**
  $$d_{\text{Minkowski}}(x, t) = \left( \sum_{j=1}^d |x_j - t_j|^p \right)^{1/p}$$

---

### 2. Decision Rules & Voting
Let $\mathcal{N}_k(x) \subset \mathcal{D}_{\text{train}}$ denote the set of $k$ training points with smallest distance to $x$:

* **Uniform Majority Voting:**
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c)$$
* **Distance-Weighted Voting:**
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(x)} \frac{1}{d(x, x_i) + \epsilon} \cdot \mathbb{I}(y_i = c)$$

---

### 3. The Curse of Dimensionality
As dimension $d \to \infty$, the volume of the space grows exponentially ($V \propto r^d$). The distance between any two randomly chosen points converges to the same value:
$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
Hence, distance-based neighborhood concepts degrade unless dimensionality reduction (PCA) or feature selection is applied.

---

## Computational Complexity

| Operation | Brute Force (NumPy) | KD-Tree / Ball Tree |
| :--- | :--- | :--- |
| **Training Phase** | $\mathcal{O}(1)$ (Memory Storage Only) | $\mathcal{O}(d \cdot n \log n)$ |
| **Query (Per Sample)** | $\mathcal{O}(n \cdot d + n \log k)$ | $\mathcal{O}(2^d \cdot \log n)$ |
| **Memory Footprint** | $\mathcal{O}(n \cdot d)$ | $\mathcal{O}(n \cdot d)$ |

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_knn.py` | Standalone implementation supporting custom distance metrics, weighted voting, and missing-value tolerance. |
| `library_knn.py` | Scikit-Learn `KNeighborsClassifier` integrated with `GridSearchCV` hyperparameter tuning. |
| `compare_knn.py` | Prediction parity checks on Digits dataset and $K$-sweep training vs testing accuracy curves on Iris. |

## Quick Execution

```bash
python3 scratch_knn.py
python3 library_knn.py
python3 compare_knn.py
```
