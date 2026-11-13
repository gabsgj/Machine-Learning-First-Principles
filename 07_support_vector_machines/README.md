# Module 07: Support Vector Machines & Kernel Methods

## Mathematical Foundations & Derivations

The Support Vector Machine (SVM) finds the unique optimal separating hyperplane that maximizes the geometric margin between two linearly or non-linearly separable classes.

---

### 1. Geometric Margin & Why We Minimize $\frac{1}{2}\|w\|^2$
Let the separating hyperplane be defined by $w^T x + b = 0$. The perpendicular Euclidean distance from any sample point $x_i$ to the hyperplane is:
$$\text{dist}(x_i) = \frac{|w^T x_i + b|}{\|w\|}$$

For a correctly classified sample with label $y_i \in \{-1, +1\}$, the functional margin is $y_i (w^T x_i + b) > 0$. We scale $w$ and $b$ such that the closest points (the support vectors) satisfy:
$$|w^T x_{\text{SV}} + b| = 1$$

The total geometric margin between the positive boundary ($w^T x + b = +1$) and negative boundary ($w^T x + b = -1$) is:
$$\text{Margin} = \frac{(w^T x_+ + b) - (w^T x_- + b)}{\|w\|} = \frac{1 - (-1)}{\|w\|} = \frac{2}{\|w\|}$$

Maximizing the margin $\frac{2}{\|w\|}$ is mathematically equivalent to minimizing the reciprocal squared norm:
$$\min_{w, b} \frac{1}{2} \|w\|_2^2 \quad \text{subject to } y_i (w^T x_i + b) \ge 1, \quad \forall i=1,\dots,n$$

---

### 2. Soft-Margin SVM & Hinge Loss
In the presence of noise or overlapping distributions, slack variables $\xi_i \ge 0$ allow bounded margin violations:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Rewriting in unconstrained form via the **Hinge Loss**:
$$L_{\text{hinge}}(y_i, f(x_i)) = \max(0, 1 - y_i (w^T x_i + b))$$
$$J(w, b) = \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b))$$

The subgradient with respect to $w$ is:
$$\nabla_w J = \begin{cases}
w - C y_i x_i & \text{if } y_i(w^T x_i + b) < 1 \\
w & \text{otherwise}
\end{cases}$$

---

### 3. Dual Formulation & The Kernel Trick
Constructing the Lagrangian with multipliers $\alpha_i \ge 0$:
$$\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1]$$
Setting $\nabla_w \mathcal{L} = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i$. Substituting back yields the Wolfe Dual:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \quad \text{s.t. } 0 \le \alpha_i \le C, \sum_{i=1}^n \alpha_i y_i = 0$$

By **Mercer's Theorem**, any inner product in an implicit Hilbert feature space $\langle \phi(x_i), \phi(x_j) \rangle$ can be replaced by a valid kernel evaluation $K(x_i, x_j)$:
* **Linear:** $K(x, z) = x^T z$
* **Polynomial:** $K(x, z) = (\gamma x^T z + r)^d$
* **Radial Basis Function (RBF / Gaussian):** $K(x, z) = \exp(-\gamma \|x - z\|^2)$

The resulting decision function relies **only** on inner products with the support vectors ($\alpha_i > 0$):
$$f(x) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b \right)$$

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_svm.py` | Subgradient descent Pegasos implementation solving the soft-margin primal SVM, tracking margins and support vectors. |
| `library_svm.py` | Scikit-Learn `SVC` supporting Linear, Polynomial, and RBF kernels with multi-class extension. |
| `compare_svm.py` | Visualizes decision hyperplanes, $+1/-1$ margin lines, circled support vectors, and non-linear kernel boundaries on Iris. |

## Quick Execution

```bash
python3 scratch_svm.py
python3 library_svm.py
python3 compare_svm.py
```
