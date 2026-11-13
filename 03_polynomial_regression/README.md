# Module 03: Polynomial Regression & Model Complexity

## Mathematical Foundations

When target relationships exhibit curvature, standard linear regression underfits. Polynomial regression projects the original feature space $\mathcal{X} \subset \mathbb{R}$ into an augmented Hilbert feature space via non-linear basis transformations:
$$\phi(x) = [1, x, x^2, x^3, \dots, x^d]^T \in \mathbb{R}^{d+1}$$

The resulting model is linear in the parameters $\theta$:
$$\hat{y}(x) = \phi(x)^T \theta = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d$$

Given $n$ samples, the design matrix $\Phi \in \mathbb{R}^{n \times (d+1)}$ is a Vandermonde matrix:
$$\Phi = \begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^d \\
1 & x_2 & x_2^2 & \dots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \dots & x_n^d
\end{bmatrix}$$

---

### Parameter Estimation & Ill-Conditioning
The parameters $\theta^*$ are calculated using the Normal Equation:
$$\theta^* = (\Phi^T \Phi)^{-1} \Phi^T y$$

> [!CAUTION]
> As degree $d$ increases, the columns of $\Phi$ ($x^k$) become nearly collinear, and their magnitudes diverge exponentially. For unscaled $x \in [10, 100]$, $x^4 \in [10^4, 10^8]$, creating a severely ill-conditioned Gram matrix $\Phi^T \Phi$. **Standardizing features prior to polynomial expansion is strictly necessary** to stabilize the matrix inversion.

---

### The Bias-Variance Tradeoff in Polynomials
* **Low Degree ($d=1$):** High Bias (Underfitting). Model is incapable of expressing the underlying curvature.
* **Optimal Degree ($d=2, 3$):** Balanced Bias and Variance. Lowest validation MSE.
* **High Degree ($d \ge 7$):** High Variance (Overfitting / Runge's Phenomenon). The curve oscillates aggressively between sample points, capturing stochastic training noise.

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_polynomial_regression.py` | Vectorized basis feature generator + pseudo-inverse normal equation solver. |
| `library_polynomial_regression.py` | Scikit-Learn `PolynomialFeatures` pipeline coupled with `GridSearchCV` 5-fold cross-validation. |
| `compare_polynomial_regression.py` | Benchmarks degree fit curves, validates exact output parity, and plots the U-shaped Bias-Variance error curve. |

## Quick Execution

```bash
python3 scratch_polynomial_regression.py
python3 library_polynomial_regression.py
python3 compare_polynomial_regression.py
```
