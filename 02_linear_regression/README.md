# Module 02: Linear Regression & Regularization

## Mathematical Foundations & Derivations

Linear regression models the relationship between dependent target variable $y \in \mathbb{R}$ and feature vector $x \in \mathbb{R}^d$ via a linear combination of weights $w$ and intercept $b$:
$$\hat{y} = w^T x + b$$

By prepending a constant $1$ to the feature vector (augmenting the design matrix $X_g = [\mathbf{1}, X] \in \mathbb{R}^{n \times (d+1)}$) and appending the bias to parameters $\theta = [b, w_1, \dots, w_d]^T$:
$$\hat{y} = X_g \theta$$

---

### 1. Cost Function & Gradient Derivation
The objective is to minimize the Mean Squared Error (MSE):
$$J(\theta) = \frac{1}{n} \|X_g \theta - y\|_2^2 = \frac{1}{n} (X_g \theta - y)^T (X_g \theta - y)$$

Expanding the quadratic form:
$$J(\theta) = \frac{1}{n} \left( \theta^T X_g^T X_g \theta - 2 y^T X_g \theta + y^T y \right)$$

Taking the gradient with respect to parameter vector $\theta$:
$$\nabla_\theta J(\theta) = \frac{2}{n} X_g^T (X_g \theta - y)$$

In Batch Gradient Descent, parameters are updated iteratively opposite the gradient direction:
$$\theta^{(t+1)} = \theta^{(t)} - \alpha_t \nabla_\theta J(\theta^{(t)})$$

With **learning rate decay schedule**:
$$\alpha_t = \frac{\alpha_0}{1 + k \cdot t}$$
where $\alpha_0$ is the initial learning rate, $k$ is the decay constant, and $t$ is the epoch number.

---

### 2. Analytical Closed-Form Solution: The Normal Equation
Setting the gradient $\nabla_\theta J(\theta)$ to zero:
$$\frac{2}{n} X_g^T (X_g \theta - y) = 0 \implies X_g^T X_g \theta = X_g^T y$$

Assuming $X_g^T X_g$ is non-singular (invertible):
$$\theta^* = (X_g^T X_g)^{-1} X_g^T y$$
Using the Moore-Penrose pseudo-inverse $\text{pinv}(X_g^T X_g)$ guarantees numerical stability even when features exhibit multicollinearity.

---

### 3. Regularization: Ridge ($L_2$) & Lasso ($L_1$)
When features are correlated, $(X_g^T X_g)$ becomes ill-conditioned, leading to extreme parameter variance.

* **Ridge Regression ($L_2$ Penalty):**
  $$J_{\text{Ridge}}(\theta) = \frac{1}{n} \|X_g \theta - y\|^2 + \lambda \|\theta_{1:d}\|_2^2$$
  Closed-form solution:
  $$\theta^* = (X_g^T X_g + n\lambda I')^{-1} X_g^T y, \quad \text{where } I' = \begin{bmatrix} 0 & 0 \\ 0 & I_d \end{bmatrix}$$
* **Lasso Regression ($L_1$ Penalty):**
  $$J_{\text{Lasso}}(\theta) = \frac{1}{n} \|X_g \theta - y\|^2 + \lambda \|\theta_{1:d}\|_1$$
  Promotes exact parameter sparsity ($\theta_j = 0$), performing automated feature selection.

---

## Computational Complexity

| Method | Training Time | Prediction Time | Memory | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Equation** | $\mathcal{O}(n d^2 + d^3)$ | $\mathcal{O}(d)$ | $\mathcal{O}(d^2)$ | Small to medium feature spaces ($d < 10^4$) |
| **Gradient Descent** | $\mathcal{O}(K \cdot n d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(n d)$ | High dimensions ($d \gg 10^4$), streaming data |
| **SGDRegressor** | $\mathcal{O}(K \cdot d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(d)$ | Large-scale datasets exceeding RAM |

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_linear_regression.py` | Complete class `LinearRegressionScratch` with Normal Equation, LR Decay Gradient Descent, Ridge penalty, and custom evaluation metrics. |
| `library_linear_regression.py` | Scikit-Learn pipelines for OLS `LinearRegression`, `Ridge`, `Lasso`, and `SGDRegressor`. |
| `compare_linear_regression.py` | Benchmark script verifying numerical parity, parameter estimates, and generating fit curves and loss trajectories. |

## Quick Execution

```bash
python3 scratch_linear_regression.py
python3 library_linear_regression.py
python3 compare_linear_regression.py
```
