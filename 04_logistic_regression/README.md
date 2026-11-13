# Module 04: Logistic Regression & Regularization

## Mathematical Foundations & Derivations

Logistic Regression is the foundational probabilistic classifier for binary outcomes $y \in \{0, 1\}$.

---

### 1. Derivation of Sigmoid from the Log-Odds Assumption
Assume the log-odds (logit) of class membership is a linear combination of input features $x \in \mathbb{R}^d$:
$$\ln \left( \frac{p}{1 - p} \right) = w^T x + b = z$$

Exponentiating both sides:
$$\frac{p}{1 - p} = e^z \implies p = e^z (1 - p) = e^z - p e^z$$
$$p (1 + e^z) = e^z \implies p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} \equiv \sigma(z)$$

---

### 2. Derivation of Cross-Entropy Loss & Gradient
Under the Bernoulli likelihood assumption:
$$P(y \mid x) = p^y (1 - p)^{1 - y}$$

The Negative Log-Likelihood (Binary Cross-Entropy Loss) over $n$ samples is:
$$J(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

To derive the gradient with respect to weight $w_j$, we invoke the chain rule:
$$\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

1. **Loss derivative with respect to probability $p$:**
   $$\frac{\partial J}{\partial p} = -\left( \frac{y}{p} - \frac{1 - y}{1 - p} \right) = -\frac{y(1 - p) - p(1 - y)}{p(1 - p)} = \frac{p - y}{p(1 - p)}$$
2. **Sigmoid derivative with respect to logit $z$:**
   $$\frac{\partial p}{\partial z} = \frac{\partial}{\partial z} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} (-e^{-z}) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = p (1 - p)$$
3. **Logit derivative with respect to weight $w_j$:**
   $$\frac{\partial z}{\partial w_j} = x_j$$

Multiplying together:
$$\frac{\partial J}{\partial w_j} = \left( \frac{p - y}{p(1 - p)} \right) \cdot \left[ p(1 - p) \right] \cdot x_j = (p - y) x_j$$

In vectorized matrix form:
$$\nabla_w J(w) = \frac{1}{n} X^T (p - y)$$

---

### 3. $L_2$ Regularization (MAP Gaussian Prior)
Adding an $L_2$ penalty corresponds to imposing a zero-mean Gaussian prior $w \sim \mathcal{N}(0, \sigma_0^2)$ on the weights:
$$J_{\text{reg}}(w) = J(w) + \frac{\lambda}{2n} \|w\|_2^2 \implies \nabla_w J_{\text{reg}}(w) = \frac{1}{n} X^T (p - y) + \frac{\lambda}{n} w$$
*(The intercept $b$ is unregularized).*

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_logistic_regression.py` | Standalone vectorized class with stable sigmoid clipping, L2 penalty, and custom metric evaluation (TP, TN, FP, FN, Precision, Recall, F1). |
| `library_logistic_regression.py` | Production Scikit-Learn pipeline with L1/L2 penalties, class weighting, and ROC-AUC scoring. |
| `compare_logistic_regression.py` | Benchmarks on Breast Cancer and Adult Census datasets, comparing scaled vs unscaled convergence. |

## Quick Execution

```bash
python3 scratch_logistic_regression.py
python3 library_logistic_regression.py
python3 compare_logistic_regression.py
```
