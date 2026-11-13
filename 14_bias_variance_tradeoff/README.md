# Module 14: The Bias-Variance Tradeoff

## Mathematical Foundations & Rigorous Derivation

The Bias-Variance Tradeoff explains the generalization error of any supervised learning estimator.

---

### 1. Step-by-Step Mathematical Proof
Assume the true data-generating process is:
$$y = f(x) + \epsilon, \quad \text{where } \mathbb{E}[\epsilon] = 0 \text{ and } \text{Var}(\epsilon) = \sigma^2$$
Let $\hat{f}(x)$ be the estimator trained on a finite realization dataset $\mathcal{D}$. The expected squared error at query point $x$ over all possible training realizations is:
$$\mathbb{E}_\mathcal{D} \left[ (y - \hat{f}(x))^2 \right]$$

Substitute $y = f(x) + \epsilon$:
$$\mathbb{E} \left[ (f(x) + \epsilon - \hat{f}(x))^2 \right] = \mathbb{E} \left[ \Big( (f(x) - \mathbb{E}[\hat{f}(x)]) + (\mathbb{E}[\hat{f}(x)] - \hat{f}(x)) + \epsilon \Big)^2 \right]$$

Expanding the trinomial $(A + B + C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$:
1. $A^2 = (f(x) - \mathbb{E}[\hat{f}(x)])^2 \equiv \mathbf{\text{Bias}^2(\hat{f}(x))}$ (Deterministic constant).
2. $\mathbb{E}[B^2] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] \equiv \mathbf{\text{Var}(\hat{f}(x))}$.
3. $\mathbb{E}[C^2] = \mathbb{E}[\epsilon^2] \equiv \mathbf{\sigma^2}$ (Irreducible Noise).
4. Cross-term $\mathbb{E}[2AB] = 2(f(x) - \mathbb{E}[\hat{f}(x)]) \cdot \mathbb{E}[\mathbb{E}[\hat{f}(x)] - \hat{f}(x)] = 2 A \cdot 0 = 0$.
5. Cross-term $\mathbb{E}[2AC] = 2(f(x) - \mathbb{E}[\hat{f}(x)]) \cdot \mathbb{E}[\epsilon] = 2 A \cdot 0 = 0$.
6. Cross-term $\mathbb{E}[2BC] = 2 \mathbb{E}[\mathbb{E}[\hat{f}(x)] - \hat{f}(x)] \cdot \mathbb{E}[\epsilon] = 0$ (due to statistical independence of noise $\epsilon$).

Combining the non-zero terms completes the proof:
$$\mathbf{\mathbb{E} \left[ (y - \hat{f}(x))^2 \right] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2}$$

---

### 2. The U-Shaped Validation Curve
* **Underfitting Region (High Bias, Low Variance):** The hypothesis class lacks capacity to capture true functional curvature (e.g. linear model on sinusoidal data). Both training error and validation error are high.
* **Overfitting Region (Low Bias, High Variance):** The hypothesis class has excessive capacity (e.g. degree 10 polynomial). Training error monotonically drops toward 0, but variance explodes, causing validation error to diverge sharply.
* **Optimal Complexity:** The minimum of the validation error curve represents the optimal balance where generalization is maximized.

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_bias_variance.py` | Monte-Carlo simulation quantifying exact numerical Bias$^2$, Variance, and Noise across polynomial degrees. |
| `compare_bias_variance.py` | Direct execution of syllabus Experiment 20 on Diabetes BMI, plotting the U-shaped error curve. |

## Quick Execution

```bash
python3 scratch_bias_variance.py
python3 compare_bias_variance.py
```
