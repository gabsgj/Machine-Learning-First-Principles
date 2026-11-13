# Module 09: Parameter Estimation (MLE vs. MAP)

## Mathematical Foundations & Derivations

Parameter estimation infers the latent parameters $\theta$ of a probability distribution from observed empirical data $\mathcal{D} = \{x_1, \dots, x_n\}$.

---

### 1. Maximum Likelihood Estimation (MLE)
MLE treats parameters $\theta$ as fixed, unknown constants and selects the parameter value that maximizes the probability of generating the observed data:
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta P(\mathcal{D} \mid \theta) = \arg\max_\theta \sum_{i=1}^n \ln P(x_i \mid \theta)$$

For a **Multinomial Distribution** over vocabulary counts $(c_1, \dots, c_V)$ with total tokens $N = \sum_{j=1}^V c_j$:
$$\mathcal{L}(\theta) = \sum_{j=1}^V c_j \ln(\theta_j) \quad \text{subject to } \sum_{j=1}^V \theta_j = 1$$

Applying Lagrange multipliers:
$$\Lambda(\theta, \lambda) = \sum_{j=1}^V c_j \ln(\theta_j) - \lambda \left(\sum_{j=1}^V \theta_j - 1\right)$$
$$\frac{\partial \Lambda}{\partial \theta_j} = \frac{c_j}{\theta_j} - \lambda = 0 \implies \theta_j = \frac{c_j}{\lambda}$$
Summing over $j$: $\sum c_j = \lambda \sum \theta_j \implies \lambda = N$. Hence:
$$\hat{\theta}_{j, \text{MLE}} = \frac{c_j}{N}$$

> [!WARNING]
> **The Zero-Count Problem:** If a word never appeared in training data ($c_j = 0$), $\hat{\theta}_{j, \text{MLE}} = 0$. When evaluating a test sentence containing this word, the joint likelihood collapses to $0.0$, regardless of all other evidence.

---

### 2. Maximum A Posteriori (MAP) Estimation
MAP adopts a Bayesian paradigm, placing a prior distribution $P(\theta)$ over parameters:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta [ \ln P(\mathcal{D} \mid \theta) + \ln P(\theta) ]$$

For the Multinomial likelihood, the conjugate prior is the **Dirichlet Distribution**:
$$P(\theta \mid \alpha) = \frac{1}{\text{B}(\alpha)} \prod_{j=1}^V \theta_j^{\alpha_j - 1}$$
The posterior distribution is also Dirichlet:
$$P(\theta \mid \mathcal{D}) \propto \prod_{j=1}^V \theta_j^{c_j + \alpha_j - 1}$$

Taking the mode (argmax) under the simplex constraint $\sum \theta_j = 1$:
$$\hat{\theta}_{j, \text{MAP}} = \frac{c_j + \alpha_j - 1}{N + \sum_{k=1}^V (\alpha_k - 1)}$$

For a symmetric prior where $\alpha_j = \alpha$:
$$\hat{\theta}_{j, \text{MAP}} = \frac{c_j + \alpha - 1}{N + V(\alpha - 1)}$$
* When $\alpha = 1$: MAP reduces exactly to MLE ($\frac{c_j}{N}$).
* When $\alpha = 2$: MAP yields **Laplace add-one smoothing**: $\hat{\theta}_{j} = \frac{c_j + 1}{N + V}$.

---

### 3. MAP Equivalence to Regularization
When estimating weights $w$ in linear or logistic models:
* A **Gaussian Prior** $w \sim \mathcal{N}(0, \sigma_0^2 I) \iff \mathbf{L_2}$ **Regularization (Ridge)** ($\lambda = \frac{1}{\sigma_0^2}$).
* A **Laplace Prior** $w \sim \text{Laplace}(0, b) \iff \mathbf{L_1}$ **Regularization (Lasso)** ($\lambda = \frac{1}{b}$).

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_mle_map_estimation.py` | Standalone implementation of Multinomial MLE, Dirichlet MAP, and Beta-Binomial coin toss estimator. |
| `compare_mle_map.py` | Reproduces syllabus EXP 5 (20 Newsgroups word probabilities plot) and EXP 4 (Logistic Regression parameter shrinkage). |

## Quick Execution

```bash
python3 scratch_mle_map_estimation.py
python3 compare_mle_map.py
```
