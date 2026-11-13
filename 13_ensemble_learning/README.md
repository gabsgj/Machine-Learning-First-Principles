# Module 13: Ensemble Methods (Bagging, Boosting & Forests)

## Mathematical Foundations & Derivations

Ensemble methods combine collections of base hypothesis functions $\{h_1(x), \dots, h_M(x)\}$ into an aggregated predictive model $H(x)$ exhibiting superior generalization.

---

### 1. Bagging & The Mechanics of Variance Reduction
Let each base estimator have variance $\text{Var}(h_m(x)) = \sigma^2$, with pairwise correlation $\rho = \text{Corr}(h_i(x), h_j(x))$. The variance of the simple average ensemble is:
$$\text{Var}\left( \frac{1}{M} \sum_{m=1}^M h_m(x) \right) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

* As $M \to \infty$, the second term $\frac{1-\rho}{M} \sigma^2 \to 0$.
* The remaining irreducible variance is $\rho \sigma^2$.
* **Random Forest Innovation:** By restricting split candidates to a random subset of $m \approx \sqrt{d}$ features, Random Forests actively de-correlate the trees (reducing $\rho$), further decreasing total ensemble variance.

---

### 2. AdaBoost Derivation (Exponential Loss Minimization)
AdaBoost sequentially minimizes the exponential loss:
$$J(H) = \sum_{i=1}^n \exp\left( -y_i \sum_{m=1}^M \alpha_m h_m(x_i) \right)$$

At stage $m$, we seek base learner $h_m$ and weight $\alpha_m$ minimizing:
$$\epsilon_m = \frac{\sum_{i: y_i \ne h_m(x_i)} w_i^{(m)}}{\sum_{i=1}^n w_i^{(m)}}$$

Differentiating with respect to $\alpha_m$ and setting to zero yields the closed-form weight:
$$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$

Instance weights are updated via:
$$w_i^{(m+1)} = w_i^{(m)} \exp\left( -\alpha_m y_i h_m(x_i) \right)$$
Instances misclassified by $h_m$ receive positive exponent multipliers, forcing subsequent learners to prioritize difficult samples.

---

### 3. Bagging vs. Boosting Comparison Matrix
| Characteristic | Bagging (Bootstrap Aggregating) | Boosting (AdaBoost, Gradient Boost) |
| :--- | :--- | :--- |
| **Architecture** | Parallel independent training | Sequential dependent training |
| **Target Error Component** | Primary **Variance** reduction | Primary **Bias** reduction |
| **Base Estimators** | High variance, deep trees | High bias, shallow stumps ($\text{depth}=1, 2$) |
| **Overfitting Risk** | Immune to overfitting as $M \to \infty$ | Vulnerable to noise/outlier overfitting if $M$ too high |

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_ensemble.py` | Full implementations of `BaggingClassifierScratch` and `AdaBoostClassifierScratch` with custom instance weight updating. |
| `library_ensemble.py` | Scikit-Learn production suite: Single Tree, Bagging, AdaBoost, and Random Forest. |
| `compare_ensemble.py` | Official Experiment 19 benchmark comparing Single Tree vs Bagging vs AdaBoost with metric curve visualization. |

## Quick Execution

```bash
python3 scratch_ensemble.py
python3 library_ensemble.py
python3 compare_ensemble.py
```
