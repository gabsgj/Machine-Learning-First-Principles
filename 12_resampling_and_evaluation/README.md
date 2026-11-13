# Module 12: Resampling Methods & Model Evaluation

## Mathematical Foundations & Derivations

Resampling techniques estimate the generalization error of a predictive model without reserving an expensive, fixed holdout testing split.

---

### 1. Bootstrapping & The 63.2% Rule
Given a dataset $\mathcal{D}$ of size $n$, a bootstrap sample $\mathcal{D}^*$ is constructed by sampling $n$ elements uniformly **with replacement**.

For any specific observation $x_i$, the probability of *not* being chosen in a single random pick is:
$$P(\text{not selected in 1 pick}) = 1 - \frac{1}{n}$$

The probability that $x_i$ is completely omitted after all $n$ independent draws is:
$$P(x_i \notin \mathcal{D}^*) = \left( 1 - \frac{1}{n} \right)^n$$

Taking the limit as dataset size $n \to \infty$:
$$\lim_{n \to \infty} \left( 1 - \frac{1}{n} \right)^n = e^{-1} \approx 0.3679 \quad (\approx 36.8\%)$$

* **In-Bag Sample Size:** ~63.2% of original unique instances.
* **Out-Of-Bag (OOB) Sample Size:** ~36.8% of original unique instances.
Evaluating the fitted model on the unchosen OOB partition provides an unbiased estimate of out-of-sample generalization.

---

### 2. K-Fold Cross-Validation
Data is partitioned into $K$ disjoint, equal-sized subsets $S_1, \dots, S_K$:
$$\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K \text{Loss}(f_{-k}, S_k)$$
where $f_{-k}$ is trained on all folds except fold $k$.
* **Variance vs. Bias:** Choosing $K=5$ or $K=10$ provides a statistically optimal tradeoff: lower variance than Leave-One-Out (LOOCV, $K=n$) and lower bias than small $K$.

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_resampling.py` | Generator routines for Bootstrap with true OOB separation, K-Fold, and Stratified K-Fold. |
| `library_resampling.py` | Scikit-Learn implementation executing official syllabus Experiment 18. |
| `compare_resampling.py` | Head-to-head empirical comparison with boxplot rendering of test accuracy distributions. |

## Quick Execution

```bash
python3 scratch_resampling.py
python3 library_resampling.py
python3 compare_resampling.py
```
