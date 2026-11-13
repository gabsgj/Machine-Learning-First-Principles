# Module 08: Naive Bayes Classifiers for Text Categorization

## Mathematical Foundations & Derivations

The Naive Bayes family of generative probabilistic models applies Bayes' Theorem under the conditional independence assumption that all features $x_j$ are mutually independent given class label $y = c$:
$$P(x_1, \dots, x_d \mid y = c) = \prod_{j=1}^d P(x_j \mid y = c)$$

By Bayes' Theorem:
$$P(y = c \mid x) = \frac{P(y = c) \prod_{j=1}^d P(x_j \mid y = c)}{P(x)}$$

In log-probability space (preventing arithmetic underflow caused by multiplying many tiny probabilities):
$$\hat{y} = \arg\max_{c \in \mathcal{C}} \left[ \ln P(y = c) + \sum_{j=1}^d \ln P(x_j \mid y = c) \right]$$

---

### 1. Multinomial Naive Bayes (Word Counts)
Suitable for discrete count data (e.g., bag-of-words text vectors). With Laplace add-1 smoothing ($\alpha = 1$):
$$P(x_j \mid y = c) = \frac{\sum_{i \in c} X_{ij} + \alpha}{\sum_{k=1}^d \sum_{i \in c} X_{ik} + \alpha \cdot d}$$
* **Numerator:** Total occurrences of token $j$ in class $c$ plus smoothing.
* **Denominator:** Total tokens across all vocabulary words in class $c$ plus smoothed vocabulary size.

---

### 2. Bernoulli Naive Bayes (Binary Word Presence)
Models boolean word presence $x_j \in \{0, 1\}$:
$$P(x \mid y = c) = \prod_{j=1}^d p_{cj}^{x_j} (1 - p_{cj})^{(1 - x_j)}$$
where $p_{cj} = P(x_j = 1 \mid y = c) = \frac{N_{cj} + \alpha}{N_c + 2\alpha}$. Note that Bernoulli NB explicitly penalizes the **absence** of words characteristic of class $c$.

---

## File Manifest

| File | Description |
| :--- | :--- |
| `scratch_naive_bayes.py` | Vectorized NumPy implementations of Multinomial and Bernoulli Naive Bayes with Laplace smoothing. |
| `library_naive_bayes.py` | Production NLP pipeline utilizing Scikit-Learn `CountVectorizer`, `MultinomialNB`, and `BernoulliNB`. |
| `compare_naive_bayes.py` | Benchmarks on 20 Newsgroups (4 topics: tech, medicine, sports, politics), validating parity and accuracy. |

## Quick Execution

```bash
python3 scratch_naive_bayes.py
python3 library_naive_bayes.py
python3 compare_naive_bayes.py
```
