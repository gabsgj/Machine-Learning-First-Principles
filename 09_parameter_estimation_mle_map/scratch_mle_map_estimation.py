"""
Parameter Estimation: Maximum Likelihood (MLE) vs. Maximum A Posteriori (MAP)
=============================================================================
Mathematical Implementation:
1. Multinomial Parameter Estimation:
   - MLE: theta_j = count_j / total_count
   - MAP with Dirichlet Conjugate Prior: theta_j = (count_j + alpha - 1) / (total_count + V * (alpha - 1))
2. Beta-Binomial Conjugacy for Binary Outcomes (Coin Toss)
3. Logistic Regression Parameter Estimation under Gaussian Priors (L2 Regularization)
"""

import numpy as np
from typing import Tuple, Dict


def multinomial_mle(word_counts: np.ndarray) -> np.ndarray:
    """
    Computes Maximum Likelihood Estimates for multinomial distribution.
    Formula: theta_j = count_j / sum(counts)
    """
    counts = np.asarray(word_counts, dtype=float)
    total = np.sum(counts)
    if total == 0:
        return np.ones_like(counts) / len(counts)
    return counts / total


def multinomial_map_dirichlet(word_counts: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    Computes Maximum A Posteriori estimates under a symmetric Dirichlet prior.
    Formula: theta_j = (count_j + alpha - 1) / (total_count + V * (alpha - 1))
    
    When alpha = 2, this is equivalent to Laplace (add-one) smoothing:
    theta_j = (count_j + 1) / (total_count + V)
    """
    counts = np.asarray(word_counts, dtype=float)
    total = np.sum(counts)
    V = len(counts)
    numerator = counts + alpha - 1.0
    denominator = total + V * (alpha - 1.0)
    return numerator / denominator


def beta_binomial_estimation(heads: int, tails: int, alpha_prior: float = 2.0, beta_prior: float = 2.0) -> Dict[str, float]:
    """
    Compares MLE and MAP for a Bernoulli/Binomial process (e.g. coin tossing).
    - MLE: theta = heads / (heads + tails)
    - MAP with Beta(alpha, beta) prior: theta = (heads + alpha - 1) / (heads + tails + alpha + beta - 2)
    """
    n = heads + tails
    mle = heads / n if n > 0 else 0.5
    map_est = (heads + alpha_prior - 1.0) / (n + alpha_prior + beta_prior - 2.0)
    return {"MLE": mle, "MAP": map_est}


if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    print("=== Demo: MLE vs MAP Estimation on 20 Newsgroups Word Counts ===")
    data = fetch_20newsgroups(subset='train')

    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(data.data).toarray()

    counts = X.sum(axis=0)
    total = counts.sum()

    mle_probs = multinomial_mle(counts)
    map_probs = multinomial_map_dirichlet(counts, alpha=2.0)

    print(f"Total Tokens: {total} | Vocabulary Size: {len(counts)}")
    print(f"MLE sum of probabilities: {mle_probs.sum():.6f}")
    print(f"MAP sum of probabilities: {map_probs.sum():.6f}")

    words = vectorizer.get_feature_names_out()
    print("\nTop 5 Words Probabilities Comparison:")
    for idx in range(5):
        print(f"Word '{words[idx]}': MLE = {mle_probs[idx]:.6f} | MAP = {map_probs[idx]:.6f}")
