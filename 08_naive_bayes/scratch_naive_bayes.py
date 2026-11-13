"""
Naive Bayes Classifiers from Scratch
====================================
Vectorized implementations of Bayesian Generative Models:
1. Multinomial Naive Bayes (Word frequency counts with Laplace smoothing)
2. Bernoulli Naive Bayes (Binary feature occurrences)
3. Gaussian Naive Bayes (Continuous feature likelihoods)
"""

import numpy as np
from typing import Optional, Union


class MultinomialNaiveBayesScratch:
    """
    Multinomial Naive Bayes classifier for discrete word count features.
    
    Formula:
        P(x_j | c) = (Count(w_j, c) + alpha) / (Total_words(c) + alpha * |V|)
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.log_priors_: Optional[np.ndarray] = None
        self.log_likelihoods_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayesScratch":
        X_mat = np.asarray(X, dtype=float)
        y_vec = np.asarray(y, dtype=int).ravel()
        
        self.classes_ = np.unique(y_vec)
        n_classes = len(self.classes_)
        n_features = X_mat.shape[1]
        
        self.log_priors_ = np.zeros(n_classes)
        self.log_likelihoods_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X_mat[y_vec == c]
            # Prior: P(c) = N_c / N
            self.log_priors_[idx] = np.log(len(X_c) / len(y_vec))
            
            # Word counts in class c
            word_counts = X_c.sum(axis=0)
            total_words = word_counts.sum()
            
            # Laplace Smoothed Likelihood: P(x_j | c) = (count + alpha) / (total + alpha * V)
            smoothed_prob = (word_counts + self.alpha) / (total_words + self.alpha * n_features)
            self.log_likelihoods_[idx] = np.log(smoothed_prob)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.log_priors_ is None or self.log_likelihoods_ is None:
            raise RuntimeError("Model is not fitted.")
        X_mat = np.asarray(X, dtype=float)
        
        # Log-posterior: X @ log_likelihoods.T + log_priors
        log_posteriors = (X_mat @ self.log_likelihoods_.T) + self.log_priors_
        best_class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[best_class_indices]


class BernoulliNaiveBayesScratch:
    """
    Bernoulli Naive Bayes for binary feature occurrences {0, 1}.
    """
    def __init__(self, alpha: float = 1.0, binarize_threshold: float = 0.0):
        self.alpha = alpha
        self.binarize_threshold = binarize_threshold
        self.classes_: Optional[np.ndarray] = None
        self.log_priors_: Optional[np.ndarray] = None
        self.feature_prob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNaiveBayesScratch":
        X_mat = (np.asarray(X, dtype=float) > self.binarize_threshold).astype(float)
        y_vec = np.asarray(y, dtype=int).ravel()
        
        self.classes_ = np.unique(y_vec)
        n_classes = len(self.classes_)
        n_features = X_mat.shape[1]
        
        self.log_priors_ = np.zeros(n_classes)
        self.feature_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X_mat[y_vec == c]
            self.log_priors_[idx] = np.log(len(X_c) / len(y_vec))
            doc_counts = X_c.sum(axis=0)
            # P(x_j = 1 | c) = (doc_count + alpha) / (N_c + 2 * alpha)
            p_j = (doc_counts + self.alpha) / (len(X_c) + 2.0 * self.alpha)
            self.feature_prob_[idx] = p_j

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_mat = (np.asarray(X, dtype=float) > self.binarize_threshold).astype(float)
        n_samples = len(X_mat)
        n_classes = len(self.classes_)
        
        log_posteriors = np.zeros((n_samples, n_classes))
        for idx in range(n_classes):
            p = self.feature_prob_[idx]
            # log P(x | c) = sum(x_j * log(p_j) + (1 - x_j) * log(1 - p_j))
            log_lik = X_mat @ np.log(p) + (1.0 - X_mat) @ np.log(1.0 - p)
            log_posteriors[:, idx] = log_lik + self.log_priors_[idx]

        return self.classes_[np.argmax(log_posteriors, axis=1)]


if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    print("=== Demo: Naive Bayes from Scratch on 20 Newsgroups ===")
    cats = ['sci.med', 'comp.graphics']
    train_data = fetch_20newsgroups(subset='train', categories=cats)
    test_data = fetch_20newsgroups(subset='test', categories=cats)

    vectorizer = CountVectorizer(max_features=500, stop_words='english')
    X_train = vectorizer.fit_transform(train_data.data).toarray()
    X_test = vectorizer.transform(test_data.data).toarray()
    y_train = train_data.target
    y_test = test_data.target

    # Multinomial NB
    mnb = MultinomialNaiveBayesScratch(alpha=1.0)
    mnb.fit(X_train, y_train)
    p_mnb = mnb.predict(X_test)
    print(f"Scratch Multinomial NB Accuracy: {np.mean(p_mnb == y_test) * 100:.2f}%")

    # Bernoulli NB
    bnb = BernoulliNaiveBayesScratch(alpha=1.0)
    bnb.fit(X_train, y_train)
    p_bnb = bnb.predict(X_test)
    print(f"Scratch Bernoulli NB Accuracy:   {np.mean(p_bnb == y_test) * 100:.2f}%")
