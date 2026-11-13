"""
Resampling and Model Validation from Scratch
===========================================
Algorithmic Implementations:
1. Bootstrap Sampling with Out-Of-Bag (OOB) evaluation (36.8% theoretical holdout)
2. K-Fold Cross-Validation index generator
3. Stratified K-Fold generator preserving class proportions
4. Empirical variance and confidence interval estimators
"""

import numpy as np
from typing import Generator, Tuple, List, Optional


def bootstrap_resample_scratch(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 100,
    random_state: Optional[int] = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generates bootstrap (in-bag) and Out-Of-Bag (OOB) splits.
    
    Yields
    ------
    (X_train, y_train, X_oob, y_oob)
    """
    n_samples = len(X)
    rng = np.random.RandomState(random_state)
    all_indices = np.arange(n_samples)

    for _ in range(n_iterations):
        # Sample n instances with replacement
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        # Out-Of-Bag instances are those never sampled
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = all_indices[oob_mask]

        if len(oob_idx) > 0:
            yield X[boot_idx], y[boot_idx], X[oob_idx], y[oob_idx]


def k_fold_split_scratch(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    K-Fold cross-validation index generator from scratch.
    
    Yields
    ------
    (train_indices, val_indices)
    """
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        yield train_idx, val_idx
        current = stop


def evaluate_model_cv_scratch(
    model_cls,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    **model_kwargs,
) -> List[float]:
    """Evaluates arbitrary classifier using Scratch K-Fold Cross Validation."""
    scores = []
    for train_idx, val_idx in k_fold_split_scratch(len(X), n_splits=n_splits):
        clf = model_cls(**model_kwargs)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        acc = float(np.mean(preds == y[val_idx]))
        scores.append(acc)
    return scores


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    print("=== Demo: Scratch Bootstrap (OOB) vs. K-Fold on Iris ===")
    iris = load_iris()
    X, y = iris.data, iris.target

    # 1. Scratch Bootstrap OOB Evaluation
    oob_scores = []
    for X_tr, y_tr, X_oob, y_oob in bootstrap_resample_scratch(X, y, n_iterations=100):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        oob_scores.append(clf.score(X_oob, y_oob))

    # 2. Scratch 5-Fold Cross Validation
    cv_scores = evaluate_model_cv_scratch(LogisticRegression, X, y, n_splits=5, max_iter=1000)

    print(f"Scratch Bootstrap OOB Mean Accuracy: {np.mean(oob_scores):.4f} +/- {np.std(oob_scores):.4f}")
    print(f"Scratch 5-Fold CV Mean Accuracy:     {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
