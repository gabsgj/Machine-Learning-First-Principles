"""
Resampling and Cross-Validation with Scikit-Learn
=================================================
Production implementation of syllabus Experiment 18:
- Bootstrap resampling via sklearn.utils.resample
- Out-Of-Bag (OOB) error computation
- Stratified 5-Fold and 10-Fold cross-validation
- Statistical performance reporting
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.utils import resample


def run_experiment_18_sklearn(n_iterations: int = 100, cv_splits: int = 5):
    """Executes Exp 18 comparison on Iris using Scikit-Learn."""
    X, y = load_iris(return_X_y=True)
    n = len(X)
    model = LogisticRegression(max_iter=1000, random_state=42)

    # 1. Bootstrap with Out-of-Bag (OOB) evaluation
    boot_scores = []
    for i in range(n_iterations):
        idx = resample(np.arange(n), replace=True, random_state=i)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) > 0:
            model.fit(X[idx], y[idx])
            boot_scores.append(model.score(X[oob], y[oob]))

    # 2. Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf)

    return {
        "bootstrap_mean": np.mean(boot_scores),
        "bootstrap_std": np.std(boot_scores),
        "bootstrap_raw": boot_scores,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "cv_raw": cv_scores
    }


if __name__ == "__main__":
    print("=== Demo: EXP 18 Scikit-Learn Resampling on Iris ===")
    res = run_experiment_18_sklearn(n_iterations=100, cv_splits=5)
    print(f"Bootstrap (100 runs, OOB) Mean Accuracy: {res['bootstrap_mean']:.4f} +/- {res['bootstrap_std']:.4f}")
    print(f"5-Fold Cross-Validation Mean Accuracy:   {res['cv_mean']:.4f} +/- {res['cv_std']:.4f}")
