"""
Comparison: Bootstrapping vs. K-Fold Cross-Validation (EXP 18)
==============================================================
Direct execution of syllabus Experiment 18:
1. Bootstrapping with true Out-Of-Bag (OOB) error estimation (100 resamples).
2. 5-Fold Cross-Validation on the Iris dataset.
3. Statistical variance analysis and Boxplot distribution rendering.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample

from scratch_resampling import bootstrap_resample_scratch, evaluate_model_cv_scratch


def run_experiment_18_benchmark():
    print("==================================================")
    print("EXP 18: Bootstrap vs. Cross-Validation on Iris")
    print("==================================================")
    X, y = load_iris(return_X_y=True)
    n = len(X)
    model = LogisticRegression(max_iter=1000, random_state=42)

    # 1. Bootstrap Resampling (100 iterations with OOB evaluation)
    boot_scores = []
    for i in range(100):
        idx = resample(np.arange(n), replace=True, random_state=i)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) > 0:
            model.fit(X[idx], y[idx])
            boot_scores.append(model.score(X[oob], y[oob]))

    # 2. 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"Bootstrap Mean Accuracy: {np.mean(boot_scores):.4f} +/- {np.std(boot_scores):.4f}")
    print(f"5-Fold CV Mean Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Render Boxplot (reproducing WhatsApp Exp 18 visual)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(7, 4.8), dpi=150)
    plt.boxplot([boot_scores, cv_scores], tick_labels=["Bootstrap (OOB)", "5-Fold CV"], patch_artist=True,
                boxprops=dict(facecolor="#d1e5f0", color="#2166ac"),
                medianprops=dict(color="#b2182b", linewidth=2.0))

    plt.ylabel("Testing Accuracy", fontsize=10, fontweight="bold")
    plt.title("Resampling Performance Distributions (Iris Dataset)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/12_resampling_boxplot.png")
    plt.close()
    print(">> Saved comparison boxplot to 'plots/12_resampling_boxplot.png' <<\n")


if __name__ == "__main__":
    run_experiment_18_benchmark()
