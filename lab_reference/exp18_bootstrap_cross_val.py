"""
Experiment 18: Resampling Strategies: Bootstrapping vs. K-Fold Cross-Validation
===============================================================================
Dataset: Iris
Compares 100 Out-of-Bag (OOB) Bootstrap rounds against 5-Fold Cross-Validation.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion
X, y = load_iris(return_X_y=True)
n = len(X)
model = LogisticRegression(max_iter=1000)

# 2. Bootstrap Resampling with Out-of-Bag (OOB) Generalization Evaluation
boot_scores = []
for i in range(100):
    idx = resample(np.arange(n), replace=True, random_state=i)
    oob = np.setdiff1d(np.arange(n), idx)
    if len(oob) > 0:
        model.fit(X[idx], y[idx])
        boot_scores.append(model.score(X[oob], y[oob]))

# 3. Stratified 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)

# 4. Statistical Summary
print("=== Experiment 18: Resampling Estimation ===")
print(f"Bootstrap OOB Mean Accuracy: {np.mean(boot_scores):.4f} +/- {np.std(boot_scores):.4f}")
print(f"5-Fold CV Mean Accuracy:     {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

# 5. Distribution Comparison Boxplot
plt.figure(figsize=(7, 4.8), dpi=150)
plt.boxplot([boot_scores, cv_scores], tick_labels=['Bootstrap (OOB)', '5-Fold CV'], patch_artist=True,
            boxprops=dict(facecolor="#d1e5f0", color="#2166ac"),
            medianprops=dict(color="#b2182b", linewidth=2.0))
plt.ylabel('Testing Accuracy')
plt.title('Experiment 18: Resampling Performance Distributions (Iris)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('exp18_resampling_boxplot.png')
print("Saved visualization to exp18_resampling_boxplot.png")
