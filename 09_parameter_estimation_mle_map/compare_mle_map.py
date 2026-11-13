"""
Comparison: MLE vs. MAP Estimation Across Discrete & Continuous Domains
=======================================================================
Benchmarks:
1. EXP 5: Multinomial word counts from 20 Newsgroups under Dirichlet prior.
2. EXP 4: Logistic Regression MLE (unregularized) vs. MAP (L2 Gaussian prior).
3. Visual comparison plot of probability distributions across vocabulary tokens.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups, load_breast_cancer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from scratch_mle_map_estimation import multinomial_mle, multinomial_map_dirichlet


def run_exp5_multinomial_text_estimation():
    print("==================================================")
    print("EXP 5: MLE vs MAP Multinomial Estimation (20 Newsgroups)")
    print("==================================================")
    data = fetch_20newsgroups(subset='train')
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(data.data).toarray()

    counts = X.sum(axis=0)
    total = counts.sum()

    mle = multinomial_mle(counts)
    alpha = 2.0
    map_est = multinomial_map_dirichlet(counts, alpha=alpha)

    print(f"Vocabulary size: {len(counts)} | Total word count: {total}")
    print(f"MLE sum of probabilities: {mle.sum():.6f}")
    print(f"MAP sum of probabilities: {map_est.sum():.6f}")

    words = vectorizer.get_feature_names_out()
    print(f"\n{'Rank':<5} {'Word':<15} {'Count':<8} {'MLE Prob':<15} {'MAP Prob (alpha=2)':<20}")
    print("-" * 65)
    for i in range(10):
        print(f"{i+1:<5} {words[i]:<15} {counts[i]:<8.0f} {mle[i]:<15.6f} {map_est[i]:<20.6f}")

    # Plot top 10 words (Reproducing EXP 5 from syllabus)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(mle[:10], label="MLE (Raw Frequencies)", marker="o", color="#1f77b4", lw=2)
    plt.plot(map_est[:10], label=f"MAP (Dirichlet Prior alpha={alpha})", marker="s", color="#2ca02c", lw=2)

    plt.xlabel("Vocabulary Words", fontsize=10)
    plt.ylabel("Estimated Probability", fontsize=10)
    plt.title("Parameter Estimation: MLE vs. MAP under Dirichlet Conjugacy", fontsize=12, fontweight="bold")
    plt.xticks(range(10), words[:10], rotation=35, ha="right", fontsize=9)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/09_mle_vs_map_probabilities.png")
    plt.close()
    print(">> Saved probability curve to 'plots/09_mle_vs_map_probabilities.png' <<\n")


def run_exp4_logistic_regression_mle_vs_map():
    print("==================================================")
    print("EXP 4: Logistic Regression MLE vs MAP (Breast Cancer)")
    print("==================================================")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # 1. MLE: Unregularized Logistic Regression (C -> inf)
    mle_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
    mle_model.fit(X_train_std, y_train)
    p_mle = mle_model.predict(X_test_std)

    # 2. MAP: L2 Regularized with Gaussian Prior (C = 1.0)
    map_model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    map_model.fit(X_train_std, y_train)
    p_map = map_model.predict(X_test_std)

    w_mle_norm = np.linalg.norm(mle_model.coef_)
    w_map_norm = np.linalg.norm(map_model.coef_)

    print(f"MLE (Unregularized)  | Accuracy: {accuracy_score(y_test, p_mle)*100:.2f}% | Weight Norm ||w||: {w_mle_norm:.3f}")
    print(f"MAP (Gaussian Prior) | Accuracy: {accuracy_score(y_test, p_map)*100:.2f}% | Weight Norm ||w||: {w_map_norm:.3f}")
    print(f">> MAP imposes prior shrinkage, reducing weight norm from {w_mle_norm:.2f} down to {w_map_norm:.2f}!")


if __name__ == "__main__":
    run_exp5_multinomial_text_estimation()
    run_exp4_logistic_regression_mle_vs_map()
