"""
Comparison and Benchmark: Naive Bayes (Scratch vs. Scikit-Learn)
================================================================
Side-by-side evaluation:
1. Exact numerical agreement between Scratch and Sklearn estimators.
2. Comparative study: Multinomial NB (word frequencies) vs. Bernoulli NB (binary occurrence).
3. 2-Topic and 4-Topic classification benchmarks on 20 Newsgroups.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

from scratch_naive_bayes import MultinomialNaiveBayesScratch, BernoulliNaiveBayesScratch


def benchmark_20newsgroups_topics():
    print("==================================================")
    print("NAIVE BAYES BENCHMARK: 20 Newsgroups Text Classification")
    print("==================================================")
    cats = ['sci.med', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
    train_data = fetch_20newsgroups(subset='train', categories=cats)
    test_data = fetch_20newsgroups(subset='test', categories=cats)

    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X_train = vectorizer.fit_transform(train_data.data).toarray()
    X_test = vectorizer.transform(test_data.data).toarray()
    y_train = train_data.target
    y_test = test_data.target

    # 1. Multinomial NB: Scratch vs Sklearn
    mnb_scratch = MultinomialNaiveBayesScratch(alpha=1.0).fit(X_train, y_train)
    p_mnb_sc = mnb_scratch.predict(X_test)
    acc_mnb_sc = accuracy_score(y_test, p_mnb_sc)

    mnb_sklearn = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    p_mnb_sk = mnb_sklearn.predict(X_test)
    acc_mnb_sk = accuracy_score(y_test, p_mnb_sk)

    diff_mnb = int(np.sum(p_mnb_sc != p_mnb_sk))

    # 2. Bernoulli NB: Scratch vs Sklearn
    bnb_scratch = BernoulliNaiveBayesScratch(alpha=1.0).fit(X_train, y_train)
    p_bnb_sc = bnb_scratch.predict(X_test)
    acc_bnb_sc = accuracy_score(y_test, p_bnb_sc)

    bnb_sklearn = BernoulliNB(alpha=1.0).fit(X_train, y_train)
    p_bnb_sk = bnb_sklearn.predict(X_test)
    acc_bnb_sk = accuracy_score(y_test, p_bnb_sk)

    diff_bnb = int(np.sum(p_bnb_sc != p_bnb_sk))

    print(f"{'Model':<30} {'Scratch Acc':<15} {'Sklearn Acc':<15} {'Disagreements':<15}")
    print("-" * 75)
    print(f"{'Multinomial Naive Bayes':<30} {acc_mnb_sc*100:<15.2f}% {acc_mnb_sk*100:<15.2f}% {diff_mnb:<15}")
    print(f"{'Bernoulli Naive Bayes':<30} {acc_bnb_sc*100:<15.2f}% {acc_bnb_sk*100:<15.2f}% {diff_bnb:<15}")

    # Plot
    os.makedirs("plots", exist_ok=True)
    models = ["Multinomial NB", "Bernoulli NB"]
    scores_scratch = [acc_mnb_sc * 100, acc_bnb_sc * 100]
    scores_sklearn = [acc_mnb_sk * 100, acc_bnb_sk * 100]

    x = np.arange(len(models))
    width = 0.3

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.bar(x - width/2, scores_scratch, width, label="Scratch (NumPy)", color="#1f77b4")
    plt.bar(x + width/2, scores_sklearn, width, label="Scikit-Learn", color="#2ca02c")

    plt.title("Naive Bayes: 4-Topic Text Categorization Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Model Variant", fontsize=10)
    plt.ylabel("Testing Accuracy (%)", fontsize=10)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/08_naive_bayes_comparison.png")
    plt.close()
    print(">> Saved benchmark plot to 'plots/08_naive_bayes_comparison.png' <<\n")


if __name__ == "__main__":
    benchmark_20newsgroups_topics()
