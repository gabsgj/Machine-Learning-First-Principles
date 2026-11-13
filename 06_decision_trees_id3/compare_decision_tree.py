"""
Comparison: Decision Tree vs. Logistic Regression & Scratch ID3
==============================================================
Benchmarks:
1. Scratch ID3 vs Scikit-Learn Decision Tree on categorical Adult Income subset.
2. Head-to-Head Architectural Battle: Logistic Regression (Linear Boundary) vs.
   Decision Tree (Orthogonal Axis-Aligned Splits) on Breast Cancer dataset.
3. Visualization: Metrics comparison bar chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scratch_id3_decision_tree import ID3DecisionTreeClassifierScratch


def benchmark_scratch_id3_adult():
    print("==================================================")
    print("1. ID3 SCRATCH VS SKLEARN: Adult Income Dataset")
    print("==================================================")
    data_path = "/Users/gabriel/Projects/ML/data/adult.csv"
    if not os.path.exists(data_path):
        print("adult.csv not found, skipping adult benchmark.")
        return

    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(data_path, header=None, names=cols, skipinitialspace=True).dropna()
    if df['age'].iloc[0] == 'age':
        df = df.iloc[1:]

    # Select representative categorical features
    cat_features = ['workclass', 'education', 'marital-status']
    sample_df = df[cat_features + ['income']].head(1000).copy()

    # Split train/test
    train_df = sample_df.iloc[:700]
    test_df = sample_df.iloc[700:]

    # Scratch ID3
    id3_scratch = ID3DecisionTreeClassifierScratch(max_depth=3)
    id3_scratch.fit(train_df[cat_features], train_df['income'])
    pred_scratch = id3_scratch.predict(test_df[cat_features])
    acc_scratch = accuracy_score(test_df['income'], pred_scratch)

    # Sklearn Decision Tree (with one-hot encoding)
    X_ohe = pd.get_dummies(sample_df[cat_features], drop_first=True)
    y_num = (sample_df['income'].str.contains('>50K')).astype(int)
    X_train_sk, X_test_sk = X_ohe.iloc[:700], X_ohe.iloc[700:]
    y_train_sk, y_test_sk = y_num.iloc[:700], y_num.iloc[700:]

    dt_sk = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    dt_sk.fit(X_train_sk, y_train_sk)
    pred_sk = dt_sk.predict(X_test_sk)
    acc_sk = accuracy_score(y_test_sk, pred_sk)

    print(f"Scratch ID3 Accuracy (Categorical): {acc_scratch * 100:.2f}%")
    print(f"Sklearn CART Accuracy (One-Hot):     {acc_sk * 100:.2f}%")


def benchmark_logistic_regression_vs_decision_tree():
    print("\n==================================================")
    print("2. HEAD-TO-HEAD: Logistic Regression vs Decision Tree")
    print("==================================================")
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize for Logistic Regression
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_std, y_train)
    p_lr = lr.predict(X_test_std)

    # Decision Tree
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    p_dt = dt.predict(X_test)

    models = {"Logistic Regression": p_lr, "Decision Tree (CART)": p_dt}
    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

    print(f"{'Metric':<15} {'Logistic Regression':<22} {'Decision Tree':<15}")
    print("-" * 52)

    for m_name, preds in models.items():
        metrics["Accuracy"].append(accuracy_score(y_test, preds))
        metrics["Precision"].append(precision_score(y_test, preds))
        metrics["Recall"].append(recall_score(y_test, preds))
        metrics["F1-Score"].append(f1_score(y_test, preds))

    for idx, metric_name in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
        val_lr = metrics[metric_name][0]
        val_dt = metrics[metric_name][1]
        print(f"{metric_name:<15} {val_lr:<22.4f} {val_dt:<15.4f}")

    # Plot model comparison
    os.makedirs("plots", exist_ok=True)
    labels = list(models.keys())
    x = np.arange(len(labels))
    width = 0.18

    plt.figure(figsize=(8, 4.8), dpi=150)
    for i, (k, scores) in enumerate(metrics.items()):
        plt.bar(x + i * width, scores, width, label=k)

    plt.xticks(x + width * 1.5, labels, fontsize=10, fontweight="bold")
    plt.ylabel("Score", fontsize=10)
    plt.title("Architectural Comparison: Logistic Regression vs Decision Tree", fontsize=12, fontweight="bold")
    plt.ylim(0.85, 1.02)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/06_dt_vs_logistic_regression.png")
    plt.close()
    print(">> Saved comparison plot to 'plots/06_dt_vs_logistic_regression.png' <<\n")


if __name__ == "__main__":
    benchmark_scratch_id3_adult()
    benchmark_logistic_regression_vs_decision_tree()
