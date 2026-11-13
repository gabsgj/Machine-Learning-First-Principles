"""
Experiment 19: Ensemble Learning: Bagging and Boosting vs. Single Decision Tree
==============================================================================
Dataset: Breast Cancer Wisconsin Diagnostic
Compares Single Tree vs. BaggingClassifier vs. AdaBoostClassifier.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion & Partitioning (80:20)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Committee Models
models = {
    'Single Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Bagging':     BaggingClassifier(n_estimators=100, random_state=42),
    'AdaBoost':    AdaBoostClassifier(n_estimators=100, random_state=42)
}

# 3. Model Training & Metric Accumulation
acc, pre, rec, f1 = [], [], [], []

print("=== Experiment 19: Ensemble Methods Comparison ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc.append(accuracy_score(y_test, pred))
    pre.append(precision_score(y_test, pred))
    rec.append(recall_score(y_test, pred))
    f1.append(f1_score(y_test, pred))

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc[-1]:.4f}")
    print(f"Precision: {pre[-1]:.4f}")
    print(f"Recall:    {rec[-1]:.4f}")
    print(f"F1-Score:  {f1[-1]:.4f}")

# 4. Multi-Metric Trajectory Plot
plt.figure(figsize=(7.5, 4.8), dpi=150)
plt.plot(acc, label='Accuracy', marker='o', lw=2)
plt.plot(pre, label='Precision', marker='s', lw=2)
plt.plot(rec, label='Recall', marker='^', lw=2)
plt.plot(f1, label='F1 Score', marker='d', lw=2)

plt.xticks([0, 1, 2], ['Single Tree', 'Bagging', 'AdaBoost'], fontweight='bold')
plt.ylabel('Evaluation Score')
plt.title('Experiment 19: Ensemble Committee Generalization')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('exp19_ensemble_comparison.png')
print("Saved visualization to exp19_ensemble_comparison.png")
