"""
Experiment 09: Decision Tree Induction using Shannon Entropy
============================================================
Dataset: Breast Cancer Wisconsin Diagnostic
Demonstrates Information Gain feature splitting and importance rankings.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Dataset Ingestion & Partitioning
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Model Induction using Entropy (Information Gain)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 3. Model Inference & Metrics
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("=== Experiment 09: Decision Tree Results ===")
print(f"Decision Tree (Entropy) Test Accuracy: {acc*100:.2f}%")

# 4. Feature Importance Hierarchy
top_features = pd.Series(clf.feature_importances_, index=data.feature_names).sort_values(ascending=False).head(3)
print("\nTop 3 Important Splitting Attributes:\n", top_features)
