"""
Experiment 10: Linear vs. Axis-Aligned Decision Boundaries
==========================================================
Dataset: Breast Cancer Wisconsin Diagnostic
Compares Logistic Regression (linear boundary) against Decision Tree (axis-aligned orthogonal splits).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Dataset Ingestion & Partitioning
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Model Definitions
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    'Decision Tree (CART)': DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
}

# 3. Model Comparison
print("=== Experiment 10: Architectural Model Comparison ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall:    {recall_score(y_test, pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, pred):.4f}")
