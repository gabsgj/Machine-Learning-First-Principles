"""
Experiment 08: K-Nearest Neighbors Classifier with Parameter Tuning
===================================================================
Dataset: Optical Recognition of Handwritten Digits
Sweeps neighbor count K across odd integers [1, 3, 5, 7, 9].
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Dataset Ingestion & Min-Max Pixel Normalization to [0, 1]
X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Evaluation across K
print("=== Experiment 08: KNN Performance across K ===")
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred) * 100
    print(f"Number of Neighbors (K={k}) | Test Accuracy = {acc:.2f}%")
