"""
Experiment 13: Multilayer Perceptron Architectural Variations
=============================================================
Dataset: Wine Quality Classification
Compares feed-forward networks across single and multi-layer topologies.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Dataset Ingestion, Partitioning, and Standardization
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 2. Hidden Layer Architectures
architectures = [(5,), (10,), (10, 10), (50, 25)]

print("=== Experiment 13: Multilayer Perceptron Architectures ===")
for arch in architectures:
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=2000, random_state=42)
    mlp.fit(X_train, y_train)
    tr_acc = mlp.score(X_train, y_train) * 100
    te_acc = mlp.score(X_test, y_test) * 100
    print(f"Architecture {str(arch):<12} | Training Acc: {tr_acc:.2f}% | Testing Acc: {te_acc:.2f}%")
