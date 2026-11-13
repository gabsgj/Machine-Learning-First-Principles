"""
Experiment 14: Neural Network Activation Dynamics
=================================================
Dataset: Optical Recognition of Handwritten Digits
Compares training duration and convergence across Sigmoid, Tanh, and ReLU.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

# 1. Dataset Ingestion & Normalization
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Activation Function Comparison
activations = ['logistic', 'tanh', 'relu']

print("=== Experiment 14: Activation Functions Dynamics ===")
for act in activations:
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(32,), activation=act, max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    elapsed = time.time() - t0
    acc = mlp.score(X_test, y_test) * 100
    print(f"Activation: {act.capitalize():<10} | Test Accuracy: {acc:.2f}% | Elapsed Time: {elapsed:.2f}s")
