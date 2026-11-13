"""
Multilayer Perceptron with Scikit-Learn
======================================
Production implementation covering official syllabus experiments:
1. Exp 13: Architecture variations - single vs. multi-layer feed-forward networks
2. Exp 14: Activation functions - Logistic (Sigmoid), ReLU, and Tanh speed and accuracy
3. Exp 15: Hyperparameter Tuning - Grid search over learning rates, batch sizes, and epochs
"""

import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def evaluate_architectures(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Experiment 13: Compare single-hidden-layer vs deep architectures."""
    architectures = [
        ("Single Layer (50,)", (50,)),
        ("Deep Network (100, 50)", (100, 50)),
        ("Wide Network (128,)", (128,))
    ]
    results = {}
    for label, arch in architectures:
        mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        acc = mlp.score(X_test, y_test)
        results[label] = acc
    return results


def evaluate_activations(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """Experiment 14: Compare Sigmoid (logistic), ReLU, and Tanh activation dynamics."""
    activations = ["logistic", "relu", "tanh"]
    results = {}
    for act in activations:
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(100,), activation=act, max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        elapsed = time.time() - t0
        acc = mlp.score(X_test, y_test)
        results[act] = {"accuracy": acc, "training_time_sec": elapsed}
    return results


def tune_mlp_hyperparameters(X_train: np.ndarray, y_train: np.ndarray):
    """Experiment 15: Grid search over learning rates, batch sizes, and max iterations."""
    mlp = MLPClassifier(hidden_layer_sizes=(50,), random_state=42)
    param_grid = {
        "learning_rate_init": [0.001, 0.01],
        "batch_size": [64, 128],
        "max_iter": [100, 200]
    }
    grid = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


if __name__ == "__main__":
    from sklearn.datasets import load_wine, load_digits

    print("=== Demo: MLP Experiments 13, 14, 15 with Scikit-Learn ===")
    
    # Exp 13: Wine Quality
    wine = load_wine()
    X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42)
    scaler_w = StandardScaler()
    X_tr_w = scaler_w.fit_transform(X_tr_w)
    X_te_w = scaler_w.transform(X_te_w)

    print("\n[EXP 13: Architecture Evaluation on Wine Quality]")
    for arch_name, acc in evaluate_architectures(X_tr_w, y_tr_w, X_te_w, y_te_w).items():
        print(f"Architecture: {arch_name:<25} | Accuracy = {acc*100:.2f}%")

    # Exp 14: Activation functions on Digits
    digits = load_digits()
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)
    scaler_d = StandardScaler()
    X_tr_d = scaler_d.fit_transform(X_tr_d)
    X_te_d = scaler_d.transform(X_te_d)

    print("\n[EXP 14: Activation Functions on Digits]")
    for act_name, stats in evaluate_activations(X_tr_d, y_tr_d, X_te_d, y_te_d).items():
        print(f"Activation: {act_name.capitalize():<10} | Acc: {stats['accuracy']*100:.2f}% | Time: {stats['training_time_sec']:.2f}s")
