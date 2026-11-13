"""
Comparison and Benchmarking: Multilayer Perceptron & Activations
================================================================
Side-by-side study:
1. Parity and accuracy comparison between Scratch MLP and Scikit-Learn MLPClassifier.
2. Training loss trajectories comparing Sigmoid, ReLU, and Tanh activation functions.
3. Analysis of vanishing gradient phenomenon in deep sigmoidal networks.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from scratch_mlp import MultilayerPerceptronScratch


def benchmark_mlp_parity_digits():
    print("==================================================")
    print("1. MLP BENCHMARK: Digits Dataset (Parity Test)")
    print("==================================================")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # 1. Scratch MLP (Topology [64, 40, 10])
    t0 = time.time()
    mlp_scratch = MultilayerPerceptronScratch(
        layer_sizes=[64, 40, 10], activation="relu", learning_rate=0.05, epochs=120, batch_size=32
    )
    mlp_scratch.fit(X_train_std, y_train)
    pred_sc = mlp_scratch.predict(X_test_std)
    time_sc = time.time() - t0
    acc_sc = accuracy_score(y_test, pred_sc) * 100

    # 2. Sklearn MLPClassifier (Architecture (40,))
    t1 = time.time()
    mlp_sklearn = MLPClassifier(hidden_layer_sizes=(40,), activation="relu", max_iter=120, random_state=42)
    mlp_sklearn.fit(X_train_std, y_train)
    pred_sk = mlp_sklearn.predict(X_test_std)
    time_sk = time.time() - t1
    acc_sk = accuracy_score(y_test, pred_sk) * 100

    print(f"{'Framework':<22} {'Testing Accuracy':<20} {'Training Time':<15}")
    print("-" * 57)
    print(f"{'Scratch MLP (NumPy)':<22} {acc_sc:<20.2f}% {time_sc:<15.2f}s")
    print(f"{'Scikit-Learn (Cython)':<22} {acc_sk:<20.2f}% {time_sk:<15.2f}s")


def compare_activation_dynamics():
    print("\n==================================================")
    print("2. ACTIVATION CONVERGENCE STUDY: Sigmoid vs ReLU vs Tanh")
    print("==================================================")
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    activations = ["sigmoid", "relu", "tanh"]
    colors = {"sigmoid": "#d62728", "relu": "#2ca02c", "tanh": "#1f77b4"}
    histories = {}

    for act in activations:
        mlp = MultilayerPerceptronScratch(
            layer_sizes=[64, 32, 10], activation=act, learning_rate=0.03, epochs=100, batch_size=32
        )
        mlp.fit(X_std, y)
        histories[act] = mlp.loss_history_
        final_loss = mlp.loss_history_[-1]
        print(f"Activation: {act.capitalize():<10} | Final Loss: {final_loss:.4f}")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 4.8), dpi=150)
    for act in activations:
        plt.plot(histories[act], label=f"{act.capitalize()} Activation", color=colors[act], lw=2)

    plt.title("MLP Training Loss Dynamics Across Activation Functions", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Cross-Entropy Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/10_mlp_activations_convergence.png")
    plt.close()
    print(">> Saved activation convergence plot to 'plots/10_mlp_activations_convergence.png' <<\n")


if __name__ == "__main__":
    benchmark_mlp_parity_digits()
    compare_activation_dynamics()
