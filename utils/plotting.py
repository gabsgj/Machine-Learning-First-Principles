"""
Plotting and Visualization Utilities
====================================
Publication-quality plotting functions for model evaluations, decision boundaries,
loss convergence, clustering structures, and bias-variance tradeoff curves.
"""

import os
import matplotlib
# Use non-interactive backend for robust batch/CLI execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple


def plot_loss_curve(
    losses: List[float],
    title: str = "Training Loss vs. Iterations",
    xlabel: str = "Epochs / Iterations",
    ylabel: str = "Loss (MSE / Cross-Entropy)",
    save_path: Optional[str] = None,
):
    """Plots and saves an optimization loss convergence curve."""
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(losses, color="#1f77b4", linewidth=2.0, label="Loss")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_2d_decision_boundary(
    predict_func,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "2D Decision Boundary",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
    support_vectors: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Plots a 2D decision boundary using meshgrid and overlays data points.
    """
    plt.figure(figsize=(7.5, 5), dpi=150)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = predict_func(grid)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40, alpha=0.8)
    
    if support_vectors is not None and len(support_vectors) > 0:
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=120,
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=1.8,
            label="Support Vectors",
        )
        plt.legend(loc="upper right")
        
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_model_comparison(
    model_names: List[str],
    metrics_dict: dict,
    title: str = "Model Comparison Across Evaluation Metrics",
    save_path: Optional[str] = None,
):
    """
    Plots a multi-bar chart comparing several models across metrics.
    
    Parameters
    ----------
    metrics_dict : dict
        Dict where keys are metric names (e.g. 'Accuracy', 'F1') and values are lists of scores.
    """
    plt.figure(figsize=(8, 4.8), dpi=150)
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics_dict)
    
    for i, (metric_name, scores) in enumerate(metrics_dict.items()):
        plt.bar(x + i * width, scores, width, label=metric_name)

    plt.xlabel("Models", fontsize=10, fontweight="bold")
    plt.ylabel("Score", fontsize=10, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xticks(x + width * (len(metrics_dict) - 1) / 2, model_names, rotation=15)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
