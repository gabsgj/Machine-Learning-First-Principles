"""
Bias-Variance Decomposition from Scratch
========================================
Mathematical Implementation:
1. Exact Analytical Proof: E[(y - f_hat)^2] = Bias^2 + Variance + Noise
2. Monte-Carlo Simulation of Generalization Error:
   - Repeated sampling from true data-generating distribution f(x) = sin(pi * x) + eps
   - Fitting estimators of varying complexities across T independent realization datasets
   - Computing empirical Bias^2 and Variance across the test domain
"""

import numpy as np
from typing import Tuple, Dict, List


def true_signal_function(x: np.ndarray) -> np.ndarray:
    """True underlying data-generating function."""
    return np.sin(np.pi * x)


def generate_synthetic_realization(n_samples: int = 40, noise_std: float = 0.3, rng=None):
    """Generates synthetic dataset from known generative process: y = f(x) + epsilon."""
    if rng is None:
        rng = np.random.RandomState()
    x = rng.uniform(-1.0, 1.0, n_samples)
    noise = rng.normal(0.0, noise_std, n_samples)
    y = true_signal_function(x) + noise
    return x, y


def monte_carlo_bias_variance_decomposition(
    degree: int,
    n_realizations: int = 150,
    n_train_samples: int = 30,
    noise_std: float = 0.3,
) -> Dict[str, float]:
    """
    Empirically decomposes mean squared error into Bias^2, Variance, and Irreducible Noise.
    """
    rng = np.random.RandomState(42)
    # Fixed evaluation grid
    x_test = np.linspace(-1.0, 1.0, 200)
    y_test_clean = true_signal_function(x_test)

    # Store predictions of models fitted on each realization
    predictions = np.zeros((n_realizations, len(x_test)))

    for r in range(n_realizations):
        x_train, y_train = generate_synthetic_realization(n_train_samples, noise_std, rng)
        
        # Fit polynomial of specified degree
        Phi_train = np.vander(x_train, degree + 1, increasing=True)
        theta = np.linalg.pinv(Phi_train) @ y_train
        
        Phi_test = np.vander(x_test, degree + 1, increasing=True)
        predictions[r] = Phi_test @ theta

    # Mean prediction across all training datasets: E[f_hat(x)]
    mean_prediction = np.mean(predictions, axis=0)

    # Bias^2: (E[f_hat(x)] - f(x))^2
    bias_squared = float(np.mean((mean_prediction - y_test_clean) ** 2))

    # Variance: E[(f_hat(x) - E[f_hat(x)])^2]
    variance = float(np.mean(np.var(predictions, axis=0)))

    # Irreducible noise: sigma^2
    irreducible_noise = float(noise_std ** 2)

    total_expected_error = bias_squared + variance + irreducible_noise

    return {
        "Degree": degree,
        "Bias_Squared": bias_squared,
        "Variance": variance,
        "Noise": irreducible_noise,
        "Total_Error": total_expected_error,
    }


if __name__ == "__main__":
    print("=== Demo: Monte-Carlo Bias-Variance Decomposition across Degrees ===")
    print(f"{'Degree':<8} {'Bias^2':<12} {'Variance':<12} {'Irreducible Noise':<18} {'Total MSE':<12}")
    print("-" * 64)

    for deg in [1, 2, 3, 5, 8]:
        stats = monte_carlo_bias_variance_decomposition(degree=deg, n_realizations=100)
        print(f"{stats['Degree']:<8} {stats['Bias_Squared']:<12.4f} {stats['Variance']:<12.4f} {stats['Noise']:<18.4f} {stats['Total_Error']:<12.4f}")
