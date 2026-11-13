"""
Unit Tests for Multilayer Perceptron
"""

import unittest
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import os, importlib.util

def load_module(rel_path, mod_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_dir, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

scratch_mlp = load_module("10_neural_networks_mlp/scratch_mlp.py", "scratch_mlp")


class TestNeuralNetwork(unittest.TestCase):
    def test_mlp_forward_and_loss_decrease(self):
        digits = load_digits()
        X = digits.data[:200]
        y = digits.target[:200]
        
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        mlp = scratch_mlp.MultilayerPerceptronScratch(
            layer_sizes=[64, 20, 10], activation="relu", learning_rate=0.05, epochs=50, batch_size=32
        )
        mlp.fit(X_std, y)

        # Ensure loss decreased over training
        self.assertLess(mlp.loss_history_[-1], mlp.loss_history_[0])
        
        # Test predictions
        preds = mlp.predict(X_std)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.70)


if __name__ == "__main__":
    unittest.main()
