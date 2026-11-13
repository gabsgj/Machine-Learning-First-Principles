"""
Unit Tests for Linear and Polynomial Regression Modules
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils.metrics import mean_squared_error_scratch, r2_score_scratch
import importlib.util
import os

# Helper to load modules dynamically
def load_module(rel_path, mod_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_dir, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

scratch_lin = load_module("02_linear_regression/scratch_linear_regression.py", "scratch_lin")
scratch_poly = load_module("03_polynomial_regression/scratch_polynomial_regression.py", "scratch_poly")


class TestRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_1d = np.linspace(1, 10, 100).reshape(-1, 1)
        self.y_1d = 3.5 * self.X_1d.ravel() + 1.2 + np.random.normal(0, 0.2, 100)

    def test_linear_regression_normal_equation_parity(self):
        # Scratch
        model_scratch = scratch_lin.LinearRegressionScratch(method="normal_equation")
        model_scratch.fit(self.X_1d, self.y_1d)
        
        # Sklearn
        model_sklearn = LinearRegression()
        model_sklearn.fit(self.X_1d, self.y_1d)

        self.assertAlmostEqual(model_scratch.bias_, model_sklearn.intercept_, places=3)
        self.assertAlmostEqual(model_scratch.weights_[0], model_sklearn.coef_[0], places=3)

    def test_linear_regression_gradient_descent_convergence(self):
        # Standardize for stable GD
        mu, sigma = self.X_1d.mean(), self.X_1d.std()
        X_std = (self.X_1d - mu) / sigma

        model_gd = scratch_lin.LinearRegressionScratch(
            method="gradient_descent", learning_rate=0.1, decay_rate=0.01, epochs=1000
        )
        model_gd.fit(X_std, self.y_1d)
        preds = model_gd.predict(X_std)
        r2 = r2_score_scratch(self.y_1d, preds)
        self.assertGreater(r2, 0.95)

    def test_polynomial_regression_parity(self):
        # Generate quadratic curve
        x = np.linspace(-2, 2, 80)
        y = 1.5 * (x**2) - 0.8 * x + 2.0 + np.random.normal(0, 0.1, 80)

        # Scratch degree 2
        p_scratch = scratch_poly.PolynomialRegressionScratch(degree=2)
        p_scratch.fit(x.reshape(-1, 1), y)
        pred_sc = p_scratch.predict(x.reshape(-1, 1))

        # Sklearn degree 2
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        sk_lin = LinearRegression(fit_intercept=False)
        sk_lin.fit(X_poly, y)
        pred_sk = sk_lin.predict(X_poly)

        max_diff = np.max(np.abs(pred_sc - pred_sk))
        self.assertLess(max_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
