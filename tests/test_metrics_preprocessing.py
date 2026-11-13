"""
Unit Tests for Preprocessing and Evaluation Metrics
"""

import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

from utils.preprocessing import (
    StandardScalerScratch,
    MinMaxScalerScratch,
    SimpleImputerScratch,
    train_test_split_scratch
)
from utils.metrics import (
    accuracy_score_scratch,
    precision_score_scratch,
    recall_score_scratch,
    f1_score_scratch,
    mean_squared_error_scratch,
    r2_score_scratch,
    confusion_matrix_scratch
)


class TestMetricsPreprocessing(unittest.TestCase):
    def test_standard_scaler_parity(self):
        np.random.seed(42)
        X = np.random.randn(50, 4) * 15 + 10
        
        sc_scratch = StandardScalerScratch().fit_transform(X)
        sc_sklearn = StandardScaler().fit_transform(X)
        
        max_diff = np.max(np.abs(sc_scratch - sc_sklearn))
        self.assertLess(max_diff, 1e-12)

    def test_min_max_scaler_parity(self):
        np.random.seed(42)
        X = np.random.randn(50, 4) * 15 + 10
        
        sc_scratch = MinMaxScalerScratch().fit_transform(X)
        sc_sklearn = MinMaxScaler().fit_transform(X)
        
        max_diff = np.max(np.abs(sc_scratch - sc_sklearn))
        self.assertLess(max_diff, 1e-12)

    def test_imputer_mean(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        imputed = SimpleImputerScratch(strategy="mean").fit_transform(X)
        self.assertEqual(imputed[1, 0], 2.0)
        self.assertEqual(imputed[2, 1], 3.0)

    def test_metrics_parity(self):
        np.random.seed(42)
        yt = np.random.randint(0, 2, size=100)
        yp = np.random.randint(0, 2, size=100)

        self.assertAlmostEqual(accuracy_score_scratch(yt, yp), accuracy_score(yt, yp), places=6)
        self.assertAlmostEqual(precision_score_scratch(yt, yp), precision_score(yt, yp), places=6)
        self.assertAlmostEqual(recall_score_scratch(yt, yp), recall_score(yt, yp), places=6)
        self.assertAlmostEqual(f1_score_scratch(yt, yp), f1_score(yt, yp), places=6)

    def test_regression_metrics_parity(self):
        np.random.seed(42)
        yt = np.random.randn(50)
        yp = yt + np.random.normal(0, 0.1, 50)

        self.assertAlmostEqual(mean_squared_error_scratch(yt, yp), mean_squared_error(yt, yp), places=6)
        self.assertAlmostEqual(r2_score_scratch(yt, yp), r2_score(yt, yp), places=6)


if __name__ == "__main__":
    unittest.main()
