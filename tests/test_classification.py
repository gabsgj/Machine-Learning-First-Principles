"""
Unit Tests for Classification Algorithms
"""

import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import os, importlib.util

def load_module(rel_path, mod_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_dir, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

scratch_log = load_module("04_logistic_regression/scratch_logistic_regression.py", "scratch_log")
scratch_knn = load_module("05_k_nearest_neighbors/scratch_knn.py", "scratch_knn")
scratch_svm = load_module("07_support_vector_machines/scratch_svm.py", "scratch_svm")
scratch_nb = load_module("08_naive_bayes/scratch_naive_bayes.py", "scratch_nb")


class TestClassification(unittest.TestCase):
    def test_logistic_regression_accuracy(self):
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
        )
        clf = scratch_log.LogisticRegressionScratch(learning_rate=0.1, epochs=1500, scale_features=True)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = np.mean(preds == y_test)
        self.assertGreater(acc, 0.90)

    def test_knn_parity(self):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        # Scratch KNN
        knn_sc = scratch_knn.KNNClassifierScratch(k=3, metric="euclidean")
        knn_sc.fit(X_train, y_train)
        pred_sc = knn_sc.predict(X_test)

        # Sklearn KNN
        knn_sk = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
        knn_sk.fit(X_train, y_train)
        pred_sk = knn_sk.predict(X_test)

        disagreements = np.sum(pred_sc != pred_sk)
        self.assertEqual(disagreements, 0)

    def test_linear_svm_separable(self):
        iris = load_iris()
        X = iris.data[:, [2, 3]]
        y = (iris.target == 0).astype(int)  # Linearly separable Setosa
        svm = scratch_svm.LinearSVMScratch(C=10.0, learning_rate=0.01, epochs=1000)
        svm.fit(X, y)
        preds = svm.predict(X)
        acc = np.mean(preds == y)
        self.assertGreaterEqual(acc, 0.98)

    def test_multinomial_nb_parity(self):
        # Generate simple word count matrix
        np.random.seed(42)
        X = np.random.randint(0, 10, size=(100, 20))
        y = np.random.randint(0, 2, size=100)

        mnb_sc = scratch_nb.MultinomialNaiveBayesScratch(alpha=1.0).fit(X, y)
        pred_sc = mnb_sc.predict(X)

        mnb_sk = MultinomialNB(alpha=1.0).fit(X, y)
        pred_sk = mnb_sk.predict(X)

        self.assertEqual(np.sum(pred_sc != pred_sk), 0)


if __name__ == "__main__":
    unittest.main()
