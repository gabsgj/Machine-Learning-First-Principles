"""
Unit Tests for Clustering Algorithms
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import os, importlib.util

def load_module(rel_path, mod_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(base_dir, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

scratch_km = load_module("11_unsupervised_clustering/scratch_kmeans.py", "scratch_km")


class TestClustering(unittest.TestCase):
    def test_kmeans_convergence_and_inertia(self):
        X = load_iris().data
        km_scratch = scratch_km.KMeansScratch(n_clusters=3, max_iter=300, random_state=42)
        km_scratch.fit(X)

        self.assertEqual(km_scratch.cluster_centers_.shape, (3, 4))
        self.assertEqual(len(km_scratch.labels_), len(X))
        self.assertGreater(km_scratch.inertia_, 50.0)
        self.assertLess(km_scratch.inertia_, 120.0)

    def test_silhouette_scratch_bounds(self):
        X = load_iris().data
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        sil = scratch_km.compute_silhouette_scratch(X, km.labels_)
        # Silhouette must be strictly in [-1, +1]
        self.assertGreaterEqual(sil, -1.0)
        self.assertLessEqual(sil, 1.0)
        self.assertGreater(sil, 0.40)  # Well-clustered Iris data


if __name__ == "__main__":
    unittest.main()
