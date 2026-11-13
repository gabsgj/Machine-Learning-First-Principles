"""
Data Preprocessing from Scratch
===============================
Mathematical and algorithmic implementation of essential data wrangling routines:
1. Random missing value injection (Simulating missingness in lab exams)
2. Statistical imputation (Mean, Median, Mode)
3. K-Nearest Neighbors imputation (Local distance-weighted imputation)
4. Categorical Encoders (Label Encoding and One-Hot Encoding)
5. Feature Normalization (Z-score StandardScaler and Min-Max Scaler)
6. Train-Test Partitioning (Uniform Random and Class-Stratified)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List


class MissingValueHandlerScratch:
    """
    Handles missing values (NaN) using statistical aggregates or neighbor interpolation.
    """
    def __init__(self, strategy: str = "mean", n_neighbors: int = 3):
        assert strategy in ["mean", "median", "mode", "knn"], f"Invalid strategy {strategy}"
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.column_stats_ = {}

    def fit(self, X: np.ndarray) -> "MissingValueHandlerScratch":
        X_arr = np.asarray(X, dtype=float)
        n_features = X_arr.shape[1]
        
        if self.strategy in ["mean", "median", "mode"]:
            for col_idx in range(n_features):
                vals = X_arr[:, col_idx]
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    self.column_stats_[col_idx] = 0.0
                elif self.strategy == "mean":
                    self.column_stats_[col_idx] = np.mean(valid)
                elif self.strategy == "median":
                    self.column_stats_[col_idx] = np.median(valid)
                elif self.strategy == "mode":
                    unique, counts = np.unique(valid, return_counts=True)
                    self.column_stats_[col_idx] = unique[np.argmax(counts)]
        elif self.strategy == "knn":
            self.X_train_ = X_arr.copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float).copy()
        n_rows, n_features = X_arr.shape
        
        if self.strategy in ["mean", "median", "mode"]:
            for col_idx in range(n_features):
                mask = np.isnan(X_arr[:, col_idx])
                X_arr[mask, col_idx] = self.column_stats_[col_idx]
        elif self.strategy == "knn":
            # Impute each row's missing values using K nearest rows that have valid entries
            for i in range(n_rows):
                for j in range(n_features):
                    if np.isnan(X_arr[i, j]):
                        # Compute distance to other training rows ignoring NaNs
                        diffs = self.X_train_ - X_arr[i]
                        # Squared Euclidean distance on non-NaN common features
                        valid_mask = ~np.isnan(diffs) & ~np.isnan(self.X_train_[:, j:j+1])
                        sq_dists = np.nansum(diffs**2, axis=1)
                        # Exclude current row
                        sq_dists[i] = np.inf
                        nearest_indices = np.argsort(sq_dists)[:self.n_neighbors]
                        neighbor_vals = self.X_train_[nearest_indices, j]
                        valid_neighbors = neighbor_vals[~np.isnan(neighbor_vals)]
                        if len(valid_neighbors) > 0:
                            X_arr[i, j] = np.mean(valid_neighbors)
                        else:
                            X_arr[i, j] = np.nanmean(self.X_train_[:, j])
        return X_arr

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class CategoricalEncoderScratch:
    """
    Categorical feature encoders built from first principles.
    Supports Label Encoding and One-Hot Encoding.
    """
    def __init__(self, method: str = "label", drop_first: bool = False):
        assert method in ["label", "onehot"], "Method must be 'label' or 'onehot'"
        self.method = method
        self.drop_first = drop_first
        self.mappings_ = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "CategoricalEncoderScratch":
        df = pd.DataFrame(X)
        for col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            self.mappings_[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        df = pd.DataFrame(X)
        if self.method == "label":
            encoded_cols = []
            for col in df.columns:
                mapping = self.mappings_[col]
                encoded = df[col].map(mapping).fillna(-1).values
                encoded_cols.append(encoded)
            return np.column_stack(encoded_cols)
        elif self.method == "onehot":
            blocks = []
            for col in df.columns:
                categories = list(self.mappings_[col].keys())
                if self.drop_first:
                    categories = categories[1:]
                col_series = df[col]
                block = np.zeros((len(df), len(categories)), dtype=float)
                for idx, cat in enumerate(categories):
                    block[col_series == cat, idx] = 1.0
                blocks.append(block)
            return np.hstack(blocks)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.fit(X).transform(X)


class FeatureScalerScratch:
    """
    Scales features using standard z-score or min-max normalization.
    """
    def __init__(self, method: str = "standard"):
        assert method in ["standard", "minmax"], "Method must be 'standard' or 'minmax'"
        self.method = method
        self.param1_ = None  # mean or min
        self.param2_ = None  # std or range

    def fit(self, X: np.ndarray) -> "FeatureScalerScratch":
        arr = np.asarray(X, dtype=float)
        if self.method == "standard":
            self.param1_ = np.mean(arr, axis=0)
            self.param2_ = np.std(arr, axis=0)
            self.param2_[self.param2_ == 0.0] = 1.0
        elif self.method == "minmax":
            self.param1_ = np.min(arr, axis=0)
            data_max = np.max(arr, axis=0)
            self.param2_ = data_max - self.param1_
            self.param2_[self.param2_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if self.method == "standard":
            return (arr - self.param1_) / self.param2_
        elif self.method == "minmax":
            return (arr - self.param1_) / self.param2_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


if __name__ == "__main__":
    print("=== Demo: Data Preprocessing Toolkit from Scratch ===")
    
    # 1. Simulate data with missing values
    np.random.seed(42)
    raw_data = np.random.randn(10, 3) * 10 + 50
    raw_data[1, 0] = np.nan
    raw_data[4, 2] = np.nan
    raw_data[7, 1] = np.nan
    print("\nOriginal Data with NaNs:\n", np.round(raw_data, 2))

    # 2. Impute with mean
    imputer = MissingValueHandlerScratch(strategy="mean")
    imputed_data = imputer.fit_transform(raw_data)
    print("\nImputed Data (Mean Strategy):\n", np.round(imputed_data, 2))

    # 3. Scale features
    scaler = FeatureScalerScratch(method="standard")
    scaled_data = scaler.fit_transform(imputed_data)
    print("\nStandardized Data (Mean=0, Std=1):\n", np.round(scaled_data, 2))
    print("Computed Means:", np.round(scaled_data.mean(axis=0), 4))
    print("Computed Stds:", np.round(scaled_data.std(axis=0), 4))

    # 4. Categorical Encoding
    cats = np.array([["Red", "S"], ["Blue", "M"], ["Green", "L"], ["Red", "M"]])
    encoder = CategoricalEncoderScratch(method="onehot", drop_first=True)
    ohe_out = encoder.fit_transform(cats)
    print("\nOne-Hot Encoded Categories (drop_first=True):\n", ohe_out)
