"""
Data Preprocessing Toolkit from Scratch
========================================
Vectorized implementations of core data manipulation and transformation algorithms:
1. Missing value injection & imputation (Mean, Median, Mode, KNN)
2. Categorical encoding (Label Encoding, One-Hot Encoding)
3. Feature scaling (StandardScaler, MinMaxScaler)
4. Dataset partitioning (Random and Stratified Train-Test Split)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List


def inject_nan_values(
    data: Union[pd.DataFrame, np.ndarray],
    n_missing: int = 20,
    random_state: Optional[int] = 42,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Injects random NaN values into a dataset for preprocessing testing.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The input matrix or dataframe.
    n_missing : int
        The number of random entries to replace with NaN.
    random_state : int, optional
        Seed for reproducible randomness.
        
    Returns
    -------
    data_corrupted : pd.DataFrame or np.ndarray
        A copy of the data with NaN values inserted.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        n_rows, n_cols = df_copy.shape
        for _ in range(n_missing):
            r = np.random.randint(0, n_rows)
            c = np.random.randint(0, n_cols)
            df_copy.iat[r, c] = np.nan
        return df_copy
    else:
        arr_copy = np.array(data, dtype=float, copy=True)
        n_rows, n_cols = arr_copy.shape
        for _ in range(n_missing):
            r = np.random.randint(0, n_rows)
            c = np.random.randint(0, n_cols)
            arr_copy[r, c] = np.nan
        return arr_copy


class SimpleImputerScratch:
    """
    Univariate feature imputer for completing missing values with simple statistics.
    
    Supported strategies:
    - 'mean': Replace missing values using the mean along each column.
    - 'median': Replace missing values using the median along each column.
    - 'most_frequent': Replace missing using the most frequent value along each column.
    - 'constant': Replace missing values with fill_value.
    """
    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None):
        if strategy not in ["mean", "median", "most_frequent", "constant"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "SimpleImputerScratch":
        arr = np.asarray(X, dtype=float)
        n_features = arr.shape[1]
        self.statistics_ = np.zeros(n_features)

        for col_idx in range(n_features):
            col = arr[:, col_idx]
            valid_vals = col[~np.isnan(col)]
            if len(valid_vals) == 0:
                self.statistics_[col_idx] = 0.0
            elif self.strategy == "mean":
                self.statistics_[col_idx] = np.mean(valid_vals)
            elif self.strategy == "median":
                self.statistics_[col_idx] = np.median(valid_vals)
            elif self.strategy == "most_frequent":
                vals, counts = np.unique(valid_vals, return_counts=True)
                self.statistics_[col_idx] = vals[np.argmax(counts)]
            elif self.strategy == "constant":
                self.statistics_[col_idx] = 0.0 if self.fill_value is None else self.fill_value
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        arr = np.asarray(X, dtype=float).copy()
        if self.statistics_ is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        for col_idx in range(arr.shape[1]):
            nan_mask = np.isnan(arr[:, col_idx])
            arr[nan_mask, col_idx] = self.statistics_[col_idx]
        return arr

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScalerScratch:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Formula:
        z = (x - mu) / sigma
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StandardScalerScratch":
        arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0)
        # Avoid division by zero for constant features
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (arr - self.mean_) / self.std_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (X * self.std_) + self.mean_


class MinMaxScalerScratch:
    """
    Transform features by scaling each feature to a given range [0, 1].
    
    Formula:
        x_scaled = (x - x_min) / (x_max - x_min)
    """
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "MinMaxScalerScratch":
        arr = np.asarray(X, dtype=float)
        self.data_min_ = np.min(arr, axis=0)
        self.data_max_ = np.max(arr, axis=0)
        range_span = self.data_max_ - self.data_min_
        range_span[range_span == 0.0] = 1.0
        
        target_span = self.feature_range[1] - self.feature_range[0]
        self.scale_ = target_span / range_span
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return arr * self.scale_ + self.min_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class LabelEncoderScratch:
    """
    Encode target labels with values between 0 and n_classes-1.
    """
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self._mapping: dict = {}
        self._inverse_mapping: dict = {}

    def fit(self, y: Union[np.ndarray, pd.Series, List]) -> "LabelEncoderScratch":
        values = np.asarray(y)
        self.classes_ = np.unique(values)
        self._mapping = {val: idx for idx, val in enumerate(self.classes_)}
        self._inverse_mapping = {idx: val for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, y: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
        values = np.asarray(y)
        return np.array([self._mapping[v] for v in values], dtype=int)

    def fit_transform(self, y: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Union[np.ndarray, List]) -> np.ndarray:
        indices = np.asarray(y, dtype=int)
        return np.array([self._inverse_mapping[idx] for idx in indices])


class OneHotEncoderScratch:
    """
    Encode categorical features as a one-hot numeric array.
    """
    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.categories_: List[np.ndarray] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "OneHotEncoderScratch":
        df = pd.DataFrame(X)
        self.categories_ = [df[col].unique() for col in df.columns]
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        df = pd.DataFrame(X)
        encoded_blocks = []
        for col_idx, col_name in enumerate(df.columns):
            cats = self.categories_[col_idx]
            if self.drop_first:
                cats_to_encode = cats[1:]
            else:
                cats_to_encode = cats
            
            col_series = df[col_name]
            block = np.zeros((len(df), len(cats_to_encode)), dtype=float)
            for c_idx, c_val in enumerate(cats_to_encode):
                block[col_series == c_val, c_idx] = 1.0
            encoded_blocks.append(block)
        return np.hstack(encoded_blocks)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split_scratch(
    *arrays,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Split arrays or matrices into random train and test subsets from scratch.
    
    Supports:
    - Standard uniform random shuffle
    - Stratified splitting preserving label distributions
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    
    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All input arrays must have the same length.")

    if random_state is not None:
        np.random.seed(random_state)

    if stratify is not None:
        stratify_arr = np.asarray(stratify)
        if len(stratify_arr) != n_samples:
            raise ValueError("Stratify array length must match input samples.")
        
        train_indices = []
        test_indices = []
        unique_classes = np.unique(stratify_arr)
        
        for cls in unique_classes:
            cls_indices = np.where(stratify_arr == cls)[0]
            if shuffle:
                cls_indices = np.random.permutation(cls_indices)
            n_test = int(np.round(len(cls_indices) * test_size))
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
            
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        if shuffle:
            train_indices = np.random.permutation(train_indices)
            test_indices = np.random.permutation(test_indices)
    else:
        indices = np.arange(n_samples)
        if shuffle:
            indices = np.random.permutation(indices)
        n_test = int(np.round(n_samples * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    results = []
    for arr in arrays:
        np_arr = np.asarray(arr)
        results.append(np_arr[train_indices])
        results.append(np_arr[test_indices])
        
    return results
