"""
Data Preprocessing with Scikit-Learn
====================================
Production-grade data preprocessing pipelines using Scikit-Learn:
- SimpleImputer and KNNImputer
- OneHotEncoder and OrdinalEncoder
- StandardScaler and MinMaxScaler
- ColumnTransformer pipeline integration
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def run_sklearn_preprocessing_pipeline(df: pd.DataFrame, target_col: str):
    """
    Constructs and fits a production ColumnTransformer pipeline handling
    both numerical (missing values + scaling) and categorical columns (encoding).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numerical pipeline: impute with median, then standardize
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit on training data only to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    print("=== Demo: Scikit-Learn Data Preprocessing Pipeline ===")
    
    # Create sample synthetic dataframe
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 35, 40, 22, np.nan, 58],
        "salary": [50000, 60000, 75000, np.nan, 110000, 48000, 52000, 130000],
        "department": ["IT", "HR", "IT", "Sales", np.nan, "HR", "Sales", "IT"],
        "purchased": [0, 1, 0, 1, 1, 0, 0, 1]
    })
    
    print("\nInitial DataFrame with missing values:")
    print(df)

    X_train_p, X_test_p, y_tr, y_te, pipe = run_sklearn_preprocessing_pipeline(df, "purchased")
    print(f"\nProcessed Train Shape: {X_train_p.shape}")
    print(f"Processed Test Shape:  {X_test_p.shape}")
    print("\nProcessed Train Matrix sample:")
    print(np.round(X_train_p, 3))
