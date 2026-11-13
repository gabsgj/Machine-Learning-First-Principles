"""
Data Preprocessing Reference Toolkit
====================================
Comprehensive routine covering:
- Stage A: Missing value injection for robustness testing
- Stage B: Univariate imputation (Mean / Median)
- Stage C: Feature-target isolation
- Stage D: Stratified & uniform train-test splitting
- Stage E: Data-leakage-free feature scaling (StandardScaler & MinMaxScaler)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Dataset Generation
np.random.seed(42)
df = pd.DataFrame(
    np.random.randint(10, 100, size=(12, 4)),
    columns=['feature_1', 'feature_2', 'feature_3', 'target']
)

# 2. Stage A: Missing Value Injection
for _ in range(5):
    r = np.random.randint(len(df))
    c = np.random.randint(df.shape[1] - 1)
    df.iat[r, c] = np.nan

print("Missing values count per column:\n", df.isna().sum())

# 3. Stage B: Statistical Imputation
df_imputed = df.fillna(df.mean())

# 4. Stage C: Feature Matrix and Target Separation
X = df_imputed.drop(columns='target').values
y = df_imputed['target'].values

# 5. Stage D: Train-Test Partitioning (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Stage E: Standardization (Learned strictly from training statistics)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nProcessed Training Matrix Shape:", X_train_scaled.shape)
print("Processed Testing Matrix Shape: ", X_test_scaled.shape)
print("Training Column Means:          ", np.round(X_train_scaled.mean(axis=0), 4))
print("Training Column Stds:           ", np.round(X_train_scaled.std(axis=0), 4))
