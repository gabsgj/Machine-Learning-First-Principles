"""
Experiment 15: Neural Network Hyperparameter Tuning via Grid Search
===================================================================
Dataset: Optical Recognition of Handwritten Digits
Sweeps learning rates, batch sizes, and training epochs via 3-Fold Cross-Validation.
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

# 1. Dataset Ingestion & Normalization
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Hyperparameter Grid
param_grid = {
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [32, 128],
    'max_iter': [30, 60]
}

# 3. Exhaustive Grid Search
print("=== Experiment 15: Hyperparameter Tuning ===")
grid = GridSearchCV(MLPClassifier(hidden_layer_sizes=(32,), random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

print("Optimal Hyperparameters:", grid.best_params_)
print(f"Optimal CV Accuracy:    {grid.best_score_ * 100:.2f}%")
print(f"Independent Test Acc:   {grid.score(X_test, y_test) * 100:.2f}%")
