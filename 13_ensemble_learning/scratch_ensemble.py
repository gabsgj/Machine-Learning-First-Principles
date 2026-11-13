"""
Ensemble Learning from Scratch
==============================
Algorithmic Implementations:
1. Bagging Classifier (Bootstrap Aggregating with Majority Voting)
2. AdaBoost Classifier (Adaptive Boosting with instance reweighting)
3. Performance comparison on tabular diagnostic data
"""

import numpy as np
from typing import List, Optional
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifierScratch:
    """
    Bootstrap Aggregating (Bagging) classifier built from first principles.
    Reduces variance by aggregating predictions of independent base estimators.
    """
    def __init__(self, base_estimator=None, n_estimators: int = 50, random_state: Optional[int] = 42):
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=4)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifierScratch":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int).ravel()
        n_samples = len(X_arr)
        
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Draw bootstrap sample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X_arr[indices], y_arr[indices]
            
            # Clone and fit base estimator
            est = DecisionTreeClassifier(max_depth=4, random_state=rng.randint(10000))
            est.fit(X_boot, y_boot)
            self.estimators_.append(est)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        # Collect predictions from all committee estimators: shape (n_estimators, n_samples)
        all_preds = np.array([est.predict(X_arr) for est in self.estimators_])
        # Majority vote across estimators
        voted_preds = []
        for sample_col in all_preds.T:
            voted_preds.append(np.bincount(sample_col).argmax())
        return np.array(voted_preds)


class AdaBoostClassifierScratch:
    """
    Adaptive Boosting (AdaBoost) algorithm implemented from first principles.
    Reduces bias sequentially by re-weighting incorrectly classified instances.
    """
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, random_state: Optional[int] = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List = []
        self.estimator_weights_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifierScratch":
        X_arr = np.asarray(X, dtype=float)
        # Convert binary {0, 1} to {-1, +1}
        y_arr = np.asarray(y, dtype=int).ravel()
        y_sign = np.where(y_arr <= 0, -1.0, 1.0)
        n_samples = len(X_arr)

        # Initialize uniform sample weights
        w = np.full(n_samples, 1.0 / n_samples)
        self.estimators_ = []
        self.estimator_weights_ = []
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            # Fit decision stump (depth-1 tree) on weighted data
            stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(10000))
            stump.fit(X_arr, y_sign, sample_weight=w)
            pred = stump.predict(X_arr)

            # Compute weighted misclassification error
            misclassified = (pred != y_sign)
            error = float(np.sum(w[misclassified])) / float(np.sum(w))
            
            # Avoid division by zero or negative weights
            error = np.clip(error, 1e-10, 1.0 - 1e-10)
            if error >= 0.5:
                # Weak learner no better than random guessing
                break

            # Estimator weight alpha_m = 0.5 * ln((1 - error) / error)
            alpha = float(0.5 * np.log((1.0 - error) / error)) * self.learning_rate

            # Exponential sample weight update: w_i <- w_i * exp(-alpha * y_i * h_m(x_i))
            w *= np.exp(-alpha * y_sign * pred)
            w /= np.sum(w)  # Re-normalize

            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        # Weighted committee sum: sum(alpha_m * h_m(x))
        committee_votes = np.zeros(len(X_arr))
        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            committee_votes += alpha * stump.predict(X_arr)
            
        final_sign = np.sign(committee_votes)
        # Convert {-1, +1} back to {0, 1}
        return np.where(final_sign <= 0, 0, 1)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    print("=== Demo: Ensemble Methods from Scratch (Breast Cancer) ===")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # 1. Scratch Bagging
    bagging = BaggingClassifierScratch(n_estimators=50)
    bagging.fit(X_train, y_train)
    p_bag = bagging.predict(X_test)
    print(f"Scratch Bagging Accuracy:  {np.mean(p_bag == y_test) * 100:.2f}%")

    # 2. Scratch AdaBoost
    adaboost = AdaBoostClassifierScratch(n_estimators=50)
    adaboost.fit(X_train, y_train)
    p_ada = adaboost.predict(X_test)
    print(f"Scratch AdaBoost Accuracy: {np.mean(p_ada == y_test) * 100:.2f}%")
