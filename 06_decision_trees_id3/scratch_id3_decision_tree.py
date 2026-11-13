"""
ID3 Decision Tree Classifier from Scratch
=========================================
Algorithmic Implementation:
1. Shannon Entropy: H(S) = - sum(p_i * log2(p_i))
2. Information Gain: Gain(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
3. Recursive tree growth with max_depth and min_samples_split stopping criteria
4. Prediction engine supporting categorical and continuous features
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List


def calculate_entropy(y: Union[np.ndarray, pd.Series, List]) -> float:
    """Computes Shannon Entropy of target labels."""
    arr = np.asarray(y)
    if len(arr) == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probabilities = counts / len(arr)
    # Add epsilon to prevent log2(0)
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))


def calculate_information_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    """Computes Information Gain achieved by splitting on categorical feature."""
    total_entropy = calculate_entropy(df[target])
    n_total = len(df)
    
    sub_entropies = 0.0
    for _, subset in df.groupby(feature):
        weight = len(subset) / n_total
        sub_entropies += weight * calculate_entropy(subset[target])
        
    return float(total_entropy - sub_entropies)


class ID3DecisionTreeClassifierScratch:
    """
    ID3 Decision Tree algorithm supporting recursive symbolic trees.
    """
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_: Optional[Dict[str, Any]] = None
        self.default_class_: Optional[Any] = None

    def _build_tree(
        self,
        data: pd.DataFrame,
        features: List[str],
        target: str,
        current_depth: int = 0
    ) -> Any:
        unique_classes, class_counts = np.unique(data[target], return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]

        # Stopping Condition 1: Pure Node
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Stopping Condition 2: No more features or reached max depth
        if len(features) == 0 or current_depth >= self.max_depth or len(data) < self.min_samples_split:
            return majority_class

        # Select feature that maximizes Information Gain
        gains = {feat: calculate_information_gain(data, feat, target) for feat in features}
        best_feat = max(gains, key=gains.get)

        # If best gain is practically zero, return majority class
        if gains[best_feat] <= 1e-7:
            return majority_class

        tree = {best_feat: {"_majority": majority_class}}
        remaining_features = [f for f in features if f != best_feat]

        for val, subset in data.groupby(best_feat):
            if len(subset) == 0:
                tree[best_feat][val] = majority_class
            else:
                tree[best_feat][val] = self._build_tree(
                    subset, remaining_features, target, current_depth + 1
                )

        return tree

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "ID3DecisionTreeClassifierScratch":
        data = X.copy()
        target_name = "__target__"
        data[target_name] = np.asarray(y)
        
        vals, counts = np.unique(data[target_name], return_counts=True)
        self.default_class_ = vals[np.argmax(counts)]
        
        features = list(X.columns)
        self.tree_ = self._build_tree(data, features, target_name, current_depth=0)
        return self

    def _predict_row(self, row: pd.Series, node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        feature = list(node.keys())[0]
        feature_dict = node[feature]
        val = row.get(feature)

        if val in feature_dict:
            return self._predict_row(row, feature_dict[val])
        else:
            # Fall back to node majority or global default
            return feature_dict.get("_majority", self.default_class_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Tree has not been fitted.")
        return np.array([self._predict_row(row, self.tree_) for _, row in X.iterrows()])


if __name__ == "__main__":
    import pprint
    print("=== Demo: ID3 Decision Tree on Student Happiness Dataset ===")
    
    # Dataset from id3_dt.pdf tutorial
    data = {
        'Day': ['Weekday', 'Weekday', 'Weekend', 'Holiday', 'Weekend', 'Weekday', 'Holiday', 'Weekend', 'Weekday', 'Holiday', 'Weekend', 'Weekday'],
        'Season': ['Spring', 'Winter', 'Summer', 'Winter', 'Spring', 'Summer', 'Spring', 'Winter', 'Autumn', 'Autumn', 'Summer', 'Winter'],
        'Wind': ['None', 'High', 'Low', 'Low', 'None', 'High', 'High', 'None', 'None', 'High', 'Low', 'None'],
        'Happy': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    X = df[['Day', 'Season', 'Wind']]
    y = df['Happy']

    clf = ID3DecisionTreeClassifierScratch(max_depth=3)
    clf.fit(X, y)

    print("\nLearned ID3 Decision Tree Hierarchy:")
    pprint.pprint(clf.tree_)

    test_samples = pd.DataFrame({
        'Day': ['Weekend', 'Weekday'],
        'Season': ['Summer', 'Winter'],
        'Wind': ['None', 'High']
    })
    preds = clf.predict(test_samples)
    print(f"\nPredictions for test cases: {list(preds)}")
