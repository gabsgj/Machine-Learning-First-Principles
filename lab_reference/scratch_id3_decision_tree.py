"""
Core Scratch: ID3 Decision Tree Classifier
==========================================
First-principles implementation of the ID3 algorithm:
1. Shannon Entropy: H(S) = - sum(p * log2(p))
2. Information Gain: Gain(S, A) = H(S) - sum(|S_v| / |S| * H(S_v))
3. Recursive top-down tree induction with stopping conditions
"""

import numpy as np
import pandas as pd
import pprint

# 1. Representative Categorical Dataset
data = {
    'Day': ['Weekday', 'Weekday', 'Weekend', 'Holiday', 'Weekend', 'Weekday', 'Holiday', 'Weekend'],
    'Season': ['Spring', 'Winter', 'Summer', 'Winter', 'Spring', 'Summer', 'Spring', 'Winter'],
    'Wind': ['None', 'High', 'Low', 'Low', 'None', 'High', 'High', 'None'],
    'Happy': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)
features = ['Day', 'Season', 'Wind']
target = 'Happy'

# 2. Shannon Entropy Formulation
def calculate_entropy(col: pd.Series) -> float:
    probs = col.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-9))

# 3. Information Gain Formulation
def calculate_information_gain(df_subset: pd.DataFrame, feat: str, target_col: str) -> float:
    total_entropy = calculate_entropy(df_subset[target_col])
    weighted_entropy = sum(
        (len(sub) / len(df_subset)) * calculate_entropy(sub[target_col])
        for _, sub in df_subset.groupby(feat)
    )
    return total_entropy - weighted_entropy

# 4. Recursive ID3 Induction
def build_id3_tree(df_subset: pd.DataFrame, feat_list: list, target_col: str, depth: int = 0, max_depth: int = 2):
    unique_targets = np.unique(df_subset[target_col])
    if len(unique_targets) == 1:
        return unique_targets[0]
    if not feat_list or depth >= max_depth:
        return df_subset[target_col].mode()[0]

    best_feat = max(feat_list, key=lambda f: calculate_information_gain(df_subset, f, target_col))
    remaining = [f for f in feat_list if f != best_feat]

    tree = {
        best_feat: {
            val: build_id3_tree(sub, remaining, target_col, depth + 1, max_depth)
            for val, sub in df_subset.groupby(best_feat)
        }
    }
    return tree

print("=== ID3 Symbolic Hierarchy Output ===")
tree_structure = build_id3_tree(df, features, target)
pprint.pprint(tree_structure)
