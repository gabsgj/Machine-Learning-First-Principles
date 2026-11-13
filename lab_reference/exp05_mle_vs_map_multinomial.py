"""
Experiment 05: MLE vs. MAP for a Multinomial Distribution
=========================================================
Dataset: 20 Newsgroups (Text Document Counts)
Demonstrates:
1. Maximum Likelihood Estimation via raw token counts
2. Maximum A Posteriori with Dirichlet conjugate prior (alpha = 2, Laplace smoothing)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion
data = fetch_20newsgroups(subset='train')

# 2. Text Tokenization & Bag-of-Words Vectorization
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data.data).toarray()

# 3. Frequency Statistics
counts = X.sum(axis=0)
total_tokens = counts.sum()

# 4. Parameter Estimation
# MLE: theta = counts / total
mle_probs = counts / total_tokens

# MAP with Symmetric Dirichlet Prior (alpha = 2, Laplace add-one smoothing):
alpha = 2.0
map_probs = (counts + alpha - 1.0) / (total_tokens + len(counts) * (alpha - 1.0))

print("=== Experiment 05: Probability Estimates ===")
print("MLE Probabilities Sum: ", round(mle_probs.sum(), 4))
print("MAP Probabilities Sum: ", round(map_probs.sum(), 4))

words = vectorizer.get_feature_names_out()
for i in range(5):
    print(f"Token '{words[i]}': MLE = {mle_probs[i]:.6f} | MAP = {map_probs[i]:.6f}")

# 5. Comparative Probability Plot
plt.figure(figsize=(8, 4.5), dpi=150)
plt.plot(mle_probs[:10], label='MLE (Raw Frequencies)', marker='o', lw=2)
plt.plot(map_probs[:10], label='MAP (Dirichlet Prior alpha=2)', marker='s', lw=2)
plt.xlabel('Vocabulary Words')
plt.ylabel('Estimated Probability')
plt.title('Experiment 05: MLE vs. MAP under Dirichlet Conjugacy')
plt.xticks(range(10), words[:10], rotation=35)
plt.legend()
plt.tight_layout()
plt.savefig('exp05_mle_vs_map.png')
print("Saved visualization to exp05_mle_vs_map.png")
