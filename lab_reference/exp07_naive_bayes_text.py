"""
Experiment 07: Multinomial vs. Bernoulli Naive Bayes for NLP
============================================================
Dataset: 20 Newsgroups
Compares word count frequencies (Multinomial) against binary word presence (Bernoulli).
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

# 1. Dataset Ingestion (Training and Testing Partitions)
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

X_train, y_train = train.data, train.target
X_test, y_test = test.data, test.target

# 2. Pipeline Definitions
models = {
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB':   BernoulliNB()
}

# 3. Model Training & Evaluation
print("=== Experiment 07: Text Classification with Naive Bayes ===")
for name, model in models.items():
    clf = make_pipeline(CountVectorizer(max_features=1000, stop_words='english'), model)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"{name:<15} | Accuracy: {acc*100:.2f}% | Weighted F1: {f1:.4f}")
