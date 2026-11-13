"""
Naive Bayes Text Classification with Scikit-Learn
=================================================
Pipeline implementations for NLP text categorization:
- CountVectorizer text tokenization and stop-word filtering
- MultinomialNB vs. BernoulliNB comparison
- Evaluation with Accuracy, Weighted F1, and Precision/Recall reports
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


def run_text_classification_pipeline(categories=None):
    """Fetches text, constructs pipeline, and outputs performance reports."""
    train = fetch_20newsgroups(subset='train', categories=categories)
    test = fetch_20newsgroups(subset='test', categories=categories)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Bernoulli Naive Bayes": BernoulliNB(alpha=1.0)
    }

    results = {}
    for name, model in models.items():
        pipe = make_pipeline(
            CountVectorizer(max_features=1000, stop_words='english'),
            model
        )
        pipe.fit(train.data, train.target)
        preds = pipe.predict(test.data)

        acc = accuracy_score(test.target, preds)
        f1 = f1_score(test.target, preds, average="weighted")
        report = classification_report(test.target, preds, target_names=train.target_names)
        results[name] = {"accuracy": acc, "f1_weighted": f1, "report": report}

    return results


if __name__ == "__main__":
    print("=== Demo: Scikit-Learn Naive Bayes Text Classification Pipeline ===")
    topics = ['sci.med', 'comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
    res = run_text_classification_pipeline(categories=topics)
    for model_name, metrics in res.items():
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        print("\nClassification Report:")
        print(metrics["report"])
