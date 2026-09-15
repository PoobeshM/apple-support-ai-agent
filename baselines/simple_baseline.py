"""
Simple Baseline Classifier for Customer Support Intent Classification.
Implements TF-IDF Vectorization with Logistic Regression classifier.
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SimpleBaselineClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')),
            ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=42))
        ])

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: List[str]) -> List[str]:
        return self.pipeline.predict(X).tolist()
