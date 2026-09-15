"""
Trivial Baseline Classifier for Customer Support Intent Classification.
Implements Most Frequent Class & simple naive keyword heuristics.
"""

from typing import List
import numpy as np


class TrivialBaselineClassifier:
    def __init__(self, mode: str = "most_frequent"):
        self.mode = mode
        self.most_frequent_class_ = "technical_glitch_device"

    def fit(self, X: List[str], y: List[str]):
        if len(y) > 0:
            vals, counts = np.unique(y, return_counts=True)
            self.most_frequent_class_ = vals[np.argmax(counts)]
        return self

    def predict(self, X: List[str]) -> List[str]:
        if self.mode == "most_frequent":
            return [self.most_frequent_class_] * len(X)
        
        # Simple naive keyword matching fallback
        preds = []
        for text in X:
            t = text.lower()
            if "refund" in t or "charge" in t:
                preds.append("billing_refund_subscription")
            elif "order" in t or "tracking" in t:
                preds.append("order_status_shipping")
            elif "apple id" in t or "password" in t:
                preds.append("account_access_auth")
            else:
                preds.append(self.most_frequent_class_)
        return preds
