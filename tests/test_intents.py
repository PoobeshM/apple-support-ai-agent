"""
Unit tests for src/intents.py
"""

import pytest
from src.intents import IntentClassifier, OperationalIntent, IntentClassificationResult


def test_intent_classifier_technical_glitch():
    clf = IntentClassifier()
    res = clf.classify("My iPhone 14 screen keeps freezing after updating to iOS 17.2.")
    assert isinstance(res, IntentClassificationResult)
    assert res.intent == OperationalIntent.TECHNICAL_GLITCH_DEVICE


def test_intent_classifier_billing():
    clf = IntentClassifier()
    res = clf.classify("I was charged $9.99 twice for my Apple Music subscription this month!")
    assert res.intent == OperationalIntent.BILLING_REFUND_SUBSCRIPTION


def test_intent_classifier_account_access():
    clf = IntentClassifier()
    res = clf.classify("My Apple ID was locked for security reasons and 2FA code is not sending.")
    assert res.intent == OperationalIntent.ACCOUNT_ACCESS_AUTH
