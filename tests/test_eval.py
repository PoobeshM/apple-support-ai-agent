"""
Unit tests for src/eval.py
"""

import pytest
from src.eval import evaluate_metrics, compute_llm_judge_scores, calculate_human_llm_agreement


def test_evaluate_metrics():
    y_true = ["intent_a", "intent_b", "intent_a"]
    y_pred = ["intent_a", "intent_b", "intent_b"]
    metrics = evaluate_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == pytest.approx(2/3)


def test_compute_llm_judge_scores():
    tweet = "My screen is frozen."
    reply = "We can help! DM us your iOS version: https://t.co/apple-dm"
    scores = compute_llm_judge_scores(tweet, reply)
    assert 1.0 <= scores["groundedness"] <= 5.0
    assert 1.0 <= scores["tone_safety"] <= 5.0
    assert 1.0 <= scores["actionability"] <= 5.0
    assert 1.0 <= scores["overall_quality"] <= 5.0


def test_calculate_human_llm_agreement():
    h = [5.0, 4.0, 3.0, 2.0, 1.0]
    l = [5.0, 4.0, 3.0, 2.0, 1.0]
    agreement = calculate_human_llm_agreement(h, l)
    assert agreement["pearson_r"] == pytest.approx(1.0)
    assert agreement["cohen_kappa"] == pytest.approx(1.0)
