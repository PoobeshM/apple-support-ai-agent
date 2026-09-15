"""
Unit tests for src/agent.py
"""

import pytest
from src.agent import SupportAgent, evaluate_escalation_rules, AgentResponse


def test_evaluate_escalation_rules_security():
    auto_handle, reason = evaluate_escalation_rules("I forgot my password and my credit card was charged.", "account_access_auth")
    assert auto_handle is False
    assert "security" in reason.lower() or "password" in reason.lower()


def test_evaluate_escalation_rules_safe():
    auto_handle, reason = evaluate_escalation_rules("Does MagSafe battery pack work with iPhone 13 mini?", "product_inquiry_compatibility")
    assert auto_handle is True


def test_support_agent_process_query():
    agent = SupportAgent()
    resp = agent.process_query("How do I connect AirPods Pro to Apple TV 4K?")
    assert isinstance(resp, AgentResponse)
    assert resp.intent == "product_inquiry_compatibility"
    assert resp.auto_handle is True
    assert "https://t.co" in resp.drafted_reply or "DM" in resp.drafted_reply
