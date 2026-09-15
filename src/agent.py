"""
Grounded Reply Generator and Escalation Decision Engine for @AppleSupport Agent.
Implements RAG Exemplar Retriever, Escalation Engine, and Agent Execution Core.
"""

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intents import OperationalIntent, IntentClassifier


class AgentResponse(BaseModel):
    intent: str = Field(description="Detected operational intent tag")
    drafted_reply: str = Field(description="Grounded support response drafted for customer")
    auto_handle: bool = Field(description="True if agent auto-handles; False if escalated to human agent")
    escalation_reason: Optional[str] = Field(default=None, description="Reason string explaining escalation")
    grounded_exemplar_id: Optional[str] = Field(default=None, description="ID of exemplar response used for RAG grounding")


# Risk signals for Escalation Decision Engine
SECURITY_PATTERNS = [
    r"\bpassword\b", r"\bpin\b", r"\bcredit card\b", r"\bssn\b", r"\bsecurity question\b",
    r"\bhacked\b", r"\bstolen\b", r"\bunauthorized purchase\b"
]

COMPLIANCE_LEGAL_PATTERNS = [
    r"\blawsuit\b", r"\blawyer\b", r"\battorney\b", r"\bsue\b", r"\bftc\b", r"\bbetter business bureau\b", r"\bbbb\b"
]

PROFANITY_RAGE_PATTERNS = [
    r"\bscam\b", r"\bgarbage\b", r"\bworst company\b", r"\bhorrible\b", r"\bthieves\b", r"\bshame on you\b"
]


def evaluate_escalation_rules(customer_tweet: str, intent: str) -> Tuple[bool, str]:
    """
    Evaluates risk signals and business policy rules to decide if query can be auto-handled
    or must be escalated to a human support representative.
    
    Returns:
        (auto_handle: bool, escalation_reason: str)
    """
    text_lower = customer_tweet.lower()

    # 1. Check Security / Credential Risks
    for pattern in SECURITY_PATTERNS:
        if re.search(pattern, text_lower):
            clean_pattern = pattern.strip(r'\b')
            return False, f"Escalated due to security/credential risk trigger matching: '{clean_pattern}'"

    # 2. Check Legal / Compliance / Regulatory Threats
    for pattern in COMPLIANCE_LEGAL_PATTERNS:
        if re.search(pattern, text_lower):
            clean_pattern = pattern.strip(r'\b')
            return False, f"Escalated due to legal/compliance threat trigger matching: '{clean_pattern}'"
    # 3. Check Rage / Brand Reputation Risk
    for pattern in PROFANITY_RAGE_PATTERNS:
        if re.search(pattern, text_lower):
            clean_pattern = pattern.strip(r'\b')
            return False, f"Escalated due to high-rage/reputation risk trigger matching: '{clean_pattern}'"
    # 4. Intent-specific Policy Guardrails
    if intent == OperationalIntent.ACCOUNT_ACCESS_AUTH.value:
        if "locked" in text_lower or "2fa" in text_lower:
            return False, "Escalated: Locked Apple ID / 2FA issues require secondary identity verification by human agent."

    if intent == OperationalIntent.BILLING_REFUND_SUBSCRIPTION.value:
        if "unauthorized" in text_lower or "twice" in text_lower or "refund" in text_lower:
            return False, "Escalated: Refund and billing dispute requests require manual agent approval."

    if intent == OperationalIntent.REPAIR_SERVICE_WARRANTY.value:
        if "cracked" in text_lower or "liquid" in text_lower or "flickering" in text_lower:
            return False, "Escalated: Physical hardware damage check requires human Genius Bar technician review."

    return True, "Auto-handled: Query matches standard resolution workflow and safety criteria."


class SupportAgent:
    """
    Complete AI Support Agent combining Intent Classifier, RAG Grounded Generator,
    and Escalation Decision Engine.
    """
    def __init__(self, historical_df=None):
        self.intent_classifier = IntentClassifier()
        self.historical_df = historical_df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        
        if historical_df is not None and not historical_df.empty:
            self.corpus = historical_df['customer_tweet'].tolist()
            self.vectorizer.fit(self.corpus)
            self.tfidf_matrix = self.vectorizer.transform(self.corpus)
        else:
            self.corpus = []
            self.tfidf_matrix = None

    def retrieve_grounded_response(self, customer_tweet: str, intent: str) -> Tuple[str, str]:
        """
        Retrieves the most semantically relevant historical brand response grounded in past resolutions.
        """
        if self.historical_df is not None and self.tfidf_matrix is not None:
            # Filter historical df by intent if possible
            intent_mask = self.historical_df['intent'] == intent
            filtered_indices = self.historical_df[intent_mask].index.tolist()
            
            if filtered_indices:
                sub_matrix = self.tfidf_matrix[filtered_indices]
                query_vec = self.vectorizer.transform([customer_tweet])
                sims = cosine_similarity(query_vec, sub_matrix)[0]
                best_sub_idx = int(sims.argmax())
                best_orig_idx = filtered_indices[best_sub_idx]
                
                row = self.historical_df.iloc[best_orig_idx]
                return str(row['brand_response']), str(row['thread_id'])

        # Fallback grounded templates by intent if RAG corpus empty
        fallback_replies = {
            "technical_glitch_device": "We'd love to help with your device issue. Please send us a DM with your model and current iOS version: https://t.co/apple-dm",
            "account_access_auth": "Security is top priority. Please DM us so we can guide you safely through account recovery steps: https://t.co/apple-dm",
            "billing_refund_subscription": "We can help check your billing details. Please DM us your invoice or order number so we can inspect your account: https://t.co/apple-dm",
            "order_status_shipping": "We know you're eager for your order! Send us a DM with your order number and billing zip code so we can track it: https://t.co/apple-dm",
            "product_inquiry_compatibility": "Great question! Please send us a DM if you'd like full compatibility details and step-by-step setup guides: https://t.co/apple-dm",
            "repair_service_warranty": "We can help schedule a repair or service visit for you. DM us your current location to find store availability: https://t.co/apple-dm",
            "general_feedback_complaint": "We appreciate your feedback regarding our products and services. We'll pass your thoughts along to our team."
        }
        
        reply = fallback_replies.get(intent, "We'd love to help. Please DM us so we can take a closer look: https://t.co/apple-dm")
        return reply, "FALLBACK-GROUNDED"

    def process_query(self, customer_tweet: str) -> AgentResponse:
        # 1. Intent Classification
        clf_result = self.intent_classifier.classify(customer_tweet)
        intent_str = clf_result.intent.value

        # 2. Grounded Response Generation
        drafted_reply, exemplar_id = self.retrieve_grounded_response(customer_tweet, intent_str)

        # 3. Escalation Decision Engine
        auto_handle, escalation_reason = evaluate_escalation_rules(customer_tweet, intent_str)

        # 4. If escalated, prepend explicit handoff statement to drafted response
        if not auto_handle:
            drafted_reply = f"[ESCALATED TO HUMAN AGENT] {drafted_reply}"

        return AgentResponse(
            intent=intent_str,
            drafted_reply=drafted_reply,
            auto_handle=auto_handle,
            escalation_reason=escalation_reason,
            grounded_exemplar_id=exemplar_id
        )


if __name__ == "__main__":
    agent = SupportAgent()
    resp = agent.process_query("@AppleSupport I was charged $9.99 twice for my subscription this month!")
    print(resp.model_dump())
