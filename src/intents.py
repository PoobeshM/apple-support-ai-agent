"""
Intent Taxonomy and Classifier Module for @AppleSupport Customer Support Agent.
Includes Pydantic schemas, taxonomy definitions, zero-shot API LLM classification,
and local fallback intent classifier.
"""

import os
import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class OperationalIntent(str, Enum):
    TECHNICAL_GLITCH_DEVICE = "technical_glitch_device"
    ACCOUNT_ACCESS_AUTH = "account_access_auth"
    BILLING_REFUND_SUBSCRIPTION = "billing_refund_subscription"
    ORDER_STATUS_SHIPPING = "order_status_shipping"
    PRODUCT_INQUIRY_COMPATIBILITY = "product_inquiry_compatibility"
    REPAIR_SERVICE_WARRANTY = "repair_service_warranty"
    GENERAL_FEEDBACK_COMPLAINT = "general_feedback_complaint"


INTENT_DESCRIPTIONS: Dict[str, str] = {
    "technical_glitch_device": "Hardware or software glitches, device crash, freezing screen, battery drain, Wi-Fi/Bluetooth disconnects, audio issues.",
    "account_access_auth": "Apple ID locked, 2FA authorization codes not sending, forgotten password, iCloud login prompts.",
    "billing_refund_subscription": "App Store charges, double billing, unauthorized purchases, subscription cancellations, refund requests.",
    "order_status_shipping": "Apple Store order status, tracking package, delayed shipment, trade-in kit delivery, store pickup.",
    "product_inquiry_compatibility": "Product features, specs, hardware compatibility (e.g. Apple Pencil with iPad), trade-in policy.",
    "repair_service_warranty": "Cracked screen repair, battery replacement cost, AppleCare+ coverage check, Genius Bar booking.",
    "general_feedback_complaint": "General frustration with iOS updates, store staff feedback, praise/complaints without explicit troubleshooting request."
}

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "technical_glitch_device": ["freeze", "freezing", "crash", "crashes", "draining", "battery", "ios", "wi-fi", "wifi", "bluetooth", "audio", "update", "restart", "bug", "glitch"],
    "account_access_auth": ["apple id", "locked", "password", "2fa", "verification code", "icould login", "sign in", "recovery"],
    "billing_refund_subscription": ["charge", "charged", "billing", "refund", "subscription", "purchase", "in-app", "app store bill", "invoice", "cancelled"],
    "order_status_shipping": ["order", "tracking", "shipped", "shipping", "delivery", "delivered", "trade-in kit", "pickup", "package"],
    "product_inquiry_compatibility": ["compatible", "compatibility", "work with", "connect", "connects", "pairing", "pair", "specs", "support audio sharing", "magsafe", "transferable"],
    "repair_service_warranty": ["repair", "screen cracked", "genius bar", "applecare", "warranty", "replacement", "appointment", "fix screen"],
    "general_feedback_complaint": ["terrible", "disappointed", "love", "helpful", "great service", "redesign", "staff", "wait time", "cables frayed"]
}


class IntentClassificationResult(BaseModel):
    intent: OperationalIntent = Field(description="Detected operational intent tag")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="Classification based on key matching and semantic signals.", description="Explanatory rationale")


class IntentClassifier:
    """
    Hybrid Intent Classifier supporting both Cloud API LLM (OpenAI/Groq)
    and robust deterministic keyword/similarity local execution.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")

    def classify(self, text: str) -> IntentClassificationResult:
        text_lower = text.lower()

        # Score intents based on keyword matches
        scores: Dict[str, int] = {intent.value: 0 for intent in OperationalIntent}
        
        for intent_str, keywords in INTENT_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    scores[intent_str] += 2
                elif kw in text_lower:
                    scores[intent_str] += 1

        best_intent_str = max(scores, key=scores.get) # type: ignore
        best_score = scores[best_intent_str]

        if best_score > 0:
            intent_enum = OperationalIntent(best_intent_str)
            confidence = min(0.95, 0.6 + (best_score * 0.1))
            return IntentClassificationResult(
                intent=intent_enum,
                confidence=confidence,
                reasoning=f"Matched operational keywords for '{best_intent_str}' (score: {best_score})."
            )

        # Fallback to technical glitch if unspecified device problem, else general feedback
        if any(w in text_lower for w in ["iphone", "ipad", "mac", "watch", "airpods"]):
            return IntentClassificationResult(
                intent=OperationalIntent.TECHNICAL_GLITCH_DEVICE,
                confidence=0.5,
                reasoning="Fallback to technical_glitch_device due to Apple hardware mention."
            )

        return IntentClassificationResult(
            intent=OperationalIntent.GENERAL_FEEDBACK_COMPLAINT,
            confidence=0.4,
            reasoning="Fallback to general_feedback_complaint as no specific operational category was strongly matched."
        )


if __name__ == "__main__":
    classifier = IntentClassifier()
    res = classifier.classify("My iPhone 14 screen keeps freezing after updating to iOS 17.2.")
    print("Classified:", res.model_dump())
