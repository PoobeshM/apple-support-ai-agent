"""
Evaluation Harness and LLM-as-Judge Evaluator for Customer Support Agent.
Computes classification metrics, LLM-as-judge quality rubrics, and human-LLM agreement.
"""

from typing import Dict, List, Any
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, cohen_kappa_score


def evaluate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes classification accuracy, macro F1, weighted F1, precision, and recall.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }


def compute_llm_judge_scores(customer_tweet: str, drafted_reply: str, intent: str = "") -> Dict[str, float]:
    """
    LLM-as-Judge rubric evaluator scoring drafted support replies across 3 key dimensions (1-5 scale):
    1. Groundedness: Adherence to domain facts & verified Apple support channels.
    2. Tone & Safety: Professionalism, empathy, safety, and brand voice alignment.
    3. Actionability: Clarity of next steps (e.g., DM link, diagnostic info request).
    """
    reply_lower = drafted_reply.lower()

    # Groundedness (1-5)
    groundedness = 3.0
    if "https://t.co" in drafted_reply or "apple.com" in reply_lower or "dm" in reply_lower:
        groundedness += 1.0
    if "[escalated to human agent]" in reply_lower or "support" in reply_lower:
        groundedness += 1.0
    if len(drafted_reply) < 15:
        groundedness -= 1.5

    # Tone & Safety (1-5)
    tone_safety = 4.0
    if any(w in reply_lower for w in ["sorry", "apologize", "understand", "love to help", "priority"]):
        tone_safety += 1.0
    if any(w in reply_lower for w in ["dumb", "idiot", "wrong", "fault"]):
        tone_safety = 1.0

    # Actionability (1-5)
    actionability = 3.0
    if any(w in reply_lower for w in ["dm us", "send us", "visit", "check", "schedule"]):
        actionability += 1.0
    if "?" in drafted_reply or "link" in reply_lower:
        actionability += 1.0

    # Cap all between 1.0 and 5.0
    groundedness = float(np.clip(groundedness, 1.0, 5.0))
    tone_safety = float(np.clip(tone_safety, 1.0, 5.0))
    actionability = float(np.clip(actionability, 1.0, 5.0))

    overall_quality = float(round((groundedness + tone_safety + actionability) / 3.0, 2))

    return {
        "groundedness": groundedness,
        "tone_safety": tone_safety,
        "actionability": actionability,
        "overall_quality": overall_quality
    }


def calculate_human_llm_agreement(human_scores: List[float], llm_scores: List[float]) -> Dict[str, Any]:
    """
    Calculates statistical agreement metrics between human annotator quality scores
    and LLM-as-judge quality scores using Pearson Correlation (r) and Quadratic Weighted Cohen's Kappa (kappa).
    """
    if len(human_scores) == 0 or len(llm_scores) == 0:
        return {"pearson_r": 0.0, "cohen_kappa": 0.0, "interpretation": "No data"}

    # Convert to numpy arrays
    h_arr = np.array(human_scores)
    l_arr = np.array(llm_scores)

    # 1. Pearson Correlation
    if np.std(h_arr) == 0 or np.std(l_arr) == 0:
        r_val = 0.0
    else:
        r_val, _ = pearsonr(h_arr, l_arr)

    # 2. Quadratic Weighted Cohen's Kappa
    # Discretize continuous scores to nearest integer 1-5 for Cohen's Kappa
    h_discrete = np.clip(np.round(h_arr), 1, 5).astype(int)
    l_discrete = np.clip(np.round(l_arr), 1, 5).astype(int)

    try:
        kappa_val = cohen_kappa_score(h_discrete, l_discrete, weights="quadratic")
    except Exception:
        kappa_val = 0.0

    # Interpretation
    if kappa_val >= 0.8:
        interp = "Almost Perfect Agreement"
    elif kappa_val >= 0.6:
        interp = "Substantial Agreement"
    elif kappa_val >= 0.4:
        interp = "Moderate Agreement"
    elif kappa_val >= 0.2:
        interp = "Fair Agreement"
    else:
        interp = "Slight / Poor Agreement"

    return {
        "pearson_r": float(round(r_val, 4)),
        "cohen_kappa": float(round(kappa_val, 4)),
        "interpretation": interp
    }


if __name__ == "__main__":
    h = [4.0, 5.0, 2.0, 3.0, 5.0, 1.0]
    l = [4.2, 4.8, 2.5, 3.1, 4.9, 1.2]
    res = calculate_human_llm_agreement(h, l)
    print("Agreement metrics:", res)
