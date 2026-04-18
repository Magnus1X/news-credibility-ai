"""
Agent Module — multi-step reasoning pipeline using a free-tier LLM.
Uses HuggingFace Inference API (free tier) with Mistral-7B-Instruct.
Falls back to a deterministic rule-based report if the API is unavailable.
State is maintained explicitly across all 5 reasoning steps.
"""
import os
import json
import re
import requests
from datetime import datetime

HF_API_URL = "https://api-inference.huggingface.co/models/NousResearch/Hermes-3-Llama-3.1-8B"
HF_TOKEN = os.getenv("HF_TOKEN", "")

SYSTEM_PROMPT = """You are a professional fact-checking analyst. Your job is to produce a structured credibility report.

STRICT RULES:
1. Use PLAIN ENGLISH. Avoid academic jargon. Make insights easy for a high-schooler to understand.
2. NEVER fabricate facts, quotes, or sources not provided to you.
3. If evidence is insufficient, write "Insufficient evidence to determine."
4. Clearly separate FACTS (from the data provided) from ASSESSMENTS (your analysis).
5. Ensure the "Summary" provides a meaningful "bottom line" for the user.
6. Do not hallucinate URLs, author names, or publication dates.
"""

REPORT_PROMPT_TEMPLATE = """
{system_prompt}

You have been given the following analysis data for a news article:

--- ARTICLE EXCERPT (first 500 chars) ---
{article_excerpt}

--- ML MODEL OUTPUT ---
Prediction: {label} (Confidence: {confidence}%, Tier: {confidence_tier})
Real probability: {real_prob}% | Fake probability: {fake_prob}%
Top TF-IDF features: {top_features}

--- RISK ANALYSIS ---
Risk Score: {risk_score}/100
Risk Factors: {risk_factors}
Credibility Indicators: {credibility_indicators}

--- RETRIEVED CONTEXT ---
{retrieved_context}

--- TASK ---
Generate a credibility report as valid JSON with EXACTLY these keys:
{{
  "summary": "2-3 sentence plain-English verdict. State facts only.",
  "credibility_indicators": ["list of positive credibility signals found"],
  "risk_factors": ["list of risk/misinformation signals found"],
  "cross_source_verification": "What the retrieved context says about this type of content. If nothing relevant, write 'Insufficient evidence to cross-verify.'",
  "confidence_assessment": "Explain what the {confidence}% confidence score means in plain English. Note any uncertainty.",
  "sources": ["list retrieved source names used in this analysis"],
  "disclaimer": "This analysis is AI-generated and should not be the sole basis for determining truth. Always verify with primary sources."
}}

Respond with ONLY the JSON object. No markdown, no explanation outside the JSON.
"""


def _call_hf_api(prompt: str):
    """Call HuggingFace Inference API. Returns text or None on failure."""
    if not HF_TOKEN:
        return None
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 800,
                "temperature": 0.1,
                "return_full_text": False,
                "stop": ["```", "---"],
            },
        }
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "")
    except Exception as e:
        print(f"HF API error: {e}")
    return None


def _extract_json(text: str):
    """Extract JSON object from LLM output, handling markdown fences."""
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _rule_based_report(state: dict) -> dict:
    """
    Deterministic fallback report when LLM is unavailable.
    Uses only the data already computed — zero hallucination.
    """
    pred = state["prediction"]
    risk = state["risk_analysis"]
    retrieved = state["retrieved_docs"]

    label = pred["label"]
    conf = pred["confidence"]
    conf_tier = pred["confidence_tier"]

    if label == "Fake News":
        cred_note = (
            "Credibility markers were absent."
            if risk["credibility_hits"] == 0
            else "Some credibility markers were present."
        )
        summary = (
            f"The ML model classified this article as Fake News with {conf}% confidence "
            f"({conf_tier} certainty). "
            f"Risk analysis identified {len(risk['risk_factors'])} risk factor(s). "
            f"{cred_note}"
        )
    else:
        risk_note = (
            "No significant risk factors were detected."
            if not risk["risk_factors"]
            else f"{len(risk['risk_factors'])} risk factor(s) were noted."
        )
        summary = (
            f"The ML model classified this article as Real News with {conf}% confidence "
            f"({conf_tier} certainty). "
            f"Risk analysis identified {len(risk['credibility_indicators'])} credibility indicator(s). "
            f"{risk_note}"
        )

    if conf_tier == "high":
        conf_assessment = (
            f"The model is highly confident ({conf}%) in this prediction. "
            "High-confidence predictions are generally reliable."
        )
    elif conf_tier == "medium":
        conf_assessment = (
            f"The model has moderate confidence ({conf}%). "
            "This prediction should be treated with some caution and cross-verified."
        )
    else:
        conf_assessment = (
            f"The model has low confidence ({conf}%). "
            "Insufficient evidence — this result is uncertain and should not be relied upon alone."
        )

    if not retrieved:
        cross_verify = "Insufficient evidence to cross-verify."
    else:
        titles = "; ".join(d["title"] for d in retrieved[:2])
        cross_verify = f"Retrieved {len(retrieved)} relevant fact-checking reference(s): {titles}."

    return {
        "summary": summary,
        "credibility_indicators": risk["credibility_indicators"] or ["No credibility indicators detected"],
        "risk_factors": risk["risk_factors"] or ["No significant risk factors detected"],
        "cross_source_verification": cross_verify,
        "confidence_assessment": conf_assessment,
        "sources": [d["source"] for d in retrieved],
        "disclaimer": (
            "This analysis is AI-generated using a Logistic Regression model trained on the WELFake dataset. "
            "It should not be the sole basis for determining truth. Always verify with primary sources."
        ),
    }


def run_agent(raw_text: str, prediction: dict, risk_analysis: dict, initial_docs: list) -> dict:
    """
    Agentic reasoning pipeline.
    State is passed explicitly between steps with conditional logic.
    """
    from .retriever import retrieve  # Local import to avoid circular dependencies

    state = {
        "raw_text": raw_text,
        "prediction": prediction,
        "risk_analysis": risk_analysis,
        "retrieved_docs": initial_docs,
        "steps_completed": [],
        "report": None,
        "used_llm": False,
        "performed_deep_scan": False,
        "agent_reasoning": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Step 1: Interpret model output
    state["steps_completed"].append("step1_model_interpretation")
    state["agent_reasoning"].append(f"Model predicted {prediction['label']} with {prediction['confidence']}% confidence.")

    # Step 2: Risk-Based Branching (Decision Node)
    # IF confidence is low OR risk score is extremely high, expand the search.
    if prediction["confidence_tier"] == "low" or risk_analysis["risk_score"] > 80:
        state["agent_reasoning"].append("CRITICAL: Low confidence or high risk detected. Initiating Deep Scan.")
        
        # Refine query based on top TF-IDF features for better retrieval
        top_features = " ".join(prediction.get("top_features", [])[:3])
        deep_scan_query = f"{raw_text[:100]} {top_features}"
        
        additional_docs = retrieve(deep_scan_query, top_k=2)
        
        # Merge unique docs
        existing_ids = {d["id"] for d in state["retrieved_docs"]}
        for doc in additional_docs:
            if doc["id"] not in existing_ids:
                state["retrieved_docs"].append(doc)
        
        state["performed_deep_scan"] = True
        state["steps_completed"].append("step2_deep_scan")
    else:
        state["agent_reasoning"].append("Normal confidence level. Proceeding with standard analysis.")

    # Step 3: Analysis of Risk Factors
    state["steps_completed"].append("step3_risk_analysis")

    # Step 4: Cross-check retrieved data
    retrieved_context = "\n".join(
        f"[{d['source']}] {d['title']}: {d['content'][:200]}"
        for d in state["retrieved_docs"]
    ) or "No relevant documents retrieved."
    state["steps_completed"].append("step4_cross_check")

    # Step 5: Evaluate uncertainty & Generate structured report
    state["steps_completed"].append("step5_uncertainty_evaluation")
    
    # Update prompt to inform LLM about the deep scan
    final_reasoning = "\n".join([f"- {r}" for r in state["agent_reasoning"]])
    
    prompt = REPORT_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        article_excerpt=raw_text[:500].replace("{", "(").replace("}", ")"),
        label=prediction["label"],
        confidence=prediction["confidence"],
        confidence_tier=prediction["confidence_tier"],
        real_prob=prediction["real_probability"],
        fake_prob=prediction["fake_probability"],
        top_features=", ".join(prediction.get("top_features", [])[:8]),
        risk_score=risk_analysis["risk_score"],
        risk_factors="; ".join(risk_analysis["risk_factors"]) or "None detected",
        credibility_indicators="; ".join(risk_analysis["credibility_indicators"]) or "None detected",
        retrieved_context=retrieved_context + f"\n\nAGENT REASONING PATH:\n{final_reasoning}",
    )

    llm_output = _call_hf_api(prompt)
    if llm_output:
        parsed = _extract_json(llm_output)
        if parsed and all(k in parsed for k in ["summary", "credibility_indicators", "risk_factors"]):
            state["report"] = parsed
            state["used_llm"] = True

    if state["report"] is None:
        state["report"] = _rule_based_report(state)

    state["steps_completed"].append("step6_report_generation")
    return state
