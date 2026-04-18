"""
Prediction Module — wraps the Milestone 1 Logistic Regression model.
Label convention (WELFake): 0 = Real News, 1 = Fake News
"""
import os
import numpy as np
import joblib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
_vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

LABEL_MAP = {0: "Real News", 1: "Fake News"}
CONFIDENCE_THRESHOLDS = {"high": 0.80, "medium": 0.65}

# Domain note: model trained on WELFake (US wire-service real vs conspiracy fake)
# Results below 80% confidence should be treated as uncertain
UNCERTAIN_THRESHOLD = 0.80


def predict(raw_text: str) -> dict:
    """
    Run the full Milestone-1 pipeline on raw_text.
    Returns label, confidence, cleaned text, and top TF-IDF features.
    """
    cleaned = preprocess_text(raw_text)
    vector = _vectorizer.transform([cleaned])
    pred_int = int(_model.predict(vector)[0])
    probs = _model.predict_proba(vector)[0]
    confidence = float(np.max(probs))

    # Confidence tier
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        confidence_tier = "high"
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        confidence_tier = "medium"
    else:
        confidence_tier = "low"

    # Top 10 TF-IDF features that drove the prediction
    feature_names = _vectorizer.get_feature_names_out()
    dense = vector.toarray()[0]
    top_indices = np.argsort(dense)[::-1][:10]
    top_features = [feature_names[i] for i in top_indices if dense[i] > 0]

    return {
        "label": LABEL_MAP[pred_int],
        "label_int": pred_int,
        "confidence": round(confidence * 100, 2),
        "confidence_tier": confidence_tier,
        "fake_probability": round(float(probs[1]) * 100, 2),
        "real_probability": round(float(probs[0]) * 100, 2),
        "top_features": top_features,
        "cleaned_text": cleaned,
        "word_count": len(raw_text.split()),
        "uncertain": confidence < UNCERTAIN_THRESHOLD,
    }
