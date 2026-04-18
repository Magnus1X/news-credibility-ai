"""
agent_app.py — Milestone 2 FastAPI application.
Exposes:
  POST /analyze        → full agentic credibility report (JSON)
  POST /analyze/pdf    → same report exported as PDF
  GET  /agent/health   → health check
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from scraper import extract_text_from_url
from preprocessing import validate_input_text, preprocess_text
from agent.predictor import predict
from agent.risk_analyzer import analyze_risk
from agent.retriever import retrieve
from agent.llm_agent import run_agent
from agent.pdf_exporter import export_pdf
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
_vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

app = FastAPI(
    title="News Credibility Agentic AI API",
    description="Milestone 2 — Multi-step agentic credibility analysis with RAG + LLM report generation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://news-credibility-ai.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = ""
    url: str = ""


def _get_raw_text(request: AnalyzeRequest) -> tuple[str, str]:
    """Extract raw text and source label from request."""
    import re as _re

    if request.url.strip():
        text = extract_text_from_url(request.url.strip())
        if not text:
            raise HTTPException(400, "Unable to extract article content from URL.")
        return text, "url"

    if request.text.strip():
        # Auto-detect URL embedded inside pasted text
        url_match = _re.search(r'https?://[^\s]+', request.text.strip())
        if url_match:
            embedded_url = url_match.group(0).rstrip(')')
            extracted = extract_text_from_url(embedded_url)
            if extracted and len(extracted.split()) >= 80:
                return extracted, "url"
        return request.text.strip(), "text"

    raise HTTPException(400, "Provide either 'text' or 'url'.")


def _run_pipeline(raw_text: str) -> dict:
    """Execute the full 5-step agentic pipeline and return state."""
    if not validate_input_text(raw_text):
        raise HTTPException(400, "Input text is too short (minimum 10 words).")

    prediction = predict(raw_text)
    risk_analysis = analyze_risk(raw_text)

    # Override uncertain: if risk score is low AND credibility markers exist,
    # the model's domain bias is likely causing a false Fake verdict
    credibility_hits = risk_analysis.get("credibility_hits", 0)
    risk_score = risk_analysis.get("risk_score", 0)
    if prediction["uncertain"] or (prediction["label"] == "Fake News" and credibility_hits >= 2 and risk_score <= 20):
        prediction["uncertain"] = True

    # Build retrieval query from top features + label
    query = f"{prediction['label']} {' '.join(prediction['top_features'][:5])}"
    retrieved_docs = retrieve(query, top_k=3)

    state = run_agent(raw_text, prediction, risk_analysis, retrieved_docs)
    return state


@app.get("/")
def home():
    return {"message": "News Credibility Analysis API is running", "status": "healthy", "version": "2.0.0"}


@app.get("/agent/health")
def health():
    return {"status": "healthy", "milestone": 2, "version": "2.0.0"}


@app.post("/predict")
def predict_news(request: AnalyzeRequest):
    """Milestone 1 — fast single-model prediction."""
    import re as _re
    try:
        user_text = request.text.strip()
        user_url = request.url.strip()

        if user_url:
            raw_text = extract_text_from_url(user_url)
            if not raw_text:
                raise HTTPException(400, "Unable to extract article content from URL.")
            source = "url"
        elif user_text:
            url_match = _re.search(r'https?://[^\s]+', user_text)
            if url_match:
                extracted = extract_text_from_url(url_match.group(0).rstrip(')'))
                if extracted and len(extracted.split()) >= 80:
                    raw_text, source = extracted, "url"
                else:
                    raw_text, source = user_text, "text"
            else:
                raw_text, source = user_text, "text"
        else:
            raise HTTPException(400, "Provide either text or url.")

        if not validate_input_text(raw_text):
            raise HTTPException(400, "Input text is too short.")

        cleaned = preprocess_text(raw_text)
        vector = _vectorizer.transform([cleaned])
        pred_int = int(_model.predict(vector)[0])
        probs = _model.predict_proba(vector)[0]
        confidence = float(np.max(probs))
        label = "Real News" if pred_int == 0 else "Fake News"

        from agent.risk_analyzer import analyze_risk
        risk = analyze_risk(raw_text)
        uncertain = confidence < 0.80 or (
            label == "Fake News" and risk["credibility_hits"] >= 2 and risk["risk_score"] <= 20
        )
        word_count = len(raw_text.split())

        return {
            "status": "success",
            "prediction": label,
            "confidence_score": round(confidence * 100, 2),
            "input_source": source,
            "text_length": len(raw_text),
            "word_count": word_count,
            "reliable": word_count >= 80,
            "uncertain": uncertain,
            "message": "Credibility analysis completed successfully." if word_count >= 80 else f"Short input ({word_count} words) — results may be unreliable."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """
    Full agentic credibility analysis.
    Returns structured JSON report with all pipeline metadata.
    """
    raw_text, source = _get_raw_text(request)
    state = _run_pipeline(raw_text)

    return {
        "status": "success",
        "input_source": source,
        "text_length": len(raw_text),
        "word_count": state["prediction"]["word_count"],
        "reliable": state["prediction"]["word_count"] >= 80,
        "uncertain": state["prediction"]["uncertain"],
        "pipeline_steps": state["steps_completed"],
        "used_llm": state["used_llm"],
        "timestamp": state["timestamp"],
        # Core ML output
        "prediction": {
            "label": state["prediction"]["label"],
            "confidence": state["prediction"]["confidence"],
            "confidence_tier": state["prediction"]["confidence_tier"],
            "real_probability": state["prediction"]["real_probability"],
            "fake_probability": state["prediction"]["fake_probability"],
            "top_features": state["prediction"]["top_features"],
        },
        # Risk signals
        "risk_analysis": {
            "risk_score": state["risk_analysis"]["risk_score"],
            "risk_factors": state["risk_analysis"]["risk_factors"],
            "credibility_indicators": state["risk_analysis"]["credibility_indicators"],
        },
        # Retrieved context
        "retrieved_sources": [
            {"title": d["title"], "source": d["source"], "relevance": d.get("relevance_score", 0)}
            for d in state["retrieved_docs"]
        ],
        # LLM-generated structured report
        "report": state["report"],
    }


@app.post("/analyze/pdf")
def analyze_pdf(request: AnalyzeRequest):
    """
    Same pipeline as /analyze but returns a downloadable PDF report.
    """
    raw_text, _ = _get_raw_text(request)
    state = _run_pipeline(raw_text)

    try:
        pdf_bytes = export_pdf(
            report=state["report"],
            prediction=state["prediction"],
            risk_analysis=state["risk_analysis"],
            raw_text=raw_text,
        )
    except ImportError:
        raise HTTPException(500, "reportlab not installed. Run: pip install reportlab")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=credibility_report.pdf"},
    )
