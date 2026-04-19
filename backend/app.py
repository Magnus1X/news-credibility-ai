# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

from scraper import extract_text_from_url
from preprocessing import preprocess_text, validate_input_text

# ---------------------------
# Initialize FastAPI App
# ---------------------------
app = FastAPI(
    title="AI News Credibility Analysis API",
    description="API for classifying news articles as Real or Fake using ML + NLP",
    version="1.0.0"
)

# CORS (for Vite React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "https://news-credibility-ai.vercel.app",
        "https://news-credibility-ai-yymk.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load Model & Vectorizer (ONCE at startup)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

print("Loaded model from:", os.path.abspath(MODEL_PATH))
print("Loaded vectorizer from:", os.path.abspath(VECTORIZER_PATH))

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and Vectorizer loaded successfully")
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {e}")

# ---------------------------
# Request Schema (Input Format)
# ---------------------------
class NewsRequest(BaseModel):
    text: str = ""
    url: str = ""


# ---------------------------
# Root Endpoint (Health Check)
# ---------------------------
@app.get("/")
def home():
    return {
        "message": "News Credibility Analysis API is running",
        "status": "healthy"
    }


# ---------------------------
# Main Prediction Endpoint
# ---------------------------
@app.post("/predict")
@app.post("/analyze")
def predict_news(request: NewsRequest):
    """
    Predict whether a news article is Real or Fake
    using the full ML + Agentic logic.
    """
    try:
        user_text = request.text.strip() if request.text else ""
        user_url = request.url.strip() if request.url else ""

        # ---------------------------
        # Step 1: Get Text (URL or Direct)
        # ---------------------------
        if user_url:
            extracted_text = extract_text_from_url(user_url)

            if not extracted_text:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to extract valid article content from the provided URL."
                )

            raw_text = extracted_text
            source = "url"

        elif user_text:
            # Auto-detect URL embedded inside pasted text
            import re as _re
            url_match = _re.search(r'https?://[^\s]+', user_text)
            if url_match:
                embedded_url = url_match.group(0).rstrip(')')
                extracted = extract_text_from_url(embedded_url)
                if extracted and len(extracted.split()) >= 80:
                    raw_text = extracted
                    source = "url"
                else:
                    raw_text = user_text
                    source = "text"
            else:
                raw_text = user_text
                source = "text"

        else:
            raise HTTPException(
                status_code=400,
                detail="Please provide either news text or a valid URL."
            )

        # ---------------------------
        # Step 2: Validate Input Length
        # ---------------------------
        if not validate_input_text(raw_text):
            raise HTTPException(
                status_code=400,
                detail="Input text is too short or invalid for analysis."
            )

        # ---------------------------
        # Step 3: Preprocess Text (SAME as training pipeline)
        # ---------------------------
        cleaned_text = preprocess_text(raw_text)
        print(cleaned_text)
        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="Text preprocessing resulted in empty content."
            )

        # ---------------------------
        # Step 4: Vectorize using SAVED TF-IDF
        # ---------------------------
        vector = vectorizer.transform([cleaned_text])
        print(vector)
        # ---------------------------
        # Step 5: Model Prediction (FIXED ORDER)
        # ---------------------------
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = float(np.max(probabilities))

        print("Raw prediction:", prediction)
        print("Probabilities:", probabilities)
        print("Confidence:", confidence)

        # ---------------------------
        # Step 6: Label Mapping
        # ---------------------------
        label_map = {
            0: "Real News",
            1: "Fake News"
        }
        predicted_label = label_map.get(int(prediction), str(prediction))

        # ---------------------------
        # Step 7: Agentic AI Reasoning (Milestone 2)
        # ---------------------------
        from agent.risk_analyzer import analyze_risk
        from agent.retriever import retrieve
        from agent.llm_agent import run_agent

        # A. Risk Analysis
        risk = analyze_risk(raw_text)

        # B. Semantic Retrieval (RAG)
        # Use a combination of prediction and original text for a robust query
        retrieval_query = f"{predicted_label}: {raw_text[:200]}"
        retrieved_docs = retrieve(retrieval_query, top_k=3)

        # C. Run Agent Reasoning Pipeline
        # Map prediction to agent format
        prediction_data = {
            "label": predicted_label,
            "confidence": round(confidence * 100, 2),
            "confidence_tier": "high" if confidence > 0.85 else "medium" if confidence > 0.65 else "low",
            "real_probability": round(probabilities[0] * 100, 2),
            "fake_probability": round(probabilities[1] * 100, 2),
            "top_features": [vectorizer.get_feature_names_out()[i] for i in vector.toarray().argsort()[0][-10:][::-1]]
        }

        agent_state = run_agent(raw_text, prediction_data, risk, retrieved_docs)

        # ---------------------------
        # Step 8: Response (Strictly Aligned with Frontend)
        # ---------------------------
        word_count = len(raw_text.split())
        return {
            "status": "success",
            "input_source": source,
            "text_length": len(raw_text),
            "word_count": word_count,
            "reliable": word_count >= 80,
            "uncertain": prediction_data["confidence"] < 70,
            
            # Milestone 1 & 2 shared prediction data
            "prediction": {
                "label": predicted_label,
                "confidence": prediction_data["confidence"],
                "confidence_tier": prediction_data["confidence_tier"],
                "real_probability": prediction_data["real_probability"],
                "fake_probability": prediction_data["fake_probability"],
                "top_features": prediction_data["top_features"]
            },

            # Milestone 2 specific agent data
            "risk_analysis": {
                "risk_score": risk["risk_score"],
                "risk_factors": risk["risk_factors"],
                "credibility_indicators": risk["credibility_indicators"]
            },
            "report": agent_state["report"],
            "retrieved_sources": [
                {
                    "source": d["source"],
                    "title": d["title"],
                    "relevance": d.get("relevance_score", 1.0)
                } for d in agent_state["retrieved_docs"]
            ],
            "pipeline_steps": agent_state["steps_completed"],
            "used_llm": agent_state["used_llm"],
            "message": "Full agentic credibility assessment completed."
        }

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 INTERNAL SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# PDF Export Endpoint (Extension)
# ---------------------------
class ExportRequest(BaseModel):
    title: str = "News Analysis Report"
    report_data: dict
    prediction: str
    confidence: float

@app.post("/analyze/pdf")
def export_pdf(request: ExportRequest):
    """Generates a PDF report and returns the file path/binary."""
    from agent.pdf_exporter import generate_report_pdf
    try:
        filename = f"report_{request.prediction.replace(' ', '_')}.pdf"
        output_path = os.path.join(BASE_DIR, "exports", filename)
        
        # Ensure exports directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        generate_report_pdf(request.dict(), output_path)
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=output_path, 
            filename=filename,
            media_type='application/pdf'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failed: {str(e)}")