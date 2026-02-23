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
def predict_news(request: NewsRequest):
    """
    Predict whether a news article is Real or Fake
    based on:
    - Direct text input OR
    - URL input (auto-scraped)
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

        # Debug logs (safe now)
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
        # Step 7: Response (Frontend uses this)
        # ---------------------------
        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence_score": round(confidence * 100, 2),
            "input_source": source,
            "text_length": len(raw_text),
            "message": "Credibility analysis completed successfully."
        }

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 INTERNAL SERVER ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))