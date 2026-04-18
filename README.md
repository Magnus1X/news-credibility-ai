# 🧠 News Credibility AI

An end-to-end AI-powered web application that detects whether a news article is **Real or Fake** using Natural Language Processing (NLP) and Machine Learning.

Paste any news text or drop a URL — our system scrapes, analyzes, and returns a credibility verdict with a confidence score in seconds.

---

## 🚀 Live Features

- 📝 **Text Input** — Paste raw news article text for instant analysis
- 🌐 **URL Input** — Enter any news URL; the system auto-scrapes the article
- 📊 **Confidence Score** — See *how confident* the AI is (e.g., 94.7%)
- ⚡ **Real-time Step Animation** — Live progress feedback during analysis
- 🏷️ **Clear Verdict** — `Real News` or `Fake News` result card

---

## 🏗️ Architecture

```
User (Browser)
     ↕  HTTP
React Frontend  (Vite + React)
     ↕  REST API  POST /predict
FastAPI Backend  (Python)
     ├── scraper.py          → Extracts article text from URLs (newspaper3k)
     ├── preprocessing.py    → NLP cleaning pipeline (NLTK + regex)
     ├── model.pkl           → Trained Logistic Regression model
     └── vectorizer.pkl      → Fitted TF-IDF vectorizer
```

---

## 🤖 Machine Learning Pipeline

The full ML workflow is documented across 4 Jupyter notebooks:

| Notebook | Description |
|---|---|
| `01_data_exploration.ipynb` | EDA — class balance, word frequencies, visualizations |
| `02_data_preprocessing.ipynb` | Text cleaning — lowercasing, regex, tokenization, stop-word removal |
| `03_feature_extraction_and_model.ipynb` | TF-IDF vectorization + Logistic Regression training |
| `04_model_evaluation.ipynb` | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC |

### Key Design Decisions

- **TF-IDF** converts text into numerical feature vectors (10,000 features)
- **Logistic Regression** (`C=2.0`, `class_weight='balanced'`) chosen for speed, interpretability, and probability output
- **80/20 train-test split**
- The **exact same preprocessing pipeline** used during training runs in production — zero training-serving skew

---

## 🛠️ Tech Stack

### Backend
| Tool | Purpose |
|---|---|
| FastAPI | REST API framework |
| scikit-learn | Logistic Regression + TF-IDF |
| NLTK | Tokenization & stop-word removal |
| newspaper3k | Article scraping from URLs |
| joblib | Model serialization (`.pkl`) |
| uvicorn | ASGI server |

### Frontend
| Tool | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool & dev server |
| Axios | HTTP client for API calls |
| Lucide React | Icons |

---

## 📁 Project Structure

```
news-credibility-ai/
├── backend/
│   ├── app.py              # FastAPI app — main prediction endpoint
│   ├── preprocessing.py    # NLP text cleaning pipeline
│   ├── scraper.py          # URL article extractor
│   ├── model.pkl           # Trained ML model
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   ├── requirements.txt    # Python dependencies
│   └── render.yaml         # Cloud deployment config (Render)
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── index.css       # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_extraction_and_model.ipynb
│   └── 04_model_evaluation.ipynb
├── data/                   # Raw dataset (tracked via Git LFS)
└── artifacts/              # Saved test sets (X_test.pkl, y_test.pkl)
```

---

## ⚙️ Setup & Running Locally

### Prerequisites
- Python 3.9+
- Node.js 18+

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/news-credibility-ai.git
cd news-credibility-ai
```

---

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --reload
```

The backend will start at **http://127.0.0.1:8000**

---

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend will start at **http://localhost:5173**

---

## 🌐 API Reference

### `POST /predict`

Predicts whether a news article is Real or Fake.

**Request Body** (JSON):
```json
{
  "text": "Paste your article text here...",
  "url": ""
}
```
or
```json
{
  "text": "",
  "url": "https://example.com/news-article"
}
```

**Response**:
```json
{
  "status": "success",
  "prediction": "Fake News",
  "confidence_score": 76.28,
  "input_source": "text",
  "text_length": 312,
  "message": "Credibility analysis completed successfully."
}
```

### `GET /`

Health check endpoint.

```json
{ "message": "News Credibility Analysis API is running", "status": "healthy" }
```

---

## 🔍 How It Works — Prediction Flow

1. **Input received** — raw text or a URL
2. **Scrape** (if URL) — `newspaper3k` extracts article title + body
3. **Validate** — must have ≥ 10 words
4. **Preprocess** — lowercase → remove punctuation → tokenize → remove stop-words
5. **Vectorize** — transform using saved TF-IDF vectorizer (10,000 features)
6. **Predict** — run saved Logistic Regression model
7. **Return** — label (`Real News` / `Fake News`) + confidence score

---

## ☁️ Deployment

### Backend — [Render](https://render.com)
A `render.yaml` is included in the `backend/` folder for one-click deployment.

### Frontend — [Vercel](https://vercel.com) / [Netlify](https://netlify.com)
Set the environment variable:
```
VITE_API_URL=https://your-backend-url.onrender.com
```

---

## 📦 Dataset

The dataset is stored using **Git LFS** due to its size (~234 MB).
It contains thousands of labeled news articles with `title`, `text`, and `label` (Real/Fake) columns.

To pull the dataset after cloning:
```bash
git lfs pull
```

---

## 👥 Team

Built as a group project — contributions span data science, ML engineering, backend API development, and frontend UI.

---

## 🤖 Milestone 2 — Agentic AI Misinformation Monitor

Builds on Milestone 1 to add a full agentic reasoning pipeline with RAG and LLM report generation.

### New Features
- 🔗 **5-step agentic pipeline** — explicit state between every step
- 🗄️ **FAISS RAG** — semantic retrieval over a fact-check knowledge base
- 🧠 **LLM report** — Mistral-7B via HuggingFace free-tier API
- 🛡️ **Zero-hallucination fallback** — rule-based report when LLM unavailable
- 📄 **PDF export** — downloadable credibility report

### Milestone 2 Architecture

```
Input (text / URL)
  ↓
[Step 1] predictor.py      → Milestone 1 model: label + confidence + top TF-IDF features
  ↓
[Step 2] risk_analyzer.py  → Heuristic signals: clickbait, emotional language, caps ratio
  ↓
[Step 3] retriever.py      → FAISS semantic search → top-3 KB documents
  ↓
[Step 4] Uncertainty eval  → Confidence tier check, word count validation
  ↓
[Step 5] llm_agent.py      → HF API (Mistral-7B) or rule-based fallback
  ↓
Structured JSON Report + optional PDF
```

### New Project Structure

```
backend/
├── agent_app.py            # Milestone 2 FastAPI app (port 8001)
├── agent/
│   ├── predictor.py        # Wraps Milestone 1 model
│   ├── risk_analyzer.py    # Heuristic risk signals
│   ├── retriever.py        # FAISS RAG retrieval
│   ├── llm_agent.py        # 5-step agent + LLM report
│   └── pdf_exporter.py     # PDF generation (reportlab)
└── .env.example            # HF_TOKEN config
notebook/
├── 05_rag_setup.ipynb      # FAISS index build + retrieval demo
├── 06_agent_pipeline.ipynb # Full pipeline step-by-step
└── 07_llm_prompt_testing.ipynb  # Prompt engineering + hallucination reduction
```

### Running Milestone 2

```bash
cd backend

# 1. Install new dependencies
pip install -r requirements.txt

# 2. (Optional) Set HuggingFace token for LLM reports
cp .env.example .env
# Edit .env and add your free token from https://huggingface.co/settings/tokens

# 3. Build FAISS index (run once)
python -c "from agent.retriever import build_index; build_index()"

# 4. Start Milestone 2 API (port 8001)
uvicorn agent_app:app --port 8001 --reload
```

### Milestone 2 API Reference

#### `POST /analyze`

Full agentic credibility analysis.

**Request:**
```json
{ "text": "Article text here...", "url": "" }
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "label": "Fake News",
    "confidence": 91.2,
    "confidence_tier": "high",
    "real_probability": 8.8,
    "fake_probability": 91.2,
    "top_features": ["breaking", "cover", "secret", "exposed"]
  },
  "risk_analysis": {
    "risk_score": 75,
    "risk_factors": ["Clickbait language detected (3 patterns)", "High emotional language"],
    "credibility_indicators": []
  },
  "retrieved_sources": [
    { "title": "Conspiracy theory language patterns", "source": "First Draft News", "relevance": 0.87 }
  ],
  "report": {
    "summary": "The ML model classified this article as Fake News with 91.2% high confidence...",
    "credibility_indicators": ["No credibility indicators detected"],
    "risk_factors": ["Clickbait language detected", "Conspiracy framing"],
    "cross_source_verification": "Retrieved context confirms conspiracy language patterns match known misinformation.",
    "confidence_assessment": "91.2% confidence is high-tier. The model is reliable at this level.",
    "sources": ["First Draft News", "Snopes Methodology"],
    "disclaimer": "This analysis is AI-generated and should not be the sole basis for determining truth."
  }
}
```

#### `POST /analyze/pdf`

Same pipeline — returns a downloadable PDF report.

#### `GET /agent/health`

```json
{ "status": "healthy", "milestone": 2, "version": "2.0.0" }
```

### Milestone 2 Notebooks

| Notebook | Description |
|---|---|
| `05_rag_setup.ipynb` | Builds FAISS index, embedding visualization, retrieval demo |
| `06_agent_pipeline.ipynb` | Full 5-step pipeline demo with fake/real article comparison |
| `07_llm_prompt_testing.ipynb` | Prompt experiments, temperature ablation, hallucination reduction |

### Prompt Engineering — Anti-Hallucination Design

| Technique | Effect |
|---|---|
| System prompt rules (`NEVER fabricate`) | Prevents invented sources/quotes |
| `"Insufficient evidence"` fallback | Prevents confident wrong answers |
| Structured JSON output schema | Prevents free-form narrative invention |
| Temperature = 0.1 | Minimizes randomness |
| All facts injected into prompt | LLM can only use provided data |
| Rule-based fallback | Zero hallucination guarantee |

---

## 📄 License

This project is for educational purposes.