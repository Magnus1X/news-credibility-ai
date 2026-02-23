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

## 📄 License

This project is for educational purposes.