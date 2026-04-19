# 🧠 News Credibility AI — Agentic Fact-Check Platform

An end-to-end, agentic AI-powered web application that detects misinformation and evaluates news credibility using a hybrid stack of **Machine Learning (NLP)**, **RAG (FAISS)**, and **Agentic AI Reasoning (LLM)**.

![Credibility AI Banner](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20React%20%7C%20Scikit--Learn%20%7C%20Mistral-blue?style=for-the-badge)

---

## 🌟 Overview

Credibility AI doesn't just give you a "Real" or "Fake" label. It executes a **5-step agentic pipeline** to scrape articles, analyze linguistic risk signals, retrieve context from a fact-check knowledge base, and generate a structured credibility report using Mistral-7B.

### Key Features
- 📝 **Dual Input Mode** — Analyze raw text or auto-scrape content directly from a URL.
- 📊 **Probabilistic Verdicts** — High-precision ML model provides confidence scores and feature importance.
- 🛡️ **Risk Signal Analysis** — Heuristic detection of clickbait, emotional language, and sensationalism.
- 🗄️ **RAG-Enhanced Retrieval** — Semantic search via FAISS over verified fact-check sources.
- 🧠 **Agentic Reporting** — LLM-generated analysis with Mistral-7B (HF API) or deterministic rule-based fallback.
- 📄 **PDF Export** — Download comprehensive credibility reports for offline reading.

---

## 🏗️ System Architecture

The platform follows a layered architecture, merging traditional ML with modern Agentic RAG.

```
User (Browser)
     ↕  HTTP
React Frontend (Vite + JSX)
     ↕  REST API
FastAPI Backend
     ├── scraper.py          → URL extraction (newspaper3k)
     ├── predictor.py        → ML Model (Logistic Regression)
     ├── risk_analyzer.py    → Heuristic signals (Emotional, Caps, Clickbait)
     ├── retriever.py        → RAG (FAISS + all-MiniLM-L6-v2)
     └── llm_agent.py        → Reasoning & Report (Mistral-7B)
```

---

## 🛠️ Technology Stack

### Backend (Python Engine)
*   **FastAPI**: High-performance REST API.
*   **Scikit-Learn**: Vectorization (TF-IDF) & ML Prediction (Logistic Regression).
*   **Sentence-Transformers**: Multi-QA embeddings for semantic RAG.
*   **FAISS**: Vector database for low-latency retrieval.
*   **NLTK & Newspaper3k**: Text extraction and natural language preprocessing.

### Frontend (User Interface)
*   **React 18**: Dynamic UI with sophisticated state management.
*   **Vite**: Ultra-fast build tool and development server.
*   **CSS3 & Lucide**: Modern design system and iconography.

---

## 📁 Project Structure

```bash
news-credibility-ai/
├── backend/
│   ├── app.py              # Classic ML Prediction API (Port 8000)
│   ├── agent_app.py        # Full Agentic Analysis API (Port 8001)
│   ├── run.py              # Single-command startup for both servers
│   ├── agent/              # Agentic pipeline modules (Predictor, Risk, RAG, LLM)
│   ├── model.pkl           # Trained ML Model weights
│   └── vectorizer.pkl      # Saved TF-IDF Vectorizer
├── frontend/
│   ├── src/                # React components & UI logic
│   └── index.css           # Custom design system
├── notebook/               # R&D track: EDA, Training, RAG Build, Agent Logic
└── data/                   # Dataset (Tracked via Git LFS)
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3.9+** & **Node.js 18+**
- **Git LFS** (for dataset files)
- **HuggingFace Token** (Optional, for LLM-enhanced reports)

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the FAISS Vector Index (Required for RAG)
python -c "from agent.retriever import build_index; build_index()"

# Start the servers
python run.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 API Documentation

### `POST /predict (Port 8000)`
Fast single-model prediction for quick triage.
- **Input**: `{ "text": "...", "url": "..." }`
- **Output**: Returns label, confidence_score, and reliable flag.

### `POST /analyze (Port 8001)`
Deep agentic analysis involving the full 5-step pipeline.
- **Input**: `{ "text": "...", "url": "..." }`
- **Output**: Returns structured report, risk signals, retrieved context, and ML metrics.

### `POST /analyze/pdf (Port 8001)`
Generates and returns a downloadable binary PDF credibility report.

---

## 🧪 Research & Development

The machine learning and agentic logic are documented across 7 specialized Jupyter notebooks:

1.  **`01_Exploration`**: EDA and dataset class balance.
2.  **`02_Preprocessing`**: NLTK-based cleaning & validation.
3.  **`03_Model_Training`**: TF-IDF & Logistic Regression training.
4.  **`04_Evaluation`**: Accuracy, Confusion Matrix, and feature importance.
5.  ****`05_RAG_Setup`**: FAISS index construction and retrieval testing.
6.  **`06_Agent_Pipeline`**: End-to-end integration of ML, Risk, and RAG.
7.  **`07_Prompt_Engineering`**: Anti-hallucination logic for Mistral-7B.

---

## 🛡️ Anti-Hallucination Design
To ensure transparency and truthfulness, the agent follows strict protocols:
- **Temperature = 0.1**: For high deterministic output.
- **Injected Context**: The LLM *only* writes based on the ML prediction and retrieved documents.
- **Rule-based Fallback**: If the LLM API is slow or unavailable, a deterministic summary is generated immediately.

---

## 📄 License & Team
*   **License**: Creative Commons / Educational Use.
*   **Team**: Built by a group of passionate AI engineers and Data Scientists focusing on responsible AI deployment.