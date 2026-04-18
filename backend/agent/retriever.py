"""
Retrieval Module — FAISS-backed semantic search over a fact-check knowledge base.
Uses sentence-transformers (all-MiniLM-L6-v2, free, runs locally, ~80 MB).
Falls back gracefully if FAISS/transformers are unavailable.
"""
import os
import json
import pickle
import numpy as np

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

# Curated knowledge base — representative fact-check patterns from WELFake domain
KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "title": "How to identify clickbait headlines",
        "content": "Clickbait headlines use emotional triggers, ALL-CAPS, and vague promises. Legitimate news uses specific, factual language with named sources.",
        "source": "Media Literacy Guide",
        "category": "media_literacy",
    },
    {
        "id": "kb_002",
        "title": "Reuters fact-checking methodology",
        "content": "Reuters fact-checks verify claims against primary sources, official statements, and peer-reviewed research. Unverified claims are labeled as such.",
        "source": "Reuters Fact Check",
        "category": "fact_checking",
    },
    {
        "id": "kb_003",
        "title": "Signs of misinformation in political news",
        "content": "Political misinformation often lacks named sources, uses extreme emotional language, misquotes officials, and spreads through social media without verification.",
        "source": "PolitiFact Methodology",
        "category": "political_news",
    },
    {
        "id": "kb_004",
        "title": "Credible news source characteristics",
        "content": "Credible sources cite named officials, provide data with context, include opposing viewpoints, have editorial standards, and issue corrections when wrong.",
        "source": "AP Stylebook",
        "category": "credibility",
    },
    {
        "id": "kb_005",
        "title": "Satire vs. fake news distinction",
        "content": "Satire is clearly labeled and uses exaggeration for commentary. Fake news presents false information as factual reporting without satirical intent.",
        "source": "Snopes Methodology",
        "category": "media_literacy",
    },
    {
        "id": "kb_006",
        "title": "Conspiracy theory language patterns",
        "content": "Conspiracy theories use phrases like 'they don't want you to know', 'cover-up', 'deep state', and claim suppressed evidence without verifiable sources.",
        "source": "First Draft News",
        "category": "misinformation",
    },
    {
        "id": "kb_007",
        "title": "Scientific misinformation indicators",
        "content": "Scientific misinformation misrepresents study findings, cherry-picks data, cites retracted papers, and uses anecdotal evidence instead of peer-reviewed research.",
        "source": "FullFact.org",
        "category": "science",
    },
    {
        "id": "kb_008",
        "title": "Election misinformation patterns",
        "content": "Election misinformation includes false claims about voting procedures, fabricated candidate statements, and unverified fraud allegations without evidence.",
        "source": "Election Integrity Partnership",
        "category": "political_news",
    },
    {
        "id": "kb_009",
        "title": "Health misinformation warning signs",
        "content": "Health misinformation promotes unproven cures, exaggerates risks, contradicts medical consensus, and often sells products alongside false claims.",
        "source": "WHO Infodemic Management",
        "category": "health",
    },
    {
        "id": "kb_010",
        "title": "Verification checklist for news articles",
        "content": "Verify: (1) Is the source named? (2) Can the claim be independently confirmed? (3) Is the headline consistent with the body? (4) Is the date current? (5) Are images authentic?",
        "source": "IFCN Code of Principles",
        "category": "fact_checking",
    },
]


def _get_encoder():
    """Lazy-load sentence transformer to avoid startup cost."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def build_index():
    """Build and persist FAISS index from knowledge base."""
    try:
        import faiss
        encoder = _get_encoder()
        if encoder is None:
            return False

        texts = [f"{doc['title']} {doc['content']}" for doc in KNOWLEDGE_BASE]
        embeddings = encoder.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "encoder_name": "all-MiniLM-L6-v2"}, f)

        with open(KB_PATH, "w") as f:
            json.dump(KNOWLEDGE_BASE, f, indent=2)

        print(f"FAISS index built: {len(KNOWLEDGE_BASE)} documents, dim={embeddings.shape[1]}")
        return True
    except Exception as e:
        print(f"FAISS build failed: {e}")
        return False


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve top_k relevant knowledge base entries for the query.
    Falls back to keyword matching if FAISS is unavailable.
    """
    # Try FAISS semantic search
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            import faiss
            encoder = _get_encoder()
            if encoder:
                with open(FAISS_INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                index = data["index"]

                q_emb = encoder.encode([query], convert_to_numpy=True).astype("float32")
                faiss.normalize_L2(q_emb)
                scores, indices = index.search(q_emb, top_k)

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(KNOWLEDGE_BASE):
                        doc = KNOWLEDGE_BASE[idx].copy()
                        doc["relevance_score"] = round(float(score), 3)
                        results.append(doc)
                return results
        except Exception as e:
            print(f"FAISS retrieval failed, falling back to keyword: {e}")

    # Keyword fallback
    return _keyword_retrieve(query, top_k)


def _keyword_retrieve(query: str, top_k: int) -> list[dict]:
    """Simple keyword overlap retrieval as fallback."""
    query_words = set(query.lower().split())
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        d = doc.copy()
        d["relevance_score"] = score / max(len(query_words), 1)
        results.append(d)
    return results if results else [KNOWLEDGE_BASE[0]]
