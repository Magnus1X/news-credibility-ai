import re
import nltk
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources
try:
    stop_words = set(stopwords.words("english"))
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Preprocessing that EXACTLY matches training notebook pipeline:
    1. Lowercase
    2. Remove punctuation & special characters
    3. Remove extra whitespace
    4. Tokenize using word_tokenize (NOT split)
    5. Remove stopwords
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Lowercase (matches notebook)
    text = text.lower()

    # Step 2: Remove punctuation & special characters (same regex as notebook)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 3: Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # Step 4: Tokenization (CRITICAL - match training)
    tokens = word_tokenize(text)

    # Step 5: Stopword removal (same logic as notebook)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join back to string (TF-IDF expects string)
    cleaned_text = " ".join(filtered_tokens)

    return cleaned_text


def validate_input_text(text: str, min_length: int = 50) -> bool:
    """
    Increased min length because model trained on long articles.
    """
    if not text or not isinstance(text, str):
        return False

    if len(text.strip()) < min_length:
        return False

    return True