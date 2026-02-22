import re
import nltk
from typing import List
from nltk.corpus import stopwords

# Ensure stopwords are available (first-time setup safe)
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


def basic_clean(text: str) -> str:
    """
    Perform basic cleaning:
    - Lowercasing
    - Remove URLs
    - Remove HTML tags
    - Remove special characters & numbers
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove non-alphabetic characters (keep only letters)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords.
    Helps TF-IDF focus on meaningful words.
    """
    if not text:
        return ""

    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]

    return " ".join(filtered_words)


def preprocess_text(text: str) -> str:
    """
    Main preprocessing pipeline (MUST match training preprocessing).
    
    Steps:
    1. Basic cleaning
    2. Stopword removal
    
    Returns:
        Cleaned text ready for TF-IDF vectorization
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Basic cleaning
    text = basic_clean(text)

    # Step 2: Remove stopwords
    text = remove_stopwords(text)

    return text


def combine_title_content(title: str, content: str) -> str:
    """
    Combine title and content (important for fake news detection).
    Use this if your dataset used title + content during training.
    """
    title = title if isinstance(title, str) else ""
    content = content if isinstance(content, str) else ""

    combined = f"{title} {content}".strip()
    return combined


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Preprocess a list of texts (useful for batch predictions).
    """
    if not texts:
        return []

    return [preprocess_text(text) for text in texts]


def validate_input_text(text: str, min_length: int = 20) -> bool:
    """
    Validate if input text is suitable for prediction.
    
    Prevents:
    - Empty inputs
    - Extremely short inputs
    - Garbage text
    """
    if not text or not isinstance(text, str):
        return False

    # Remove spaces and check length
    if len(text.strip()) < min_length:
        return False

    return True