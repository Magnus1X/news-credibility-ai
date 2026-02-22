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


def remove_leakage_words(text: str) -> str:
    """
    Remove common publisher names and boilerplate text that causes data leakage.
    (e.g., 'Reuters', 'Getty Images', 'Featured Image')
    """
    if not text:
        return ""

    # Common words the model memorized as "Real" or "Fake" solely based on publishers
    leakage_words = [
        r'reuters', r'washington reuters', r'getty', r'getty images', 
        r'image', r'featured image', r'featured', r'twitter', r'twitter com', 
        r'breitbart', r'pic twitter', r'said', r'mr', r'ms'
    ]
    
    # Strip (Reuters) or (AP) prefixes often found at the beginning of real articles
    text = re.sub(r'^.*?\((reuters|ap)\).*?-', '', text, flags=re.IGNORECASE)

    for word in leakage_words:
        # Remove these exact leakage words as whole words
        text = re.sub(fr'\b{word}\b', '', text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """
    Main preprocessing pipeline (MUST match training preprocessing).
    
    Steps:
    1. Basic cleaning
    2. Stopword removal
    3. Leakage word removal
    
    Returns:
        Cleaned text ready for TF-IDF vectorization
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Basic cleaning
    text = basic_clean(text)

    # Step 2: Remove stopwords
    text = remove_stopwords(text)

    # Step 3: Remove Target Leakage Words
    text = remove_leakage_words(text)

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