import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download only once (safe)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    if not text:
        return ""

    # 1. Lowercase (same as training)
    text = text.lower()

    # 2. Remove punctuation & special characters (same regex as notebook)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # 3. Remove extra spaces (same as notebook)
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Tokenize (same as training)
    words = word_tokenize(text)

    # 5. Remove stopwords (same logic as notebook)
    filtered_words = [word for word in words if word not in stop_words]

    # 6. Return cleaned_content EXACTLY like training column
    return " ".join(filtered_words)


def validate_input_text(text: str) -> bool:
    if not text:
        return False
    # Require at least ~50 words (since model trained on long articles)
    return len(text.split()) >= 10