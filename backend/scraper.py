from newspaper import Article
import requests


def extract_text_from_url(url: str) -> str:
    """
    Extracts clean article text (title + body) from a news URL.
    
    Returns:
        str: Extracted article text (title + content)
        Empty string if extraction fails
    """

    if not url or not isinstance(url, str):
        return ""

    try:
        # Validate URL accessibility first
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"Failed to access URL. Status Code: {response.status_code}")
            return ""

        # Use newspaper3k to extract article
        article = Article(url)
        article.download()
        article.parse()

        title = article.title if article.title else ""
        text = article.text if article.text else ""

        # Combine title + content (better for ML prediction)
        full_text = f"{title} {text}".strip()

        # Basic validation (avoid empty or very short content)
        if len(full_text) < 50:
            print("Extracted content too short or invalid.")
            return ""

        return full_text

    except Exception as e:
        print(f"Error extracting article: {e}")
        return ""


def is_valid_news_content(text: str) -> bool:
    """
    Checks if extracted text looks like a valid news article.
    Useful for filtering bad URLs (blogs, ads, etc.)
    """
    if not text:
        return False

    # Minimum length threshold for a news article
    if len(text.split()) < 50:
        return False

    return True