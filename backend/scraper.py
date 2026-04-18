"""
scraper.py — Robust article extractor.
Strategy:
  1. newspaper3k (fast, handles most sites)
  2. BeautifulSoup fallback (handles paywalled/JS-blocked sites like TOI, NYT, etc.)
  3. Returns empty string if both fail
"""
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Ordered list of CSS selectors to try for article body extraction
ARTICLE_SELECTORS = [
    "article",
    "[data-articlebody]",
    "div.article-txt",
    "div.artText",
    "div._s30J",
    "div.Normal",
    "div.article__body",
    "div.article-body",
    "div.story-body",
    "div.post-content",
    "div.entry-content",
    "div.content-body",
    "div.body-content",
    "div.article-content",
    "div.story-content",
    "div.main-content",
    "div[class*='article-body']",
    "div[class*='story-body']",
    "div[class*='article-content']",
    "div[class*='article']",
    "div[itemprop='articleBody']",
    "section[itemprop='articleBody']",
]


def _extract_with_newspaper(url: str) -> str:
    """Try newspaper3k extraction."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        title = article.title or ""
        body = article.text or ""
        full = f"{title} {body}".strip()
        return full if len(full.split()) >= 50 else ""
    except Exception:
        return ""


def _extract_with_bs4(url: str) -> str:
    """BeautifulSoup fallback — tries multiple selectors."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "noscript", "figure"]):
            tag.decompose()

        # Try each selector in order, pick the one with most words
        best = ""
        for sel in ARTICLE_SELECTORS:
            elements = soup.select(sel)
            if not elements:
                continue
            text = " ".join(e.get_text(separator=" ", strip=True) for e in elements)
            text = " ".join(text.split())  # collapse whitespace
            if len(text.split()) > len(best.split()):
                best = text

        # Last resort: grab all <p> tags
        if len(best.split()) < 50:
            paragraphs = soup.find_all("p")
            p_text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            p_text = " ".join(p_text.split())
            if len(p_text.split()) > len(best.split()):
                best = p_text

        # Extract title from <title> or <h1>
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        full = f"{title} {best}".strip() if title not in best else best
        return full if len(full.split()) >= 30 else ""

    except Exception as e:
        print(f"BS4 scrape error: {e}")
        return ""


def extract_text_from_url(url: str) -> str:
    """
    Extract article text from a URL.
    Tries newspaper3k first, falls back to BeautifulSoup.
    Returns empty string if both fail.
    """
    if not url or not isinstance(url, str):
        return ""

    # Strategy 1: newspaper3k
    text = _extract_with_newspaper(url)
    if len(text.split()) >= 80:
        print(f"[scraper] newspaper3k: {len(text.split())} words")
        return text

    # Strategy 2: BeautifulSoup
    print(f"[scraper] newspaper3k got {len(text.split())} words, trying BS4 fallback...")
    text_bs4 = _extract_with_bs4(url)
    if len(text_bs4.split()) >= len(text.split()):
        print(f"[scraper] BS4: {len(text_bs4.split())} words")
        return text_bs4

    print(f"[scraper] Both strategies failed. Best: {len(text.split())} words")
    return text  # return whatever we have, even if short


def is_valid_news_content(text: str) -> bool:
    if not text:
        return False
    return len(text.split()) >= 50
