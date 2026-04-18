"""
Risk Analyzer — computes heuristic risk signals from raw article text.
All signals are derived from the same text used in Milestone 1 training.
"""
import re

# Patterns observed in WELFake fake-news articles
CLICKBAIT_PATTERNS = [
    r"\b(BREAKING|SHOCKING|BOMBSHELL|EXPOSED|UNBELIEVABLE|WATCH|MUST.?READ)\b",
    r"you won'?t believe",
    r"they don'?t want you to know",
    r"\?\?\?|!!!",
]

EMOTIONAL_WORDS = [
    "outrage", "scandal", "corrupt", "evil", "destroy", "attack",
    "crisis", "disaster", "threat", "danger", "lie", "hoax", "fraud",
    "conspiracy", "cover.?up", "traitor", "radical",
]

CREDIBILITY_MARKERS = [
    # Wire services / Western press
    r"\b(reuters|associated press|ap|bbc|according to|officials said|confirmed)\b",
    r"\b(study|research|report|survey|data|statistics)\b",
    r"\b(percent|million|billion|\$\d+)\b",
    # Indian political / parliamentary credibility markers
    r"\b(lok sabha|rajya sabha|parliament|constitution|amendment|ministry|minister)\b",
    r"\b(prime minister|chief minister|supreme court|high court|election commission)\b",
    r"\b(introduced|tabled|passed|enacted|gazette|notification|bill|act)\b",
    r"\b(narendra modi|amit shah|government|centre|state government|nda|upa)\b",
    r"\b(pti|ani|ndtv|the hindu|hindustan times|times of india|indian express)\b",
]


def analyze_risk(raw_text: str) -> dict:
    text_lower = raw_text.lower()

    clickbait_hits = sum(
        1 for p in CLICKBAIT_PATTERNS if re.search(p, raw_text, re.IGNORECASE)
    )
    emotional_hits = sum(
        1 for w in EMOTIONAL_WORDS if re.search(w, text_lower)
    )
    credibility_hits = sum(
        1 for p in CREDIBILITY_MARKERS if re.search(p, text_lower, re.IGNORECASE)
    )

    # ALL-CAPS ratio (common in fake headlines)
    words = raw_text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)

    # Exclamation / question mark density
    punct_density = (raw_text.count("!") + raw_text.count("?")) / max(len(raw_text), 1) * 1000

    risk_factors = []
    if clickbait_hits > 0:
        risk_factors.append("[HEADLINE] Sensationalized phrasing detected. This pattern is often used to manipulate reader emotions rather than provide balanced facts.")
    if emotional_hits >= 3:
        risk_factors.append("[TONE] Highly emotional or biased language detected. This may indicate an attempt to influence opinion rather than report objectively.")
    if caps_ratio > 0.1:
        risk_factors.append("[FORMAT] Unusual use of capitalized words. Professional news typically follows standard sentence casing for credibility.")
    if punct_density > 2:
        risk_factors.append("[FORMAT] Excessive punctuation usage. High density of '!' or '?' is rarely found in authoritative journalistic reporting.")
    if credibility_hits == 0:
        risk_factors.append("[SOURCES] No citations to verified news agencies found. Reference to primary sources (Reuters, AP, PTI) is a key pillar of credible news.")

    credibility_indicators = []
    if credibility_hits > 0:
        credibility_indicators.append(f"[SOURCES] Professional Citations: The article references {credibility_hits} recognized news/data sources.")
    if clickbait_hits == 0:
        credibility_indicators.append("[HEADLINE] Factual Phrasing: The headline avoids sensational traps and follows objective reporting standards.")
    if emotional_hits < 2:
        credibility_indicators.append("[TONE] Neutral Reporting: The language is balanced and focuses on informational content rather than bias.")
    if caps_ratio <= 0.05:
        credibility_indicators.append("[FORMAT] Standard Presentation: The article follows professional capitalization and formatting conventions.")

    # Risk score 0–100
    risk_score = min(100, clickbait_hits * 25 + emotional_hits * 5 + int(caps_ratio * 120) + int(punct_density * 8))
    risk_score = max(0, risk_score - credibility_hits * 12)

    return {
        "risk_factors": risk_factors,
        "credibility_indicators": credibility_indicators,
        "risk_score": risk_score,
        "clickbait_hits": clickbait_hits,
        "emotional_hits": emotional_hits,
        "credibility_hits": credibility_hits,
        "caps_ratio": round(caps_ratio, 3),
    }
