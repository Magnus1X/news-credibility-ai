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
        risk_factors.append(f"Clickbait language detected ({clickbait_hits} pattern(s))")
    if emotional_hits >= 3:
        risk_factors.append(f"High emotional language ({emotional_hits} trigger words)")
    if caps_ratio > 0.1:
        risk_factors.append(f"Excessive ALL-CAPS usage ({caps_ratio:.0%} of words)")
    if punct_density > 2:
        risk_factors.append("Excessive punctuation (! / ?)")
    if credibility_hits == 0:
        risk_factors.append("No credibility markers (sources, data, attribution)")

    credibility_indicators = []
    if credibility_hits > 0:
        credibility_indicators.append(f"Contains {credibility_hits} credibility marker(s)")
    if clickbait_hits == 0:
        credibility_indicators.append("No clickbait language detected")
    if emotional_hits < 2:
        credibility_indicators.append("Low emotional language")
    if caps_ratio <= 0.05:
        credibility_indicators.append("Normal capitalization pattern")

    # Risk score 0–100
    risk_score = min(100, clickbait_hits * 20 + emotional_hits * 5 + int(caps_ratio * 100) + int(punct_density * 5))
    risk_score = max(0, risk_score - credibility_hits * 10)

    return {
        "risk_factors": risk_factors,
        "credibility_indicators": credibility_indicators,
        "risk_score": risk_score,
        "clickbait_hits": clickbait_hits,
        "emotional_hits": emotional_hits,
        "credibility_hits": credibility_hits,
        "caps_ratio": round(caps_ratio, 3),
    }
