"""Local (free, no-API-key) analysis: sentiment scoring and theme detection."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


@lru_cache(maxsize=4096)
def _kw_pattern(kw: str) -> re.Pattern:
    """Word-boundary pattern for a keyword or phrase.

    Substring matching mis-tags short keys badly ('ui' inside 'building',
    'ad' inside 'adaptation'), so every keyword match is boundary-checked.
    """
    esc = re.escape(kw.strip().lower()).replace(r"\ ", r"\s+").replace(" ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){esc}(?![a-z0-9])")


def _kw_in(low: str, kw: str) -> bool:
    return bool(_kw_pattern(kw).search(low))

# Theme keyword map: a hit on any keyword tags the feedback with that theme.
THEMES = {
    "Bugs & Crashes": [
        "bug", "crash", "crashes", "freeze", "frozen", "broken", "error",
        "glitch", "not working", "doesn't work", "doesnt work", "fails", "failed",
    ],
    "Performance": [
        "slow", "lag", "laggy", "loading", "takes forever", "battery", "drain",
        "memory", "heavy", "speed", "fast", "performance", "hang",
    ],
    "UI / UX": [
        "ui", "ux", "design", "interface", "layout", "confusing", "hard to use",
        "intuitive", "navigation", "dark mode", "theme", "font", "button",
    ],
    "Pricing & Billing": [
        "price", "pricing", "expensive", "subscription", "billing", "charged",
        "refund", "payment", "pay", "free trial", "cost", "money",
    ],
    "Login & Account": [
        "login", "log in", "sign in", "signin", "password", "account", "otp",
        "verification", "logout", "logged out", "authentication", "2fa",
    ],
    "Feature Requests": [
        "feature", "wish", "would be nice", "please add", "add support", "missing",
        "should have", "need option", "request", "suggestion", "hope you",
    ],
    "Customer Support": [
        "support", "customer service", "no response", "contacted", "help center",
        "ticket", "reply", "respond", "service team",
    ],
    "Ads & Notifications": [
        "ads", "advert", "ad ", "notification", "spam", "popup", "pop-up",
        "annoying", "intrusive",
    ],
}

_STOPWORDS = set("""
a an and are as at be but by for from has have i if in is it its of on or so
that the this to was were will with you your app very just really would can
get got dont don't do does did im i'm not no me my we they he she them their
there than then when what who how why all any some out up about after also
been being had more most other only own too s t can will don should now
""".split())


# Customer journey stages: keyword heuristics, checked in order.
JOURNEY_STAGES = {
    "Awareness": [
        "what is", "just heard", "heard about", "saw an ad", "anyone know",
        "just found", "discovered", "never heard", "is this the app",
    ],
    "Solution Search": [
        "looking for", "recommend me", "any recommendation", "best app for",
        "need an app", "suggestions for", "how do i", "is there an app",
        "any app that", "what app",
    ],
    "Comparison": [
        " vs ", "versus", "compared to", "better than", "alternative to",
        "alternatives", "switch from", "switching from", "instead of",
        "or should i use", "competitor",
    ],
    "Purchase Decision": [
        "worth it", "should i buy", "should i get", "free trial", "about to subscribe",
        "thinking of buying", "before i pay", "is premium worth", "upgrade to",
    ],
    "Experience & Advocacy": [
        "been using", "i use it", "i've used", "i have used", "love this",
        "hate this", "uninstalled", "cancelled", "canceled", "my experience",
        "recommend it", "would recommend", "stopped using",
    ],
}

JOURNEY_ORDER = list(JOURNEY_STAGES.keys()) + ["Unclassified"]

# Sources that are inherently first-hand product experience.
_EXPERIENCE_SOURCES = {"Google Play", "App Store"}


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment (-1..1 compound), a 0-100 score, and a P/N/N label."""
    df = df.copy()
    scores = df["text"].astype(str).map(lambda t: _analyzer.polarity_scores(t[:2000])["compound"])
    df["sentiment"] = scores
    df["sentiment100"] = ((scores + 1) * 50).round().astype(int)
    df["sentiment_label"] = pd.cut(
        scores, bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


def add_journey(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each item with a customer-journey stage."""
    df = df.copy()

    def detect(row) -> str:
        low = str(row["text"]).lower()
        for stage, kws in JOURNEY_STAGES.items():
            if any(_kw_in(low, k) for k in kws):
                return stage
        # A store review with a rating is first-hand experience by definition
        if row["source"] in _EXPERIENCE_SOURCES or pd.notna(row.get("rating")):
            return "Experience & Advocacy"
        return "Unclassified"

    df["journey"] = df.apply(detect, axis=1)
    return df


def journey_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts + share per journey stage, in funnel order."""
    counts = df["journey"].value_counts()
    total = max(len(df), 1)
    rows = [{
        "stage": s,
        "count": int(counts.get(s, 0)),
        "share": counts.get(s, 0) / total,
    } for s in JOURNEY_ORDER]
    return pd.DataFrame(rows)


def add_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each feedback item with matching themes (comma-separated string)."""
    df = df.copy()

    def detect(text: str) -> str:
        low = str(text).lower()
        hits = [theme for theme, kws in THEMES.items()
                if any(_kw_in(low, k) for k in kws)]
        return ", ".join(hits) if hits else "Other"

    df["themes"] = df["text"].map(detect)
    return df


def theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-theme counts and average sentiment, sorted by volume."""
    rows = []
    for theme in list(THEMES.keys()) + ["Other"]:
        mask = df["themes"].str.contains(re.escape(theme), na=False)
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "theme": theme,
            "mentions": len(sub),
            "avg_sentiment": round(float(sub["sentiment"].mean()), 3),
            "negative_share": round(float((sub["sentiment_label"] == "Negative").mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)


def top_terms(df: pd.DataFrame, label: str = "Negative", n: int = 15) -> list[tuple[str, int]]:
    """Most frequent meaningful words in feedback with the given sentiment label."""
    texts = df.loc[df["sentiment_label"] == label, "text"].astype(str)
    counter: Counter = Counter()
    for t in texts:
        words = re.findall(r"[a-zA-Z']{3,}", t.lower())
        counter.update(w for w in words if w not in _STOPWORDS)
    return counter.most_common(n)


# ------------------------------------------------------- Brand relevance filter

# Words that signal someone is talking about a software product, not using the
# brand name as an ordinary English word.
PRODUCT_CONTEXT = [
    "app", "apps", "tool", "software", "product", "platform", "workspace",
    "template", "templates", "subscription", "pricing", "plan", "account",
    "login", "sync", "api", "feature", "features", "update", "release",
    "download", "install", "installed", "desktop", "mobile", "widget",
    "integration", "database", "review", "alternative", "alternatives",
    "vs", "versus", "saas", "startup", "user", "users", "free tier",
    "premium", "browser", "extension", "export", "import", "beta",
    # hardware brands (HP, Dell, Logitech, ...)
    "laptop", "printer", "computer", "monitor", "keyboard", "ink",
    "cartridge", "device", "warranty", "driver", "firmware", "bios",
    "charger", "repair", "customer service", "model",
]

_CTX_ALT = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in PRODUCT_CONTEXT)

# Determiner + brand means the word is being used generically:
# "the notion of", "no notion that", "some notion about"
_GENERIC_DETERMINERS = r"(the|a|an|any|no|every|some|this|that|his|her|their|my|your|our|such)"


def _theme_words() -> set[str]:
    words = set()
    for kws in THEMES.values():
        words.update(kws)
    return words


_THEME_WORDS = _theme_words()


def _generic_patterns(b: str) -> str:
    """Regex alternatives for the keyword used as plain English, not a brand:
    'the notion of', 'a lot of hp', '800 hp', 'no notion that'."""
    return "|".join([
        rf"\b{_GENERIC_DETERMINERS}\s+{b}\s+(of|that|to|about)\b",   # the notion of
        rf"\b{_GENERIC_DETERMINERS}\s+{b}(?!\s*[a-z0-9])",           # a notion. / any hp,
        rf"\b{b}\s+(of|that|to)\b",                                  # notion that ...
        rf"(\d+\s*|lots?\s+of\s+|much\s+|more\s+|less\s+|extra\s+"   # 800 hp / lot of hp
        rf"|enough\s+|max\s+|full\s+|high\s+|low\s+){b}(?![a-z0-9])",
    ])


def _brand_relevance(text: str, brand: str, url: str = "") -> tuple[int, bool]:
    """Score how likely a text is about the brand rather than the English word.

    Returns (score, has_brand_signal). Brand signals are usages that only make
    sense for the product: its domain, a link into its own community, its name
    as a proper noun, or the name right next to product vocabulary ("hp printer").
    """
    low = text.lower()
    brand = brand.strip()
    b = re.escape(brand.lower())
    short = len(brand) <= 3  # hp, arc, bee: capitalization means nothing
    score = 0
    brand_signal = False

    # the brand's website is the strongest possible signal
    if re.search(rf"(?<![a-z0-9]){b}\.(so|com|io|app|ai|co|dev|org)\b", low):
        score += 2
        brand_signal = True

    # link into the brand's own community or store page (token match, not
    # substring: 'hp' inside 'watchparty' must not count)
    if url and re.search(rf"(?<![a-z0-9]){b}(?![a-z0-9])", url.lower()):
        score += 2
        brand_signal = True

    # brand written as a proper noun mid-sentence. Skipped for very short
    # brands: gamers write "5000 HP" for hit points, drivers "800 HP".
    if not short:
        cap = " ".join(p.capitalize() for p in brand.split())
        for m in re.finditer(rf"\b{re.escape(cap)}\b", text):
            prev = text[:m.start()].rstrip()
            if not prev or prev[-1] not in ".!?\"'“”‘’:;([-":
                score += 1
                brand_signal = True
                break

    # brand adjacent to product vocabulary ("hp printer", "the Notion app")
    if re.search(rf"(?<![a-z0-9]){b}(?![a-z0-9])[^.!?\n]{{0,40}}(?<![a-z0-9])({_CTX_ALT})(?![a-z0-9])", low) or \
       re.search(rf"(?<![a-z0-9])({_CTX_ALT})(?![a-z0-9])[^.!?\n]{{0,40}}(?<![a-z0-9]){b}(?![a-z0-9])", low):
        score += 1
        brand_signal = True

    # brand followed by a capitalized model name ("HP Spectre", "Notion Calendar")
    _not_models = {"i", "the", "a", "an", "it", "he", "she", "they", "we", "you",
                   "this", "that", "if", "but", "and", "or", "so", "my", "is"}
    for m in re.finditer(rf"(?<![A-Za-z0-9]){b}(?![A-Za-z0-9])", text, re.IGNORECASE):
        nxt = re.match(r"\s+([A-Z][A-Za-z0-9']*)", text[m.end():])  # case-sensitive
        if nxt and nxt.group(1).lower() not in _not_models:
            score += 1
            brand_signal = True
            break

    # product vocabulary anywhere in the text
    if any(_kw_in(low, w) for w in PRODUCT_CONTEXT):
        score += 1

    # feedback vocabulary (crash, slow, refund, ...) implies product talk
    if any(_kw_in(low, w) for w in _THEME_WORDS):
        score += 1

    # generic-English usage patterns count against, quantities most of all
    if re.search(rf"(\d+\s*|lots?\s+of\s+|much\s+|more\s+|less\s+|extra\s+"
                 rf"|enough\s+|max\s+|full\s+|high\s+|low\s+){b}(?![a-z0-9])", low):
        score -= 3
    if re.search(rf"\b{_GENERIC_DETERMINERS}\s+{b}\s+(of|that|to|about)\b", low):
        score -= 2
    if re.search(rf"\b{b}\s+(of|that|to)\b", low):
        score -= 1

    return score, brand_signal


def filter_brand_mentions(df: pd.DataFrame, brand: str):
    """Split fetched mentions into (relevant, dropped_count) for a brand keyword.

    Adaptive strictness: when the corpus shows the keyword being used as an
    ordinary English word (like "notion"), require a real brand signal and a
    higher score. Unambiguous brands (like "spotify") pass with a light touch.
    App store reviews are on-brand by definition; do not pass them through.
    """
    brand = brand.strip()
    if df.empty or not brand:
        return df, 0

    b = re.escape(brand.lower())
    scored = [
        _brand_relevance(str(row["text"]), brand, str(row.get("url", "")))
        for _, row in df.iterrows()
    ]

    # How often is the keyword used as plain English in this batch?
    generic_pat = _generic_patterns(b)
    generic_share = df["text"].astype(str).map(
        lambda t: bool(re.search(generic_pat, t.lower()))).mean()
    signal_share = sum(1 for _, sig in scored if sig) / len(scored)
    # very short keywords (hp, arc) collide with too many things: always strict
    ambiguous = len(brand) <= 3 or generic_share > 0.12 or signal_share < 0.15

    if ambiguous:
        mask = pd.Series([s >= 2 and sig for s, sig in scored], index=df.index)
    else:
        mask = pd.Series([s >= 1 for s, _ in scored], index=df.index)
    return df[mask].reset_index(drop=True), int((~mask).sum())
