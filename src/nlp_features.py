"""
nlp_features.py — NLP feature extraction from Wikipedia artist biographies.

Two approaches are implemented and compared:
  Approach A — Transformer: distilbert-base-uncased-finetuned-sst-2-english
  Approach B — Keyword heuristic: rule-based scoring on a curated keyword list

Both produce:
  sentiment_score ∈ [-1, 1]  (confidence-weighted positivity of the bio text)
  hype_score      ∈ [0, 1]   (density of prestige/popularity keywords)

Integration with ML block:
  The NLP scores are appended as numeric features to the structured dataset,
  so the ML model can use artist reputation signals for price prediction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

_pipeline = None
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

# Prestige/popularity keywords for hype scoring
HYPE_KEYWORDS = [
    "legendary", "world-class", "iconic", "sold-out", "massive", "record-breaking",
    "unforgettable", "spectacular", "globally", "international", "prestigious",
    "award-winning", "premier", "acclaimed", "must-attend", "electrifying",
    "biggest", "largest", "world's", "famous", "renowned", "celebrated",
    "influential", "best-selling", "chart-topping", "grammy",
]

# Positive words for Approach B keyword sentiment
POSITIVE_WORDS = [
    "acclaimed", "legendary", "iconic", "celebrated", "renowned", "influential",
    "award-winning", "grammy", "best-selling", "chart-topping", "globally",
    "international", "spectacular", "world-class", "successful", "popular",
    "prominent", "noted", "known", "respected",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Approach A — Transformer (DistilBERT)
# ---------------------------------------------------------------------------

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=MODEL_ID,
            truncation=True,
            max_length=512,
        )
    return _pipeline


def transformer_sentiment(text: str) -> float:
    """Approach A: DistilBERT sentiment in [-1, 1]."""
    pipe = _get_pipeline()
    result = pipe(text)[0]
    score = result["score"]
    return score if result["label"] == "POSITIVE" else -score


def batch_transformer_sentiment(texts: list[str], batch_size: int = 32) -> np.ndarray:
    pipe = _get_pipeline()
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT sentiment"):
        batch = texts[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            s = r["score"] if r["label"] == "POSITIVE" else -r["score"]
            scores.append(s)
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Approach B — Keyword heuristic
# ---------------------------------------------------------------------------

def keyword_sentiment(text: str) -> float:
    """Approach B: rule-based sentiment score in [0, 1].

    Counts POSITIVE_WORDS hits; 5+ hits → score 1.0.
    """
    text_lower = text.lower()
    hits = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    return min(hits / 5.0, 1.0)


def batch_keyword_sentiment(texts: list[str]) -> np.ndarray:
    return np.array([keyword_sentiment(t) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hype score (shared across both approaches)
# ---------------------------------------------------------------------------

def hype_score(text: str) -> float:
    """Prestige/hype score in [0, 1]. 5+ HYPE_KEYWORDS → 1.0."""
    text_lower = text.lower()
    hits = sum(1 for kw in HYPE_KEYWORDS if kw in text_lower)
    return min(hits / 5.0, 1.0)


def batch_hype(texts: list[str]) -> np.ndarray:
    return np.array([hype_score(t) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Approach comparison (used in notebook 02)
# ---------------------------------------------------------------------------

def compare_approaches(
    texts: list[str],
    labels: list[str] | None = None,
    use_transformer: bool = True,
) -> pd.DataFrame:
    """Side-by-side comparison of Approach A vs Approach B on a text list.

    Returns DataFrame with columns:
        label, text_snippet, transformer_sentiment, keyword_sentiment,
        hype_score, abs_delta
    """
    kw_scores = batch_keyword_sentiment(texts)
    hype_scores_arr = batch_hype(texts)

    if use_transformer:
        try:
            tr_scores = batch_transformer_sentiment(texts)
        except Exception:
            tr_scores = np.full(len(texts), np.nan, dtype=np.float32)
    else:
        tr_scores = np.full(len(texts), np.nan, dtype=np.float32)

    rows = []
    for i, text in enumerate(texts):
        rows.append({
            "label":                labels[i] if labels else f"text_{i}",
            "text_snippet":         text[:90] + "…",
            "transformer_sent":     round(float(tr_scores[i]), 3) if not np.isnan(tr_scores[i]) else "N/A",
            "keyword_sent":         round(float(kw_scores[i]), 3),
            "hype_score":           round(float(hype_scores_arr[i]), 3),
            "abs_delta":            round(abs(float(tr_scores[i]) - float(kw_scores[i])), 3)
                                    if not np.isnan(tr_scores[i]) else "N/A",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline — enrich DataFrame
# ---------------------------------------------------------------------------

def enrich_with_nlp(
    df: pd.DataFrame,
    text_col: str = "bio",
    use_transformer: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Add sentiment_score and hype_score columns to df.

    Parameters
    ----------
    df:
        DataFrame containing a text column (default 'bio').
    text_col:
        Column with artist biography text.
    use_transformer:
        True → Approach A (DistilBERT).  False → Approach B (keyword, default).
    cache_path:
        Path for caching computed scores (avoids recomputation on reruns).

    Returns
    -------
    Copy of df with two added columns: sentiment_score, hype_score.
    """
    if cache_path is None:
        suffix = "transformer" if use_transformer else "keyword"
        cache_path = DATA_DIR / f"nlp_scores_{suffix}_cache.csv"
    cache_path = Path(cache_path)

    texts = df[text_col].fillna("").tolist()

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(df):
            df = df.copy()
            df["sentiment_score"] = cached["sentiment_score"].values
            df["hype_score"] = cached["hype_score"].values
            return df

    hype_scores_arr = batch_hype(texts)

    if use_transformer:
        sentiment_scores = batch_transformer_sentiment(texts)
    else:
        sentiment_scores = batch_keyword_sentiment(texts)

    df = df.copy()
    df["sentiment_score"] = sentiment_scores
    df["hype_score"] = hype_scores_arr

    cache_df = pd.DataFrame({
        "sentiment_score": sentiment_scores,
        "hype_score": hype_scores_arr,
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Price explanation (used by predict.py and Streamlit app)
# ---------------------------------------------------------------------------

def generate_price_explanation(
    predicted_price: float,
    feature_importances: dict[str, float],
    input_features: dict,
) -> str:
    """Generate a human-readable rationale for a predicted ticket price."""
    top = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:4]

    level = "high" if predicted_price > 100 else ("moderate" if predicted_price > 40 else "low")
    lines = [
        f"The predicted minimum ticket price of **${predicted_price:.0f}** is "
        f"**{level}** for a concert of this profile.",
        "",
        "Main price drivers:",
    ]

    friendly = {
        "score":           "artist popularity score",
        "pop":             "city population (market size)",
        "weekend":         "weekend event",
        "month":           "month of the event",
        "sentiment_score": "artist bio sentiment",
        "hype_score":      "artist bio hype level",
    }

    for feat, imp in top:
        val = input_features.get(feat, "n/a")
        fname = friendly.get(feat, feat.replace("_", " "))
        lines.append(f"- **{fname}** ({imp*100:.1f}% importance): {val}")

    sentiment = input_features.get("sentiment_score", 0)
    hype = input_features.get("hype_score", 0)
    if isinstance(sentiment, (int, float)):
        tone = "positive" if sentiment > 0.3 else ("neutral" if sentiment > -0.3 else "negative")
        lines.append(
            f"\nArtist bio sentiment: **{tone}** (score {sentiment:.2f}), "
            f"hype score: {hype:.2f}."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    samples = [
        "Taylor Alison Swift is an American singer-songwriter. An influential figure in pop "
        "music, she is one of the best-selling music artists of all time.",
        "A popular music artist known for live performances and chart-topping hits.",
        "Godsmack is an American rock band from Lawrence, Massachusetts, formed in 1995.",
    ]
    labels = ["Taylor Swift", "Unknown Artist", "Godsmack"]
    print(compare_approaches(samples, labels=labels, use_transformer=False).to_string())
