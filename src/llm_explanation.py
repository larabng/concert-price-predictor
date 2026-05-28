"""
llm_explanation.py — GPT-3.5-turbo price explanation via OpenAI API.

Uses gpt-3.5-turbo to generate a natural-language explanation of a predicted
concert ticket price, integrating ML feature importances and the artist bio.

NLP block comparison:
  Approach A — DistilBERT: transformer sentiment for feature extraction
  Approach B — Keyword heuristic: fast rule-based sentiment for feature extraction
  Approach C — GPT-3.5-turbo: LLM for natural-language explanation generation

Integration with other blocks:
  Input from ML block:  predicted price + top feature importances
  Input from NLP block: artist Wikipedia bio text
  Output:               2-3 sentence human-readable price rationale shown in UI
"""

from __future__ import annotations

import os
from typing import Optional

FRIENDLY_NAMES = {
    "score":           "artist popularity score",
    "pop":             "city market size (population)",
    "weekend":         "weekend event",
    "month":           "month of the event",
    "sentiment_score": "artist bio sentiment",
    "hype_score":      "artist prestige/hype level",
}


def _get_openai_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")


def generate_llm_explanation(
    artist: str,
    predicted_price: float,
    top_features: dict[str, float],
    bio: str,
    token: Optional[str] = None,
) -> Optional[str]:
    """Generate a price explanation using GPT-3.5-turbo.

    Parameters
    ----------
    artist:
        Artist or band name.
    predicted_price:
        Predicted minimum ticket price in USD.
    top_features:
        Dict of {feature_name: importance} from the ML model.
    bio:
        Artist Wikipedia biography excerpt.
    token:
        OpenAI API key. Auto-detected from OPENAI_API_KEY env var if not given.

    Returns
    -------
    str explanation, or None if the API is unavailable.
    """
    api_key = token or _get_openai_key()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    top4 = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:4]
    factors_text = "; ".join(
        f"{FRIENDLY_NAMES.get(k, k.replace('_', ' '))} ({v*100:.0f}% importance)"
        for k, v in top4
    )
    bio_excerpt = bio[:300].rsplit(" ", 1)[0] + "…" if len(bio) > 300 else bio

    system_msg = (
        "You are a helpful concert ticket pricing analyst. "
        "Write clear, concise 2-sentence explanations of predicted ticket prices. "
        "Always reference the specific artist and the top pricing factors. "
        "Be informative and friendly."
    )

    user_msg = (
        f"Artist: {artist}\n"
        f"Predicted minimum ticket price: ${predicted_price:.0f} USD\n"
        f"Top pricing factors: {factors_text}\n"
        f"Artist bio: {bio_excerpt}\n\n"
        f"Write exactly 2 sentences explaining this ticket price prediction."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=120,
            temperature=0.5,
        )
        explanation = response.choices[0].message.content.strip()
        if explanation:
            return explanation
    except Exception:
        pass

    return None
