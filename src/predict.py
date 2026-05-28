"""
predict.py — Inference pipeline for single-concert ticket price prediction.

Loads a pre-trained model and produces a price estimate plus a
natural-language explanation derived from feature importances and NLP scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model import build_feature_matrix, get_feature_importances, load_model
from nlp_features import (
    generate_price_explanation,
    hype_score,
    keyword_sentiment,
    transformer_sentiment,
)


def predict_price(
    inputs: dict,
    model_name: str = "XGB_nlp",
    use_transformer: bool = False,
) -> dict:
    """Predict the minimum ticket price for a single concert event.

    Parameters
    ----------
    inputs : dict
        Keys required:
          artist (str), genre (str), city (str), pop (int),
          weekend (int 0/1), month (int 1–12), score (float 65–100),
          bio (str)  ← artist Wikipedia biography for NLP scoring
    model_name : str
        One of RF_base, RF_nlp, XGB_base, XGB_nlp.
    use_transformer : bool
        True → DistilBERT sentiment (Approach A).
        False → keyword heuristic (Approach B, default — faster).

    Returns
    -------
    dict with keys:
        predicted_price, explanation, feature_importances,
        sentiment_score, hype_score
    """
    model, encoders = load_model(model_name)
    include_nlp = "nlp" in model_name

    row = dict(inputs)
    bio = row.get("bio", "")
    sent, hype = 0.0, 0.0

    if include_nlp and bio:
        if use_transformer:
            try:
                sent = float(transformer_sentiment(bio))
            except Exception:
                sent = float(keyword_sentiment(bio))
        else:
            sent = float(keyword_sentiment(bio))
        hype = hype_score(bio)
        row["sentiment_score"] = sent
        row["hype_score"] = hype

    df_row = pd.DataFrame([row])
    X, _ = build_feature_matrix(
        df_row,
        include_nlp=include_nlp,
        fit_encoders=False,
        encoders=encoders,
    )

    expected_cols = (
        model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
    )
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[list(expected_cols)]

    from model import LOG_TARGET
    raw = float(model.predict(X)[0])
    price = float(max(np.expm1(raw) if LOG_TARGET else raw, 1.0))
    feat_imp = get_feature_importances(model, list(X.columns))
    explanation = generate_price_explanation(
        price, feat_imp, {**row, "sentiment_score": sent, "hype_score": hype}
    )

    return {
        "predicted_price":    round(price, 2),
        "explanation":        explanation,
        "feature_importances": feat_imp,
        "sentiment_score":    round(sent, 4),
        "hype_score":         round(hype, 4),
    }


def batch_predict(df: pd.DataFrame, model_name: str = "XGB_nlp") -> pd.DataFrame:
    """Fast batch inference without NLP (for EDA / evaluation)."""
    model, encoders = load_model(model_name)
    include_nlp = "nlp" in model_name

    X, _ = build_feature_matrix(
        df, include_nlp=include_nlp, fit_encoders=False, encoders=encoders
    )
    expected_cols = (
        model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
    )
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[list(expected_cols)]

    df = df.copy()
    from model import LOG_TARGET
    raw_preds = model.predict(X)
    prices = np.expm1(raw_preds) if LOG_TARGET else raw_preds
    df["predicted_price"] = np.maximum(prices, 1.0).round(2)
    return df


if __name__ == "__main__":
    sample = {
        "artist":  "Taylor Swift",
        "genre":   "Rock",
        "city":    "Chicago",
        "pop":     2_696_555,
        "weekend": 1,
        "month":   7,
        "score":   100,
        "bio": (
            "Taylor Alison Swift is an American singer-songwriter. An influential figure "
            "in popular culture, she is one of the best-selling music artists of all time, "
            "with Grammy awards and record-breaking tour grosses."
        ),
    }
    result = predict_price(sample, model_name="XGB_nlp", use_transformer=False)
    print(f"Predicted price: ${result['predicted_price']}")
    print(f"Sentiment: {result['sentiment_score']}")
    print(f"Hype:      {result['hype_score']}")
    print("\nExplanation:\n", result["explanation"])
