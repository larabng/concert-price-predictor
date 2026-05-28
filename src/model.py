"""
model.py — Feature engineering, model training, evaluation, and persistence.

Trains and compares Random Forest and XGBoost on the concert dataset,
with and without NLP-derived features (4 models total):
  RF_base  — Random Forest, structured features only
  RF_nlp   — Random Forest, structured + NLP features
  XGB_base — XGBoost, structured features only
  XGB_nlp  — XGBoost, structured + NLP features (best model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = DATA_DIR / "models"

TARGET = "minprice"
LOG_TARGET = True  # train on log1p(price) for better handling of the skewed distribution

# Numeric features from Source 1 (concert structured data)
NUMERIC_BASE = ["weekend", "pop", "month", "score"]

# Categorical features to one-hot encode
CATEGORICAL_COLS = ["genre"]

# Columns to drop before feature matrix construction
DROP_COLS = ["city", "artist", "venue", "bio", "wiki_found", TARGET]

# NLP-derived features from Source 2 (Wikipedia bios → NLP pipeline)
NLP_COLS = ["sentiment_score", "hype_score"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    include_nlp: bool = True,
    fit_encoders: bool = True,
    encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a numeric feature matrix from the merged DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame (must contain all NUMERIC_BASE and CATEGORICAL_COLS).
    include_nlp:
        Whether to append sentiment_score and hype_score columns.
    fit_encoders:
        True → derive category lists from df; False → use provided encoders.
    encoders:
        Pre-fitted encoder dict {col: [categories]} for consistent inference encoding.

    Returns
    -------
    (X, encoders) where X is a fully numeric DataFrame.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    feature_cols = list(NUMERIC_BASE)

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit_encoders:
            cats = sorted(df[col].dropna().unique().tolist())
            encoders[col] = cats
        else:
            cats = encoders.get(col, [])

        for cat in cats:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            feature_cols.append(f"{col}_{cat}")

    if include_nlp:
        for c in NLP_COLS:
            if c in df.columns:
                feature_cols.append(c)

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].fillna(0)
    return X, encoders


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Return RMSE, MAE, and R² metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
    if label:
        print(f"{label:40s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Train all four models and return results + artifacts.

    Parameters
    ----------
    df:
        Enriched DataFrame (must include NLP columns if available).
    test_size:
        Held-out fraction for evaluation.
    seed:
        Random seed for reproducibility.
    save:
        Persist models and encoders to MODEL_DIR when True.

    Returns
    -------
    dict with keys: results, models, encoders_base, encoders_nlp,
                    X_test_base, X_test_nlp, y_test
    """
    has_nlp = all(c in df.columns for c in NLP_COLS)
    y_raw = df[TARGET].values
    y = np.log1p(y_raw) if LOG_TARGET else y_raw

    X_base, enc_base = build_feature_matrix(df, include_nlp=False)
    X_nlp, enc_nlp = build_feature_matrix(df, include_nlp=has_nlp)

    (X_tr_base, X_te_base,
     X_tr_nlp, X_te_nlp,
     y_train, y_test) = train_test_split(
        X_base, X_nlp, y, test_size=test_size, random_state=seed
    )

    configs = {
        "RF_base":  (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
                     X_tr_base, X_te_base),
        "RF_nlp":   (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
                     X_tr_nlp, X_te_nlp),
        "XGB_base": (XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                   random_state=seed, verbosity=0),
                     X_tr_base, X_te_base),
        "XGB_nlp":  (XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                   random_state=seed, verbosity=0),
                     X_tr_nlp, X_te_nlp),
    }

    results, models = {}, {}
    for name, (model, X_tr, X_te) in configs.items():
        print(f"Training {name} ...")
        model.fit(X_tr, y_train)
        preds_log = model.predict(X_te)
        # Evaluate in original price scale for interpretability
        preds = np.expm1(preds_log) if LOG_TARGET else preds_log
        y_te_orig = np.expm1(y_test) if LOG_TARGET else y_test
        results[name] = evaluate(y_te_orig, preds, label=name)
        models[name] = model

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        with open(MODEL_DIR / "encoders_base.json", "w") as f:
            json.dump(enc_base, f)
        with open(MODEL_DIR / "encoders_nlp.json", "w") as f:
            json.dump(enc_nlp, f)
        pd.DataFrame(results).T.to_csv(MODEL_DIR / "results.csv")
        print(f"\nModels saved -> {MODEL_DIR}")

    return {
        "results":       results,
        "models":        models,
        "encoders_base": enc_base,
        "encoders_nlp":  enc_nlp,
        "X_test_base":   X_te_base,
        "X_test_nlp":    X_te_nlp,
        "y_test":        np.expm1(y_test) if LOG_TARGET else y_test,  # original price scale
    }


# ---------------------------------------------------------------------------
# Loading saved models
# ---------------------------------------------------------------------------

def load_model(name: str = "XGB_nlp") -> tuple:
    """Load a saved model and its encoders from MODEL_DIR."""
    model_path = MODEL_DIR / f"{name}.joblib"
    enc_key = "nlp" if "nlp" in name else "base"
    enc_path = MODEL_DIR / f"encoders_{enc_key}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_models() first."
        )

    model = joblib.load(model_path)
    with open(enc_path) as f:
        encoders = json.load(f)
    return model, encoders


def get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Return normalised feature importances as {name: importance}."""
    imp = model.feature_importances_
    total = imp.sum()
    return {name: float(v / total) for name, v in zip(feature_names, imp)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_data
    from nlp_features import enrich_with_nlp

    df = load_data()
    df = enrich_with_nlp(df, use_transformer=False)
    out = train_models(df)
    print("\nResults:")
    print(pd.DataFrame(out["results"]).T)
