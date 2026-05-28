"""
streamlit_app.py — Concert Ticket Price Predictor (interactive web UI).

Two tabs:
  1. Price Predictor  — input concert details → predicted price + NLP explanation
  2. Model Insights   — EDA charts, NLP analysis, model comparison, feature importances
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from data_loader import load_data
from model import MODEL_DIR, build_feature_matrix, get_feature_importances, load_model, train_models
from nlp_features import compare_approaches, enrich_with_nlp
from predict import predict_price

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Concert Ticket Price Predictor",
    page_icon="🎵",
    layout="wide",
)

GENRES = [
    "Rock", "Hip-Hop/Rap", "Country", "R&B", "Metal",
    "Pop", "Dance/Electronic", "Other",
]
MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
# City size presets → approximate population values
CITY_SIZES = {
    "Small city (~100 K)":    100_000,
    "Medium city (~400 K)":   400_000,
    "Large city (~800 K)":    800_000,
    "Major metro (~2 M)":   2_000_000,
    "Mega city (~3.5 M)":   3_500_000,
}

DEFAULT_BIO = (
    "A popular music artist known for live performances and chart-topping hits, "
    "with a loyal fanbase and a history of sold-out shows."
)


# ---------------------------------------------------------------------------
# Cached data + model loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading concert data …")
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner="Computing NLP features …")
def get_enriched_data() -> pd.DataFrame:
    df = get_data()
    return enrich_with_nlp(df, use_transformer=False)


@st.cache_resource(show_spinner="Training models (first run only) …")
def get_trained_models():
    df = get_enriched_data()
    return train_models(df, save=True)


@st.cache_resource
def get_model(name: str = "XGB_nlp"):
    try:
        return load_model(name)
    except FileNotFoundError:
        get_trained_models()
        return load_model(name)


# ---------------------------------------------------------------------------
# Tab 1 — Price Predictor
# ---------------------------------------------------------------------------

def tab_predictor():
    st.header("🎟️ Concert Ticket Price Predictor")
    st.markdown(
        "Enter the concert details below. The AI model combines **structured event data** "
        "(Source 1) with **NLP analysis of the artist's Wikipedia biography** (Source 2) "
        "to estimate the minimum ticket price."
    )

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            artist = st.text_input("Artist / Band name", value="Taylor Swift")
            genre = st.selectbox("Genre", GENRES, index=0)
            city_label = st.selectbox("City size", list(CITY_SIZES.keys()), index=3)
            pop = CITY_SIZES[city_label]

        with col2:
            month_label = st.selectbox("Month", list(MONTHS.values()), index=6)
            month = [k for k, v in MONTHS.items() if v == month_label][0]
            weekend = st.checkbox("Weekend event", value=True)
            score = st.slider(
                "Artist popularity score (0–100, Spotify-style)",
                min_value=50, max_value=100, value=85,
                help="Higher = more popular artist. 65–100 is the range in the training data.",
            )

        with col3:
            model_choice = st.selectbox(
                "Model",
                ["XGB_nlp", "RF_nlp", "XGB_base", "RF_base"],
                help="*_nlp models include NLP features from the artist bio.",
            )
            use_transformer = st.checkbox(
                "Use DistilBERT for sentiment (Approach A, slower)",
                value=False,
                help="Unchecked = keyword heuristic (Approach B, fast).",
            )

        bio = st.text_area(
            "Artist Wikipedia biography (for NLP sentiment & hype scoring)",
            height=120,
            value=(
                "Taylor Alison Swift is an American singer-songwriter. An influential "
                "figure in popular culture, she is one of the best-selling music artists "
                "of all time, with record-breaking tour grosses and multiple Grammy awards."
            ),
            help="Paste or edit the artist's Wikipedia intro. Determines the NLP features.",
        )

        submitted = st.form_submit_button("Predict Ticket Price 🔮", use_container_width=True)

    if submitted:
        inputs = {
            "artist":  artist,
            "genre":   genre,
            "city":    city_label,
            "pop":     pop,
            "weekend": int(weekend),
            "month":   month,
            "score":   float(score),
            "bio":     bio,
        }

        with st.spinner("Predicting …"):
            try:
                result = predict_price(inputs, model_name=model_choice,
                                        use_transformer=use_transformer)
            except FileNotFoundError:
                st.info("Training models for the first time — this takes ~30 seconds …")
                get_trained_models()
                result = predict_price(inputs, model_name=model_choice,
                                        use_transformer=use_transformer)

        price = result["predicted_price"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Min. Ticket Price", f"${price:.0f}")
        c2.metric("Bio Sentiment Score", f"{result['sentiment_score']:+.3f}")
        c3.metric("Bio Hype Score", f"{result['hype_score']:.3f}")

        st.markdown("---")
        st.subheader("Why this price?")
        st.markdown(result["explanation"])

        st.subheader("Top Feature Importances")
        imp = result["feature_importances"]
        top_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        fig = px.bar(
            x=[v for _, v in top_imp],
            y=[k for k, _ in top_imp],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title=f"Feature Importances — {model_choice}",
            color=[v for _, v in top_imp],
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Model Insights
# ---------------------------------------------------------------------------

def tab_insights():
    st.header("📊 Model Insights & EDA")

    df = get_enriched_data()

    # ── EDA ──────────────────────────────────────────────────────────────────
    st.subheader("1. Exploratory Data Analysis")
    st.caption(
        f"Source 1: Real concert ticket price data — {len(df):,} rows, "
        f"{df['artist'].nunique()} unique artists, {df['genre'].nunique()} genres"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Ticket Price Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        capped = df[df["minprice"] <= 400]
        ax.hist(capped["minprice"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Min. Ticket Price (USD)")
        ax.set_ylabel("Count")
        ax.set_title("Price Distribution (≤ $400, 97th pct)")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig = px.box(
            df[df["minprice"] <= 400], x="genre", y="minprice",
            color="genre", title="Min. Price by Genre",
            labels={"minprice": "Price (USD)", "genre": "Genre"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.scatter(
            df[df["minprice"] <= 400], x="score", y="minprice",
            color="genre", opacity=0.5, trendline="ols",
            labels={"score": "Artist Popularity Score", "minprice": "Price (USD)"},
            title="Artist Score vs Ticket Price",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        month_avg = df.groupby("month")["minprice"].median().reset_index()
        month_avg["month_name"] = month_avg["month"].map(MONTHS)
        fig = px.bar(
            month_avg, x="month_name", y="minprice", color="minprice",
            color_continuous_scale="Viridis",
            labels={"minprice": "Median Price (USD)", "month_name": "Month"},
            title="Median Price by Month",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weekend vs weekday
    st.markdown("**Weekend vs Weekday Prices**")
    weekend_df = df.groupby("weekend")["minprice"].agg(["median", "mean", "count"]).reset_index()
    weekend_df["weekend"] = weekend_df["weekend"].map({0: "Weekday", 1: "Weekend"})
    st.dataframe(weekend_df.rename(columns={"weekend": "Event type",
                                             "median": "Median price ($)",
                                             "mean": "Mean price ($)",
                                             "count": "Count"}),
                 use_container_width=False)

    # Correlation matrix
    st.subheader("2. Correlation Matrix")
    num_cols = ["minprice", "score", "pop", "weekend", "month", "sentiment_score", "hype_score"]
    corr_cols = [c for c in num_cols if c in df.columns]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

    # NLP score distributions
    if "sentiment_score" in df.columns:
        st.subheader("3. NLP Features (Source 2: Wikipedia Bios)")
        c5, c6 = st.columns(2)
        with c5:
            fig = px.scatter(
                df[df["minprice"] <= 400], x="sentiment_score", y="minprice",
                opacity=0.4, trendline="ols",
                title="Keyword Sentiment vs Price",
                labels={"sentiment_score": "Sentiment Score", "minprice": "Price (USD)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            fig = px.scatter(
                df[df["minprice"] <= 400], x="hype_score", y="minprice",
                opacity=0.4, trendline="ols",
                title="Hype Score vs Price",
                labels={"hype_score": "Hype Score", "minprice": "Price (USD)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # NLP approach comparison on sample artists
    st.subheader("4. NLP Approach Comparison: Transformer (A) vs Keyword (B)")
    st.caption(
        "Approach A uses DistilBERT (transformer-based sentiment analysis). "
        "Approach B uses a curated keyword list. Below: Approach B scores on sample bios."
    )
    sample_bios = {
        "Taylor Swift": (
            "Taylor Alison Swift is an American singer-songwriter. An influential figure "
            "in popular culture, she is one of the best-selling music artists of all time."
        ),
        "Metallica": (
            "Metallica is an American heavy metal band. The band is credited with "
            "popularizing the genre and is one of the most celebrated rock acts globally."
        ),
        "Godsmack": (
            "Godsmack is an American rock band from Lawrence, Massachusetts, formed in 1995. "
            "Known for their heavy sound and energetic live performances."
        ),
        "Unknown Artist": (
            "A popular music artist known for live performances and chart-topping hits."
        ),
    }
    cmp_df = compare_approaches(
        list(sample_bios.values()),
        labels=list(sample_bios.keys()),
        use_transformer=False,
    )
    st.dataframe(cmp_df, use_container_width=True)

    st.markdown("---")

    # ── Model Comparison ─────────────────────────────────────────────────────
    st.subheader("5. Model Comparison: RF vs XGBoost (with / without NLP)")
    results_path = MODEL_DIR / "results.csv"

    if results_path.exists():
        results_df = pd.read_csv(results_path, index_col=0)
    else:
        st.info("Training models … this takes ~30 seconds.")
        out = get_trained_models()
        results_df = pd.DataFrame(out["results"]).T

    st.dataframe(
        results_df.style
        .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.4f}"})
        .background_gradient(cmap="RdYlGn", subset=["R2"])
        .background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE"]),
        use_container_width=True,
    )

    fig = px.bar(
        results_df.reset_index().rename(columns={"index": "Model"}),
        x="Model", y="RMSE", color="Model",
        title="RMSE Comparison Across Models (USD)",
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importances
    st.subheader("6. Feature Importances — XGBoost + NLP")
    try:
        model, encoders = get_model("XGB_nlp")
        X, _ = build_feature_matrix(df, include_nlp=True, fit_encoders=False, encoders=encoders)
        imp = get_feature_importances(model, X.columns.tolist())
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
        fig = px.bar(
            x=[v for _, v in top],
            y=[k for k, _ in top],
            orientation="h",
            title="Top 15 Feature Importances (XGBoost + NLP)",
            labels={"x": "Importance", "y": "Feature"},
            color=[v for _, v in top],
            color_continuous_scale="Teal",
        )
        fig.update_layout(showlegend=False, height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Train models first via the Price Predictor tab.")

    # Dataset preview
    st.subheader("7. Dataset Preview (Source 1 + Source 2 merged)")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(
        f"Total rows: {len(df):,} | Columns: {len(df.columns)} | "
        f"Sources: concerts.csv (GitHub) + wiki_bios_cache.csv (Wikipedia API)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🎵 Concert Ticket Price Predictor")
    st.markdown(
        "> Combines **ML on structured concert data** (Random Forest & XGBoost) "
        "with **NLP analysis of Wikipedia artist biographies** (DistilBERT & keyword heuristic) "
        "to predict minimum ticket prices in USD."
    )
    st.markdown(
        "**Data sources:** "
        "[Ticketmaster concert prices](https://github.com/ethanjaredlee/ticketmaster-price-ml) "
        "· [Wikipedia artist bios](https://en.wikipedia.org/api/rest_v1/)"
    )

    tab1, tab2 = st.tabs(["🎟️ Price Predictor", "📊 Model Insights"])

    with tab1:
        tab_predictor()
    with tab2:
        tab_insights()


if __name__ == "__main__":
    main()
