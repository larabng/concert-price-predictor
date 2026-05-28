"""
app.py — Concert Ticket Price Predictor (interactive web UI).

Three tabs:
  1. Price Predictor   — CV genre detection + ML price prediction + GPT explanation
  2. Budget Planner    — heatmap + artist suggestions for a given CHF budget
  3. Model Insights    — EDA charts, NLP comparison, model metrics, feature importances
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

try:
    import torch  # noqa: F401
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from data_loader import load_data
from model import MODEL_DIR, build_feature_matrix, get_feature_importances, load_model, train_models
from nlp_features import compare_approaches, enrich_with_nlp
from predict import predict_price
from llm_explanation import generate_llm_explanation
from cv_classifier import classify_genre
from exchange_rate import get_usd_to_chf, usd_to_chf

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
        "Combines **structured event data** (ML block), "
        "**Wikipedia artist biography** (NLP block), and "
        "**artist photo genre detection** (CV block) to estimate the minimum ticket price."
    )

    # ── CV Block: optional photo upload ──────────────────────────────────────
    with st.expander("📷 Step 1 (optional): Upload an artist photo for automatic genre detection"):
        st.caption("CLIP (ViT-B/32) classifies the image into a music genre, which pre-fills the genre field below.")
        uploaded_img = st.file_uploader("Artist or concert photo", type=["jpg", "jpeg", "png"], key="cv_upload")
        cv_genre, cv_conf = None, None
        if uploaded_img is not None:
            import io
            from PIL import Image as PILImage
            img_bytes = uploaded_img.read()          # read once into bytes
            img = PILImage.open(io.BytesIO(img_bytes))
            col_img, col_result = st.columns([1, 2])
            with col_img:
                st.image(img, width=200)
            with col_result:
                with st.spinner("CLIP is analysing the image…"):
                    cv_genre, cv_conf = classify_genre(img_bytes)  # reuse same bytes
                if cv_genre:
                    st.success(f"Detected genre: **{cv_genre}** ({cv_conf:.0%} confidence)")
                    st.caption("The genre field below has been pre-filled. You can change it if needed.")
                else:
                    st.warning("Genre detection unavailable — check HF_TOKEN secret or select genre manually.")

    with st.form("predictor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            artist = st.text_input("Artist / Band name", value="Taylor Swift")
            cv_default = GENRES.index(cv_genre) if cv_genre in GENRES else 0
            genre = st.selectbox("Genre", GENRES, index=cv_default,
                                 help="Pre-filled by CV block if a photo was uploaded above.")
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
            if TRANSFORMER_AVAILABLE:
                use_transformer = st.checkbox(
                    "Use DistilBERT for sentiment (Approach A, slower)",
                    value=False,
                    help="Unchecked = keyword heuristic (Approach B, fast).",
                )
            else:
                use_transformer = False
                st.caption("ℹ️ Approach A (DistilBERT) not available in this deployment — using keyword heuristic.")

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
        price_chf = usd_to_chf(price)
        rate = get_usd_to_chf()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Price (USD)", f"${price:.0f}")
        c2.metric("Predicted Price (CHF)", f"CHF {price_chf:.0f}",
                  help=f"Live rate: 1 USD = {rate:.4f} CHF (Source 3: Frankfurter API)")
        c3.metric("Bio Sentiment Score", f"{result['sentiment_score']:+.3f}")
        c4.metric("Bio Hype Score", f"{result['hype_score']:.3f}")

        st.markdown("---")
        st.subheader("Why this price?")

        llm_text = generate_llm_explanation(
            artist=artist,
            predicted_price=price,
            top_features=result["feature_importances"],
            bio=bio,
        )
        if llm_text:
            st.markdown(llm_text)
            with st.expander("Technical breakdown (ML feature importances)"):
                st.markdown(result["explanation"])
        else:
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
            color="genre", opacity=0.5,            labels={"score": "Artist Popularity Score", "minprice": "Price (USD)"},
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
                opacity=0.4,                title="Keyword Sentiment vs Price",
                labels={"sentiment_score": "Sentiment Score", "minprice": "Price (USD)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            fig = px.scatter(
                df[df["minprice"] <= 400], x="hype_score", y="minprice",
                opacity=0.4,                title="Hype Score vs Price",
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
# Tab 2 — Budget Planner
# ---------------------------------------------------------------------------

def tab_budget_planner():
    st.header("💰 Concert Budget Planner")
    st.markdown(
        "Enter your budget in CHF — the app scans **all genre × month combinations** "
        "across three city sizes and shows which concerts fit your wallet. "
        "Powered by the ML model (Source 1) and live exchange rate (Source 3)."
    )

    rate = get_usd_to_chf()

    col1, col2, col3 = st.columns(3)
    with col1:
        budget_chf = st.slider("Your budget (CHF)", min_value=20, max_value=500,
                               value=80, step=5)
    with col2:
        weekend = st.checkbox("Weekend events only", value=False)
    with col3:
        score = st.slider("Artist popularity (0 = unknown, 100 = superstar)",
                          min_value=65, max_value=100, value=80)

    # Generate prediction grid: genre × month × city_size
    try:
        model, encoders = get_model("XGB_nlp")
    except Exception:
        st.warning("Train models first via the Price Predictor tab.")
        return

    city_sizes = {
        "Small (~100K)": 100_000,
        "Large (~800K)": 800_000,
        "Mega (~3M)":  3_000_000,
    }
    months_short = {6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 4:"Apr", 5:"May"}

    rows = []
    for genre in GENRES[:-1]:  # skip "Other"
        for month, mname in months_short.items():
            for city_label, pop in city_sizes.items():
                rows.append({
                    "genre": genre, "month": month, "month_name": mname,
                    "pop": pop, "city_label": city_label,
                    "weekend": int(weekend), "score": float(score),
                    "sentiment_score": 0.4, "hype_score": 0.4,
                })

    grid_df = pd.DataFrame(rows)

    from model import build_feature_matrix as bfm, LOG_TARGET
    X, _ = bfm(grid_df, include_nlp=True, fit_encoders=False, encoders=encoders)
    expected = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[list(expected)]

    raw = model.predict(X)
    grid_df["pred_usd"] = np.expm1(raw) if LOG_TARGET else raw
    grid_df["pred_chf"] = (grid_df["pred_usd"] * rate).round(0).astype(int)
    grid_df["fits"] = grid_df["pred_chf"] <= budget_chf

    # ── Heatmap: Genre × Month for Large city ────────────────────────────────
    st.subheader(f"Price Heatmap — Large City (~800K), {'Weekend' if weekend else 'Weekday'}")
    st.caption(f"Budget: CHF {budget_chf} | Green = within budget | Red = over budget")

    large = grid_df[grid_df["city_label"] == "Large (~800K)"].copy()
    pivot = large.pivot(index="genre", columns="month_name", values="pred_chf")
    month_order = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    vmin, vmax = 20, min(400, grid_df["pred_chf"].quantile(0.95))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > budget_chf else "black"
            marker = "✓" if val <= budget_chf else f"{int(val)}"
            ax.text(j, i, marker, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if val <= budget_chf else "normal")

    plt.colorbar(im, ax=ax, label="Predicted min. price (CHF)")
    ax.set_title(f"Predicted Ticket Prices — ✓ = within your CHF {budget_chf} budget")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Best matches ─────────────────────────────────────────────────────────
    fits = grid_df[grid_df["fits"]].sort_values("pred_chf", ascending=False)
    st.subheader(f"Best Matches within CHF {budget_chf}")

    if fits.empty:
        st.error("No combinations fit your budget. Try increasing it.")
    else:
        summary = (
            fits.groupby(["genre", "city_label"])["pred_chf"]
            .mean().round(0).astype(int).reset_index()
            .rename(columns={"pred_chf": "Avg predicted price (CHF)",
                              "genre": "Genre", "city_label": "City size"})
            .sort_values("Avg predicted price (CHF)", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)

        # GPT recommendation
        top_combos = fits.nsmallest(3, "pred_chf")[["genre","month_name","city_label","pred_chf"]]
        combos_text = "; ".join(
            f"{r.genre} in {r.city_label} in {r.month_name} (~CHF {r.pred_chf})"
            for _, r in top_combos.iterrows()
        )

        if st.button("Get AI recommendation 🤖"):
            from llm_explanation import _get_openai_key
            key = _get_openai_key()
            if key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=key)
                    prompt = (
                        f"A concert fan has a budget of CHF {budget_chf} for a concert ticket. "
                        f"Based on ML predictions, these combinations fit their budget: {combos_text}. "
                        f"Write 2 friendly sentences recommending the best option and why."
                    )
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100, temperature=0.6,
                    )
                    st.info("💡 " + resp.choices[0].message.content.strip())
                except Exception as e:
                    st.warning(f"AI recommendation unavailable: {e}")
            else:
                st.warning("OpenAI key not configured.")

    # ── Artist suggestions from real dataset ─────────────────────────────────
    st.subheader(f"🎤 Artist Suggestions within CHF {budget_chf}")
    st.caption("Real artists from the training dataset whose predicted price fits your budget.")

    from predict import batch_predict
    real_df = get_enriched_data().copy()
    real_df["pred_chf_real"] = (batch_predict(real_df, "XGB_nlp")["predicted_price"] * rate).round(0)
    within = (
        real_df[real_df["pred_chf_real"] <= budget_chf]
        .sort_values("pred_chf_real", ascending=False)
        .drop_duplicates("artist")
        [["artist", "genre", "pred_chf_real", "score"]]
        .head(10)
        .rename(columns={"artist": "Artist", "genre": "Genre",
                          "pred_chf_real": "Est. min. price (CHF)",
                          "score": "Popularity score"})
        .reset_index(drop=True)
    )
    if within.empty:
        st.info("No known artists fit this budget. Try raising it slightly.")
    else:
        st.dataframe(within, use_container_width=True)

    # ── City comparison bar chart ─────────────────────────────────────────────
    st.subheader("Price vs City Size — July, Most Popular Genre per City")
    city_avg = grid_df[grid_df["month"] == 7].groupby("city_label")["pred_chf"].mean().reset_index()
    fig2 = px.bar(city_avg, x="city_label", y="pred_chf", color="pred_chf",
                  color_continuous_scale="RdYlGn_r",
                  labels={"pred_chf": "Avg predicted price (CHF)", "city_label": "City size"},
                  title="July concerts: average predicted price by city size")
    fig2.add_hline(y=budget_chf, line_dash="dash", line_color="green",
                   annotation_text=f"Your budget: CHF {budget_chf}")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)


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

    tab1, tab2, tab3 = st.tabs(["🎟️ Price Predictor", "💰 Budget Planner", "📊 Model Insights"])

    with tab1:
        tab_predictor()
    with tab2:
        tab_budget_planner()
    with tab3:
        tab_insights()


if __name__ == "__main__":
    main()
