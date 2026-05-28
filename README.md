---
title: Concert Ticket Price Predictor
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Concert Ticket Price Predictor

A machine learning project combining **structured data ML** (Random Forest & XGBoost)
with **NLP analysis of Wikipedia artist biographies** (DistilBERT & keyword heuristic)
to predict minimum concert ticket prices in USD.

Built as part of the AI Applications course (FS26).

---

## Two Real Data Sources

| # | Source | Type | Size | Role |
|---|---|---|---|---|
| 1 | [Ticketmaster price dataset](https://github.com/ethanjaredlee/ticketmaster-price-ml) | CSV (structured) | 1,198 rows | Concert features + real ticket prices |
| 2 | [Wikipedia artist bios](https://en.wikipedia.org/api/rest_v1/) | Text (REST API) | 87 artist summaries | NLP sentiment & hype features |

---

## Project Structure

```
kiprojekt/
├── data/
│   ├── concerts.csv             ← Source 1: real concert prices (committed)
│   ├── wiki_bios_cache.csv      ← Source 2: Wikipedia artist bios (committed)
│   └── models/                  ← trained models (generated, not committed)
├── notebooks/
│   ├── 01_eda.ipynb             ← EDA on real concert data
│   ├── 02_nlp_preprocessing.ipynb  ← NLP approach comparison (A vs B)
│   └── 03_modeling.ipynb        ← Model training & evaluation
├── src/
│   ├── data_loader.py           ← load + merge both data sources
│   ├── nlp_features.py          ← Approach A (DistilBERT) + Approach B (keyword)
│   ├── model.py                 ← RF & XGBoost training pipeline
│   └── predict.py               ← single-event inference
├── app/
│   └── app.py         ← interactive web UI
├── docs/
│   └── documentation.md         ← full project documentation
├── .streamlit/config.toml
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-generate data

The data files are already committed. Skip this unless you want to re-fetch.

```bash
python src/data_loader.py   # downloads concerts.csv + wiki_bios_cache.csv
```

### 3. Train models

```bash
python src/model.py         # trains RF_base, RF_nlp, XGB_base, XGB_nlp
```

### 4. Run the app

```bash
streamlit run app/app.py
```

Models are auto-trained on first launch if absent.

### 5. Run notebooks

```bash
cd notebooks
jupyter notebook
```

Run in order: `01_eda.ipynb` → `02_nlp_preprocessing.ipynb` → `03_modeling.ipynb`

---

## AI Blocks

### Block 1 — ML on Structured Data

| Feature | Description |
|---|---|
| `score` | Artist popularity score (0–100) |
| `pop` | City population (market size proxy) |
| `weekend` | Is the event on a weekend (0/1) |
| `month` | Month of event (1–12) |
| `genre_*` | Genre one-hot (8 categories) |
| `sentiment_score` | NLP sentiment from Wikipedia bio |
| `hype_score` | NLP prestige keyword density |

**Models compared:** RF (200 trees) vs XGBoost (300 estimators) × base/NLP = 4 models  
**Target:** `minprice` (USD, real ticket prices)

### Block 2 — NLP

- **Text data:** Wikipedia artist biographies (Source 2, fetched via REST API)
- **Approach A:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) — transformer sentiment
- **Approach B:** Keyword heuristic — count of 20 prestige/positive terms, normalised to [0, 1]
- **Comparison:** Approach B selected (more score variation, faster, no GPU needed)
- **Integration:** Both NLP scores become numeric features for the ML model

---

## Results (example run)

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| RF_base  | 92.1 | 18.8 | 0.511 |
| RF_nlp   | 92.5 | 18.3 | 0.508 |
| XGB_base | 90.8 | 18.1 | 0.526 |
| **XGB_nlp** | 91.6 | **18.1** | 0.517 |

> Note: RMSE is inflated by rare VIP/premium tickets ($500–$2,999). MAE of ~$18 is the typical absolute prediction error. Model trained on log(price) for better handling of the skewed price distribution.

---

## Deployment

Deployed at: **https://huggingface.co/spaces/banlar01/concert-price-predictor**

To deploy yourself:
1. Push repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app
3. Select repo + `app/app.py`
4. Deploy (auto-trains on first run, ~30–60 s)

---

## Dependencies

- `pandas`, `numpy`, `scikit-learn`, `xgboost` — data and modeling
- `transformers`, `torch` — DistilBERT (Approach A, optional)
- `streamlit`, `plotly` — interactive UI
- `matplotlib`, `seaborn` — EDA visualisations
- `joblib` — model persistence
- `tqdm` — progress bars

---

## Reproducibility

All random seeds fixed at `42`. Data files committed to repo. Run:

```bash
python src/model.py   # reproduces all metrics deterministically
```
