"""
data_loader.py — Loads the two real data sources and merges them.

Data Source 1: Real concert ticket price data (1,198 rows)
  File: data/concerts.csv
  Origin: ethanjaredlee/ticketmaster-price-ml (GitHub)
  Columns: city, artist, venue, weekend, pop, month, score, genre, minprice

Data Source 2: Wikipedia artist biographies (87 unique artists)
  File: data/wiki_bios_cache.csv
  Origin: Wikipedia REST API (en.wikipedia.org/api/rest_v1/page/summary)
  Columns: artist, bio, wiki_found

The two sources are merged on 'artist'. The resulting DataFrame is used
by the NLP and ML pipelines.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONCERTS_CSV = DATA_DIR / "concerts.csv"
WIKI_CACHE_CSV = DATA_DIR / "wiki_bios_cache.csv"
CONCERTS_URL = (
    "https://raw.githubusercontent.com/ethanjaredlee/ticketmaster-price-ml/master/data.csv"
)

DEFAULT_BIO = (
    "A popular music artist known for live performances and chart-topping hits, "
    "with a loyal fanbase and a history of sold-out shows."
)

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_concerts(dest: Path) -> None:
    """Download concerts.csv from GitHub if not present locally."""
    print("Downloading concert data from GitHub …")
    req = urllib.request.Request(
        CONCERTS_URL, headers={"User-Agent": "ConcertPricePredictor/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    dest.parent.mkdir(parents=True, exist_ok=True)
    import io
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {dest}")


def _fetch_wiki_bio(artist: str, override_title: str | None = None) -> tuple[str, bool]:
    """Fetch the Wikipedia summary paragraph for an artist."""
    title = override_title or artist
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "ConcertPricePredictor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        extract = data.get("extract", "")
        if extract and len(extract) > 50:
            sentences = extract.split(". ")
            bio = ". ".join(sentences[:3]) + ("." if len(sentences) > 3 else "")
            return bio, True
    except Exception:
        pass
    return DEFAULT_BIO, False


# Maps dataset artist name → correct Wikipedia page title
_WIKI_TITLE_OVERRIDES: dict[str, str] = {
    "Maroon":               "Maroon 5",
    "Janelle Monae":        "Janelle Monáe",
    "P!nk":                 "Pink (singer)",
    "Panic! At The Disco":  "Panic! at the Disco",
    "Ty Dolla $ign":        "Ty Dolla Sign",
    "NF":                   "NF (rapper)",
    "Grey":                 "Grey (music duo)",
    "Savage":               "Savage (rapper)",
    "G-Eazy":               "G-Eazy",
    "Drake":                "Drake (rapper)",
    "Halsey":               "Halsey (singer)",
    "Jordan Davis":         "Jordan Davis (singer)",
    "Khalid":               "Khalid (singer)",
    "Logic":                "Logic (rapper)",
    "Sam Hunt":             "Sam Hunt (singer)",
    "Sam Smith":            "Sam Smith (singer)",
    "SZA":                  "SZA (singer)",
    "Travis Scott":         "Travis Scott (rapper)",
    "Zedd":                 "Zedd (musician)",
    "Brett Young":          "Brett Young (singer)",
    "Bazzi":                "Bazzi (singer)",
}


def build_wiki_cache(artists: list[str], dest: Path) -> None:
    """Fetch Wikipedia bios for all artists and save to dest CSV."""
    print(f"Fetching Wikipedia bios for {len(artists)} artists …")
    records = []
    for artist in artists:
        override = _WIKI_TITLE_OVERRIDES.get(artist)
        bio, found = _fetch_wiki_bio(artist, override)
        records.append({"artist": artist, "bio": bio, "wiki_found": found})
        time.sleep(0.3)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artist", "bio", "wiki_found"])
        writer.writeheader()
        writer.writerows(records)
    found_n = sum(1 for r in records if r["wiki_found"])
    print(f"Wikipedia cache saved → {dest}  ({found_n}/{len(artists)} found)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    concerts_path: Path | None = None,
    wiki_path: Path | None = None,
) -> pd.DataFrame:
    """Load and merge the two data sources.

    Parameters
    ----------
    concerts_path:
        Path to concerts.csv. Downloads from GitHub if absent.
    wiki_path:
        Path to wiki_bios_cache.csv. Fetches from Wikipedia API if absent.

    Returns
    -------
    pd.DataFrame with columns:
        city, artist, venue, weekend, pop, month, score, genre, minprice, bio, wiki_found
    """
    if concerts_path is None:
        concerts_path = CONCERTS_CSV
    if wiki_path is None:
        wiki_path = WIKI_CACHE_CSV

    concerts_path = Path(concerts_path)
    wiki_path = Path(wiki_path)

    # ── Source 1: Concert structured data ────────────────────────────────────
    if not concerts_path.exists():
        _download_concerts(concerts_path)

    df = pd.read_csv(concerts_path, encoding="utf-8")
    # Use to_numeric to handle ArrowString backend in pandas 2.x
    for col in ("weekend", "pop", "month"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    for col in ("score", "minprice"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Source 2: Wikipedia artist biographies ───────────────────────────────
    if not wiki_path.exists():
        artists = sorted(df["artist"].unique().tolist())
        build_wiki_cache(artists, wiki_path)

    wiki_df = pd.read_csv(wiki_path, encoding="utf-8")
    wiki_df = wiki_df[["artist", "bio", "wiki_found"]].drop_duplicates("artist")

    # Merge: left join so every concert row keeps its data
    df = df.merge(wiki_df, on="artist", how="left")
    df["bio"] = df["bio"].fillna(DEFAULT_BIO)
    df["wiki_found"] = df["wiki_found"].fillna(False)

    return df


# Alias kept for notebook imports
load_or_generate = load_data


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Price stats:\n{df['minprice'].describe().round(2)}")
    print(f"\nGenre counts:\n{df['genre'].value_counts()}")
