"""
cv_classifier.py — Computer Vision block: CLIP-based music genre classification.

Uses openai/clip-vit-base-patch32 via the HuggingFace Inference API (zero-shot)
to classify an artist/concert photo into one of 8 music genres.

Integration with other blocks:
  Output (predicted genre) → replaces the genre dropdown in the ML Price Predictor
  This means CV block output directly feeds into the ML block as an input feature.

No local model required — runs via HF Inference API using the HF_TOKEN secret.
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
from typing import Optional

from PIL import Image

# Maps our dataset genre labels to descriptive CLIP text prompts
GENRE_LABELS = [
    "Rock", "Hip-Hop/Rap", "Country", "R&B",
    "Metal", "Pop", "Dance/Electronic", "Other",
]

GENRE_PROMPTS = {
    "Rock":             "a rock music band or artist performing on stage with electric guitars",
    "Hip-Hop/Rap":      "a hip-hop rapper or rap music artist performing",
    "Country":          "a country music singer or band with acoustic guitar and cowboy hat",
    "R&B":              "an R&B or soul music singer performing",
    "Metal":            "a heavy metal or hard rock band performing loudly on stage",
    "Pop":              "a pop music star or mainstream pop singer performing",
    "Dance/Electronic": "a DJ or electronic dance music artist at a turntable or festival",
    "Other":            "a musician or music artist performing on stage",
}


def preprocess_image(image_input) -> bytes:
    """Convert uploaded image to bytes for the HF API.

    Accepts a PIL Image, file-like object, or raw bytes.
    Resizes to 224x224 and converts to RGB.
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input)

    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def classify_genre(
    image_input,
    token: Optional[str] = None,
) -> tuple[str | None, float | None]:
    """Classify a concert/artist image into a music genre using CLIP.

    Parameters
    ----------
    image_input:
        PIL Image, file-like object, or raw bytes.
    token:
        HuggingFace API token. Auto-read from HF_TOKEN env var if not provided.

    Returns
    -------
    (genre, confidence) tuple, or (None, None) if unavailable.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            return None, None
    if not token:
        return None, None

    try:
        from huggingface_hub import InferenceClient
        image_bytes = preprocess_image(image_input)
        client = InferenceClient(token=token)
        results = client.zero_shot_image_classification(
            image=image_bytes,
            candidate_labels=list(GENRE_PROMPTS.values()),
            model="openai/clip-vit-base-patch32",
        )
        # Map back from prompt text to genre label
        prompt_to_genre = {v: k for k, v in GENRE_PROMPTS.items()}
        if results:
            top = results[0]
            # huggingface_hub returns objects with .label/.score OR dicts
            label = top.label if hasattr(top, "label") else top.get("label", "")
            score = top.score if hasattr(top, "score") else top.get("score", 0.0)
            genre = prompt_to_genre.get(label, "Other")
            confidence = round(float(score), 4)
            return genre, confidence
    return None, None


def fetch_artist_thumbnail(artist: str) -> Optional[bytes]:
    """Fetch the Wikipedia thumbnail image for an artist (for evaluation).

    Parameters
    ----------
    artist:
        Artist name as used in the dataset.

    Returns
    -------
    Raw image bytes, or None if no thumbnail is found.
    """
    import json
    import urllib.parse

    overrides = {
        "Maroon":           "Maroon_5",
        "P!nk":             "Pink_(singer)",
        "Panic! At The Disco": "Panic!_at_the_Disco",
        "Ty Dolla $ign":    "Ty_Dolla_Sign",
    }
    title = overrides.get(artist, artist.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ConcertPricePredictor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        thumb_url = data.get("thumbnail", {}).get("source")
        if not thumb_url:
            return None
        img_req = urllib.request.Request(thumb_url, headers={"User-Agent": "ConcertPricePredictor/1.0"})
        with urllib.request.urlopen(img_req, timeout=10) as r:
            return r.read()
    except Exception:
        return None
