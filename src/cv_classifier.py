"""
cv_classifier.py — Computer Vision block: CLIP-based music genre classification.

Uses openai/clip-vit-base-patch32 loaded locally via transformers (zero-shot).
No HF Inference API or token required — model downloads on first run (~350 MB).

Integration: predicted genre pre-fills the genre dropdown in the ML Price Predictor.
"""

from __future__ import annotations

import io
import os
import urllib.request
from functools import lru_cache
from typing import Optional

from PIL import Image

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


@lru_cache(maxsize=1)
def _load_clip():
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def preprocess_image(image_input) -> Image.Image:
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input)
    return img.convert("RGB")


def classify_genre(
    image_input,
    token: Optional[str] = None,
) -> tuple[str | None, float | None]:
    """Classify a concert/artist image into a music genre using local CLIP.

    Returns (genre, confidence) or raises on error.
    token parameter kept for API compatibility but not used.
    """
    import torch

    model, processor = _load_clip()
    img = preprocess_image(image_input)
    prompts = list(GENRE_PROMPTS.values())

    inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    top_idx = int(probs.argmax())
    prompt_to_genre = {v: k for k, v in GENRE_PROMPTS.items()}
    genre = prompt_to_genre[prompts[top_idx]]
    confidence = round(float(probs[top_idx]), 4)
    return genre, confidence


def fetch_artist_thumbnail(artist: str) -> Optional[bytes]:
    """Fetch the Wikipedia thumbnail image for an artist (for evaluation)."""
    import json
    import urllib.parse

    overrides = {
        "Maroon":              "Maroon_5",
        "P!nk":                "Pink_(singer)",
        "Panic! At The Disco": "Panic!_at_the_Disco",
        "Ty Dolla $ign":       "Ty_Dolla_Sign",
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
