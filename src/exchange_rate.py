"""
exchange_rate.py — Data Source 3: Frankfurter Exchange Rate API.

Fetches the current USD → CHF exchange rate from api.frankfurter.app.
This is a free, open, no-API-key-required service backed by the
European Central Bank reference rates.

Integration:
  Used in the app UI to display predicted ticket prices in both USD
  (original training currency) and CHF (relevant for Swiss users).
  Rate is cached per session to avoid repeated network calls.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

FRANKFURTER_URL = "https://api.frankfurter.app/latest?from=USD&to=CHF"
_cached_rate: Optional[float] = None


def get_usd_to_chf(fallback: float = 0.90) -> float:
    """Fetch the current USD → CHF exchange rate.

    Parameters
    ----------
    fallback:
        Rate to use if the API is unreachable (default ~0.90, approximate 2025 rate).

    Returns
    -------
    float exchange rate (e.g. 0.892 means 1 USD = 0.892 CHF).
    """
    global _cached_rate
    if _cached_rate is not None:
        return _cached_rate

    req = urllib.request.Request(
        FRANKFURTER_URL,
        headers={"User-Agent": "ConcertPricePredictor/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        rate = float(data["rates"]["CHF"])
        _cached_rate = rate
        return rate
    except Exception:
        _cached_rate = fallback
        return fallback


def usd_to_chf(usd_amount: float, fallback: float = 0.90) -> float:
    """Convert a USD amount to CHF using the live exchange rate."""
    rate = get_usd_to_chf(fallback=fallback)
    return round(usd_amount * rate, 2)
