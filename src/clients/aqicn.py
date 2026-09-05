"""AQICN (WAQI) client -- ground-station cross-check only.

Open-Meteo serves CAMS, which is a *model* reanalysis. AQICN serves readings
from physical monitoring stations. They will disagree. The EDA quantifies the
disagreement and the data card records it, rather than quietly assuming the
modelled values are ground truth.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from src.config import AQICN_TOKEN, CITY, REQUEST_TIMEOUT, City

log = logging.getLogger(__name__)
BASE = "https://api.waqi.info"


def current_station_reading(city: City = CITY, token: str | None = None) -> dict[str, Any] | None:
    """Nearest-station current AQI, or None when no token is configured."""
    token = token or AQICN_TOKEN
    if not token:
        log.info("AQICN_TOKEN not set; skipping ground-station cross-check")
        return None
    url = f"{BASE}/feed/geo:{city.latitude};{city.longitude}/"
    try:
        resp = requests.get(url, params={"token": token}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("AQICN request failed: %s", exc)
        return None
    if payload.get("status") != "ok":
        log.warning("AQICN returned status=%s", payload.get("status"))
        return None
    data = payload["data"]
    return {
        "aqi": data.get("aqi"),
        "station": (data.get("city") or {}).get("name"),
        "dominant_pollutant": data.get("dominentpol"),
        "time": (data.get("time") or {}).get("iso"),
        "iaqi": {k: v.get("v") for k, v in (data.get("iaqi") or {}).items()},
    }
