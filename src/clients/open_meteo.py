"""Open-Meteo client.

Chosen as the primary source because it is the only free option that serves
*both* a multi-year history and a 7-day forecast without an API key. History is
what makes the training set possible; the forecast is what makes the 3-day
prediction possible. See docs/PROJECT_PLAN.md section 2.

Three endpoints matter:
  air-quality           pollutants + US AQI, archive and forecast
  archive (ERA5)        observed weather, 1940 onward
  historical-forecast   *past forecast runs* -- weather as it was predicted at
                        the time, which is what removes the train/serve skew
                        described in PROJECT_PLAN.md section 3.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from src.config import CITY, DATA_DIR, MAX_RETRIES, REQUEST_TIMEOUT, City

log = logging.getLogger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

AIR_QUALITY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "aerosol_optical_depth", "dust", "uv_index", "us_aqi",
]

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "boundary_layer_height",
]

CACHE_DIR = DATA_DIR / "http_cache"


class OpenMeteoError(RuntimeError):
    pass


def _cache_path(url: str, params: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        (url + json.dumps(params, sort_keys=True, default=str)).encode()
    ).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def _get(url: str, params: dict[str, Any], use_cache: bool = True) -> dict:
    """GET with exponential backoff and an on-disk response cache.

    The cache exists for the backfill: a 4-year pull is chunked into dozens of
    requests and a mid-run failure must not restart from zero. Only complete
    date ranges (no `past_days`/`forecast_days`) are cached, because those are
    the only immutable ones.
    """
    cacheable = use_cache and "start_date" in params
    path = _cache_path(url, params)
    if cacheable and path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("rate limited, sleeping %ss", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload and payload.get("error"):
                raise OpenMeteoError(payload.get("reason", "unknown API error"))
            if cacheable:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
            return payload
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            wait = 2 ** attempt
            log.warning("request failed (%s), retry %d/%d in %ss",
                        exc, attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise OpenMeteoError(f"failed after {MAX_RETRIES} attempts: {last_error}")


def _hourly_frame(payload: dict, prefix: str = "") -> pd.DataFrame:
    """Convert an Open-Meteo `hourly` block into a tidy UTC-indexed frame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"time": "ts_utc"}).set_index("ts_utc")
    df = df.apply(pd.to_numeric, errors="coerce")
    if prefix:
        df = df.add_prefix(prefix)
    return df.sort_index()


def _base_params(city: City) -> dict[str, Any]:
    return {"latitude": city.latitude, "longitude": city.longitude, "timezone": "UTC"}


# --------------------------------------------------------------------- fetchers
def fetch_air_quality(start_date: str | date, end_date: str | date,
                      city: City = CITY) -> pd.DataFrame:
    params = _base_params(city) | {
        "hourly": ",".join(AIR_QUALITY_VARS),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "domains": "cams_global",
    }
    return _hourly_frame(_get(AIR_QUALITY_URL, params))


def fetch_air_quality_recent(past_days: int = 7, forecast_days: int = 5,
                             city: City = CITY) -> pd.DataFrame:
    params = _base_params(city) | {
        "hourly": ",".join(AIR_QUALITY_VARS),
        "past_days": min(past_days, 92),
        "forecast_days": min(forecast_days, 7),
    }
    return _hourly_frame(_get(AIR_QUALITY_URL, params, use_cache=False))


def fetch_weather_archive(start_date: str | date, end_date: str | date,
                          city: City = CITY) -> pd.DataFrame:
    params = _base_params(city) | {
        "hourly": ",".join(WEATHER_VARS),
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    return _hourly_frame(_get(ARCHIVE_URL, params))


def fetch_weather_forecast(past_days: int = 7, forecast_days: int = 7,
                           city: City = CITY) -> pd.DataFrame:
    params = _base_params(city) | {
        "hourly": ",".join(WEATHER_VARS),
        "past_days": min(past_days, 92),
        "forecast_days": min(forecast_days, 16),
    }
    return _hourly_frame(_get(FORECAST_URL, params, use_cache=False))


def fetch_historical_forecast(start_date: str | date, end_date: str | date,
                              city: City = CITY) -> pd.DataFrame:
    """Weather as it was *forecast* at the time, not as it was observed.

    This is the honest training signal: at prediction time we only ever have a
    forecast, so the training features should contain forecast-quality values
    too. Falls back to the ERA5 archive at the caller's discretion.
    """
    params = _base_params(city) | {
        "hourly": ",".join(WEATHER_VARS),
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    return _hourly_frame(_get(HISTORICAL_FORECAST_URL, params))


# ------------------------------------------------------------------ date chunks
def date_chunks(start: date, end: date, days: int = 180) -> Iterable[tuple[date, date]]:
    """Yield inclusive [start, end] windows. Backfill is chunked and resumable."""
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


# ------------------------------------------------------------------------ probe
def probe_coverage(city: City = CITY, write: bool = True) -> dict[str, Any]:
    """Find the true earliest available air-quality date for these coordinates.

    PROJECT_PLAN.md section 2 flags this as a day-1 task: the global CAMS
    archive and the European reanalysis start on different dates, and every
    downstream sizing decision depends on the real number rather than an
    assumption. Binary search over candidate years, then months.
    """
    today = date.today()
    result: dict[str, Any] = {
        "city_id": city.city_id, "probed_at": datetime.utcnow().isoformat() + "Z",
    }

    def available(d: date) -> bool:
        try:
            df = fetch_air_quality(d, d + timedelta(days=1), city=city)
        except OpenMeteoError:
            return False
        return not df.empty and df["pm2_5"].notna().any()

    lo, hi = date(2012, 1, 1), today - timedelta(days=10)
    if not available(hi):
        result["error"] = "no data even for recent dates; check coordinates or API status"
        return result
    if available(lo):
        earliest = lo
    else:
        while (hi - lo).days > 20:
            mid = lo + (hi - lo) / 2
            mid = date(mid.year, mid.month, mid.day)
            if available(mid):
                hi = mid
            else:
                lo = mid
        earliest = hi

    result["air_quality_start"] = earliest.isoformat()
    result["days_available"] = (today - earliest).days

    for name, fn in (("weather_archive", fetch_weather_archive),
                     ("historical_forecast", fetch_historical_forecast)):
        try:
            probe_from = max(earliest, date(2022, 1, 1))
            df = fn(probe_from, probe_from + timedelta(days=1), city=city)
            result[f"{name}_ok"] = bool(not df.empty)
            result[f"{name}_from"] = probe_from.isoformat()
        except OpenMeteoError as exc:
            result[f"{name}_ok"] = False
            result[f"{name}_error"] = str(exc)

    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "coverage.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-Meteo client utilities")
    parser.add_argument("--probe", action="store_true",
                        help="find the true archive start date and write data/coverage.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.probe:
        print(json.dumps(probe_coverage(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
