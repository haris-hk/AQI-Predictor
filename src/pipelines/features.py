"""Feature pipeline. Runs hourly in GitHub Actions.

Deliberately fetches an overlapping window rather than just the newest hour.
Scheduled workflows on GitHub Actions are best-effort: they drift by minutes and
are sometimes skipped entirely under platform load. A pipeline that assumes the
previous run happened will accumulate silent holes. This one re-fetches the last
few days on every run and upserts, so any gap left by a missed run is healed by
the next one without intervention.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src.clients import open_meteo
from src.clients.aqicn import current_station_reading
from src.config import CITY, HOURLY_LOOKBACK_HOURS, ensure_dirs
from src.features.build import build_daily_features, merge_hourly
from src.features.schema import sanitise, save_schema, validate_daily
from src.store import hopsworks_store, local_store

log = logging.getLogger(__name__)


def run(lookback_hours: int = HOURLY_LOOKBACK_HOURS, cross_check: bool = True) -> dict:
    ensure_dirs()
    past_days = max(2, lookback_hours // 24 + 1)

    air_quality = open_meteo.fetch_air_quality_recent(past_days=past_days, forecast_days=5)
    weather = open_meteo.fetch_weather_forecast(past_days=past_days, forecast_days=7)
    if air_quality.empty:
        raise SystemExit("air quality fetch returned nothing")

    hourly = merge_hourly(air_quality, weather)
    hourly = sanitise(hourly)
    hourly["city_id"] = CITY.city_id
    local_store.write_hourly(hourly)
    hopsworks_ok = hopsworks_store.write_hourly(hourly)

    # Rebuild daily features from the full local history, not just this window:
    # lag-14 and the 30-day rolling statistics need the context.
    history = local_store.read_hourly()
    daily = build_daily_features(
        history[[c for c in history.columns if c in air_quality.columns]],
        history[[c for c in history.columns if c in weather.columns]],
        CITY,
    )
    daily = sanitise(daily)
    report = validate_daily(daily)
    save_schema(daily)

    local_store.write_daily(daily)
    hopsworks_daily_ok = hopsworks_store.write_daily(daily)

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "hourly_rows_fetched": int(len(hourly)),
        "hourly_rows_total": int(len(history)),
        "daily_rows_total": int(len(daily)),
        "latest_hour": str(hourly.index.max()) if len(hourly) else None,
        "latest_day": str(daily.index.max().date()) if len(daily) else None,
        "hopsworks_hourly": hopsworks_ok,
        "hopsworks_daily": hopsworks_daily_ok,
        "quality": report,
    }

    if cross_check:
        station = current_station_reading()
        if station:
            summary["ground_station"] = station
            if len(daily) and station.get("aqi") is not None:
                summary["model_vs_station_delta"] = float(
                    daily["aqi_mean"].iloc[-1] - float(station["aqi"]))

    local_store.write_json("last_feature_run.json", summary)
    log.info("feature run complete: %s hourly rows, %s daily rows",
             summary["hourly_rows_total"], summary["daily_rows_total"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly AQI feature pipeline")
    parser.add_argument("--lookback-hours", type=int, default=HOURLY_LOOKBACK_HOURS)
    parser.add_argument("--no-cross-check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.lookback_hours, not args.no_cross_check), indent=2, default=str))


if __name__ == "__main__":
    main()
