"""Historical backfill. Manual dispatch, chunked and resumable.

This is the critical-path job: without history there is no training set, and
the backfill cannot be parallelised away. It is written to survive interruption
-- responses are cached on disk per date chunk, and a re-run picks up where it
stopped rather than starting over.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.clients import open_meteo
from src.config import ARCHIVE_START_FALLBACK, CITY, ensure_dirs
from src.features.build import build_daily_features, merge_hourly
from src.features.schema import sanitise, save_schema, validate_daily
from src.store import hopsworks_store, local_store

log = logging.getLogger(__name__)


def resolve_start(explicit: str | None) -> date:
    """Prefer the probe's verified coverage over the configured fallback."""
    if explicit:
        return date.fromisoformat(explicit)
    coverage = local_store.read_json("coverage.json")
    if coverage.get("air_quality_start"):
        log.info("using probed archive start %s", coverage["air_quality_start"])
        return date.fromisoformat(coverage["air_quality_start"])
    log.warning("no coverage.json; falling back to %s. Run "
                "`python -m src.clients.open_meteo --probe` first.",
                ARCHIVE_START_FALLBACK)
    return date.fromisoformat(ARCHIVE_START_FALLBACK)


def run(start: str | None = None, end: str | None = None, chunk_days: int = 180,
        use_historical_forecast: bool = True) -> dict:
    ensure_dirs()
    start_date = resolve_start(start)
    end_date = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
    if start_date >= end_date:
        raise SystemExit(f"empty range: {start_date} -> {end_date}")

    chunks = list(open_meteo.date_chunks(start_date, end_date, chunk_days))
    log.info("backfilling %s -> %s in %d chunks", start_date, end_date, len(chunks))

    aq_parts: list[pd.DataFrame] = []
    wx_parts: list[pd.DataFrame] = []
    fc_parts: list[pd.DataFrame] = []
    failures: list[str] = []

    for i, (c_start, c_end) in enumerate(chunks, 1):
        log.info("chunk %d/%d: %s -> %s", i, len(chunks), c_start, c_end)
        try:
            aq_parts.append(open_meteo.fetch_air_quality(c_start, c_end))
            wx_parts.append(open_meteo.fetch_weather_archive(c_start, c_end))
        except open_meteo.OpenMeteoError as exc:
            log.error("chunk %s->%s failed: %s", c_start, c_end, exc)
            failures.append(f"{c_start}..{c_end}: {exc}")
            continue
        if use_historical_forecast:
            try:
                fc_parts.append(open_meteo.fetch_historical_forecast(c_start, c_end))
            except open_meteo.OpenMeteoError as exc:
                log.warning("historical forecast unavailable for %s->%s: %s",
                            c_start, c_end, exc)

    if not aq_parts:
        raise SystemExit("backfill produced no data")

    air_quality = pd.concat(aq_parts).sort_index()
    air_quality = air_quality[~air_quality.index.duplicated(keep="last")]
    weather = pd.concat(wx_parts).sort_index() if wx_parts else pd.DataFrame()
    if not weather.empty:
        weather = weather[~weather.index.duplicated(keep="last")]

    # The train/serve skew decision, made explicit and recorded.
    future_source = None
    skew_mode = "observed_weather_fallback"
    if fc_parts:
        forecast_weather = pd.concat(fc_parts).sort_index()
        forecast_weather = forecast_weather[~forecast_weather.index.duplicated(keep="last")]
        coverage = len(forecast_weather) / max(len(weather), 1)
        if coverage > 0.8:
            from src.features.build import aggregate_daily
            future_source = aggregate_daily(
                merge_hourly(air_quality.reindex(forecast_weather.index), forecast_weather),
                CITY)
            skew_mode = "historical_forecast"
        else:
            log.warning("historical forecast covers only %.0f%% of the range; "
                        "falling back to observed weather", coverage * 100)

    hourly = sanitise(merge_hourly(air_quality, weather))
    hourly["city_id"] = CITY.city_id
    local_store.write_hourly(hourly)
    hopsworks_store.write_hourly(hourly)

    daily = sanitise(build_daily_features(
        air_quality, weather, CITY, future_weather_source=future_source))
    report = validate_daily(daily)
    save_schema(daily)
    local_store.write_daily(daily)
    hopsworks_store.write_daily(daily)
    hopsworks_store.get_or_create_feature_view()

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "start": str(start_date), "end": str(end_date),
        "chunks": len(chunks), "failed_chunks": failures,
        "hourly_rows": int(len(hourly)), "daily_rows": int(len(daily)),
        "future_weather_source": skew_mode,
        "quality": report,
    }
    local_store.write_json("backfill_summary.json", summary)
    write_data_card(summary, daily)
    return summary


def write_data_card(summary: dict, daily: pd.DataFrame) -> None:
    """docs/DATA_CARD.md, generated from what the backfill actually saw."""
    from src.config import REPO_ROOT
    q = summary["quality"]
    lines = [
        "# Data Card", "",
        f"Generated by the backfill on {summary['ran_at']}.", "",
        "## Source", "",
        "- Pollutants and AQI: Open-Meteo Air Quality API (CAMS). No API key required.",
        "- Weather: Open-Meteo ERA5 archive and forecast API.",
        f"- Future-weather features derived from: **{summary['future_weather_source']}**",
        "", "## Coverage", "",
        f"- Range: {summary['start']} to {summary['end']}",
        f"- Hourly rows: {summary['hourly_rows']:,}",
        f"- Daily rows: {summary['daily_rows']:,}",
        f"- Missing days: {q.get('missing_days', 'n/a')}",
        f"- Days with fewer than 18 hourly observations: {q.get('thin_days', 'n/a')}",
        "", "## Distribution", "",
        f"- AQI daily mean, median: {q.get('aqi_mean_median', float('nan')):.1f}",
        f"- AQI daily mean, range: {q.get('aqi_mean_min', float('nan')):.1f} "
        f"to {q.get('aqi_mean_max', float('nan')):.1f}",
        "", "## Known limitations", "",
        "- CAMS values are a *model reanalysis*, not ground-station readings. They",
        "  will disagree with a physical monitor; the EDA quantifies by how much.",
        "- The AQI is recomputed here from concentrations using the EPA breakpoint",
        "  tables and the correct averaging windows, rather than taken from the",
        "  API's hourly convenience field. See src/features/aqi.py.",
    ]
    if summary["failed_chunks"]:
        lines += ["", "## Failed chunks", ""] + [f"- {f}" for f in summary["failed_chunks"]]
    (REPO_ROOT / "docs" / "DATA_CARD.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical backfill")
    parser.add_argument("--start", help="ISO date; defaults to the probed archive start")
    parser.add_argument("--end", help="ISO date; defaults to yesterday")
    parser.add_argument("--chunk-days", type=int, default=180)
    parser.add_argument("--no-historical-forecast", action="store_true",
                        help="use observed weather for the future-weather features")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.start, args.end, args.chunk_days,
                         not args.no_historical_forecast), indent=2, default=str))


if __name__ == "__main__":
    main()
