"""Parquet mirror of the feature store.

Written on every run alongside the Hopsworks write. If the managed store is
unreachable -- free-tier limits, an outage, a rotated key -- the pipelines keep
running against these files and the system degrades instead of stopping. It is
also what makes the whole project runnable offline for development.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

log = logging.getLogger(__name__)

HOURLY_PATH = DATA_DIR / "hourly_raw.parquet"
DAILY_PATH = DATA_DIR / "daily_features.parquet"
PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"
METRICS_PATH = DATA_DIR / "experiments.csv"


def _upsert(new: pd.DataFrame, path: Path, index_name: str) -> pd.DataFrame:
    """Idempotent write. Re-running any pipeline overwrites rows rather than
    duplicating them, which is what lets the hourly job re-fetch an overlapping
    window to heal gaps left by skipped runs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if new.empty:
        return read_parquet(path)
    combined = new
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new])
        except Exception as exc:                      # pragma: no cover
            log.warning("could not read %s (%s); overwriting", path, exc)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = index_name
    combined.to_parquet(path)
    log.info("wrote %d rows to %s", len(combined), path.name)
    return combined


def write_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return _upsert(df, HOURLY_PATH, "ts_utc")


def write_daily(df: pd.DataFrame) -> pd.DataFrame:
    return _upsert(df, DAILY_PATH, "date")


def write_predictions(df: pd.DataFrame) -> pd.DataFrame:
    return _upsert(df, PREDICTIONS_PATH, "target_date")


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:                          # pragma: no cover
        log.warning("could not read %s: %s", path, exc)
        return pd.DataFrame()


def read_hourly() -> pd.DataFrame:
    return read_parquet(HOURLY_PATH)


def read_daily() -> pd.DataFrame:
    return read_parquet(DAILY_PATH)


def read_predictions() -> pd.DataFrame:
    return read_parquet(PREDICTIONS_PATH)


def append_experiment(rows: list[dict]) -> None:
    """The experiment log. Every model evaluation appends here; the report and
    the dashboard leaderboard are both built from this file."""
    if not rows:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if METRICS_PATH.exists():
        frame = pd.concat([pd.read_csv(METRICS_PATH), frame], ignore_index=True)
    frame.to_csv(METRICS_PATH, index=False)


def read_experiments() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


def write_json(name: str, payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


def read_json(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
