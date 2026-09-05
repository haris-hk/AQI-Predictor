"""The feature contract.

Enforced on every write. A schema drift between the backfill and the hourly
pipeline is the kind of fault that produces a model quietly trained on a column
that no longer exists at serving time, so it is checked rather than assumed.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import DATA_DIR

log = logging.getLogger(__name__)
SCHEMA_FILE = DATA_DIR / "feature_schema.json"

REQUIRED_DAILY = ["city_id", "aqi_mean", "aqi_max", "hours_observed"]
# A day with fewer than this many hourly observations is not a day.
MIN_HOURS_PER_DAY = 18


def sanitise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, replace infinities and drop all-null columns."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col in ("city_id", "dominant_pollutant"):
            out[col] = out[col].astype("string")
        elif not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    empty = [c for c in out.columns if out[c].isna().all()]
    if empty:
        log.warning("dropping %d all-null columns: %s", len(empty), empty[:5])
        out = out.drop(columns=empty)
    return out


def validate_daily(df: pd.DataFrame, strict: bool = False) -> dict:
    """Return a data-quality report; raise on a hard failure when strict."""
    report: dict = {"rows": int(len(df)), "columns": int(df.shape[1]), "problems": []}
    if df.empty:
        report["problems"].append("frame is empty")
        if strict:
            raise ValueError("empty feature frame")
        return report

    missing = [c for c in REQUIRED_DAILY if c not in df.columns]
    if missing:
        report["problems"].append(f"missing required columns: {missing}")

    if "hours_observed" in df.columns:
        thin = int((df["hours_observed"] < MIN_HOURS_PER_DAY).sum())
        report["thin_days"] = thin
        if thin:
            report["problems"].append(f"{thin} days with < {MIN_HOURS_PER_DAY} hourly observations")

    if isinstance(df.index, pd.DatetimeIndex):
        report["date_min"] = str(df.index.min().date())
        report["date_max"] = str(df.index.max().date())
        expected = pd.date_range(df.index.min(), df.index.max(), freq="D")
        gaps = expected.difference(df.index)
        report["missing_days"] = int(len(gaps))
        if len(gaps):
            report["missing_day_examples"] = [str(d.date()) for d in gaps[:10]]
        dupes = int(df.index.duplicated().sum())
        report["duplicate_dates"] = dupes
        if dupes:
            report["problems"].append(f"{dupes} duplicate dates")

    if "aqi_mean" in df.columns:
        aqi = df["aqi_mean"].dropna()
        if len(aqi):
            report["aqi_mean_min"] = float(aqi.min())
            report["aqi_mean_max"] = float(aqi.max())
            report["aqi_mean_median"] = float(aqi.median())
            out_of_range = int(((aqi < 0) | (aqi > 500)).sum())
            if out_of_range:
                report["problems"].append(f"{out_of_range} AQI values outside [0, 500]")

    null_share = df.isna().mean()
    heavy = null_share[null_share > 0.5].index.tolist()
    report["columns_over_50pct_null"] = heavy[:20]

    if strict and report["problems"]:
        raise ValueError(f"feature validation failed: {report['problems']}")
    return report


def save_schema(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_FILE.write_text(json.dumps(
        {c: str(df[c].dtype) for c in df.columns}, indent=2, sort_keys=True))


def check_against_saved_schema(df: pd.DataFrame) -> list[str]:
    """Columns the saved schema has that this frame does not. Empty is good."""
    if not SCHEMA_FILE.exists():
        return []
    saved = json.loads(SCHEMA_FILE.read_text())
    return sorted(set(saved) - set(df.columns))
