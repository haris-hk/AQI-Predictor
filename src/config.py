"""Single source of truth for the whole project.

Every module reads its settings from here. Nothing hardcodes a coordinate,
a horizon or a threshold anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("AQI_DATA_DIR", REPO_ROOT / "data"))
ARTIFACT_DIR = DATA_DIR / "artifacts"


@dataclass(frozen=True)
class City:
    city_id: str
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str


KARACHI = City(
    city_id="karachi_pk",
    name="Karachi",
    country="Pakistan",
    latitude=24.8607,
    longitude=67.0011,
    timezone="Asia/Karachi",
)

CITY = KARACHI

# ---------------------------------------------------------------- forecasting
HORIZONS = (1, 2, 3)          # days ahead
TARGETS = ("aqi_mean", "aqi_max")

def target_col(target: str, h: int) -> str:
    return f"target_{target}_h{h}"

TARGET_COLUMNS = tuple(target_col(t, h) for t in TARGETS for h in HORIZONS)

# ------------------------------------------------------------------- validation
# Chronological holdout, touched exactly once at the very end.
HOLDOUT_DAYS = 90
# Expanding-window walk-forward CV.
CV_MIN_TRAIN_DAYS = 365
CV_VAL_DAYS = 60
CV_N_SPLITS = 5
# Gap between train end and validation start, in days. Must be >= max(HORIZONS)
# or the target of the last training row overlaps the validation window.
CV_GAP_DAYS = max(HORIZONS)

# ------------------------------------------------------------------- data range
# Verified in P0 by `python -m src.clients.open_meteo --probe`; the probe writes
# the true value to data/coverage.json and the backfill reads that, not this.
ARCHIVE_START_FALLBACK = "2022-06-01"

# ----------------------------------------------------------------------- alerts
ALERT_THRESHOLDS = {"unhealthy": 150.0, "very_unhealthy": 200.0, "hazardous": 300.0}
ALERT_STATE_FILE = DATA_DIR / "alert_state.json"

# ------------------------------------------------------------------- feature store
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
FG_HOURLY = "aqi_hourly_raw"
FG_DAILY = "aqi_daily_features"
FEATURE_VIEW = "aqi_3day_fv"
MODEL_REGISTRY_NAME = "aqi_forecaster"
FG_VERSION = 1

# ------------------------------------------------------------------------ misc
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RANDOM_SEED = 42

# Hours of overlap re-fetched on every hourly run. The scheduler is best-effort
# and runs get skipped, so each run must heal the gaps left by the last one.
HOURLY_LOOKBACK_HOURS = 72


@dataclass(frozen=True)
class AQIBand:
    lower: float
    upper: float
    label: str
    color: str
    guidance: str


AQI_BANDS: tuple[AQIBand, ...] = (
    AQIBand(0, 50, "Good", "#00E400",
            "Air quality is satisfactory and poses little or no risk."),
    AQIBand(51, 100, "Moderate", "#FFFF00",
            "Unusually sensitive people should consider limiting prolonged outdoor exertion."),
    AQIBand(101, 150, "Unhealthy for Sensitive Groups", "#FF7E00",
            "Children, older adults and people with heart or lung conditions should limit prolonged outdoor exertion."),
    AQIBand(151, 200, "Unhealthy", "#FF0000",
            "Everyone should limit prolonged outdoor exertion; sensitive groups should avoid it."),
    AQIBand(201, 300, "Very Unhealthy", "#8F3F97",
            "Everyone should avoid prolonged outdoor exertion; sensitive groups should remain indoors."),
    AQIBand(301, 500, "Hazardous", "#7E0023",
            "Health warning of emergency conditions. Everyone should remain indoors."),
)


def band_for(aqi: float) -> AQIBand:
    """Return the AQI band containing `aqi`, clamping above 500."""
    if aqi is None:
        return AQI_BANDS[0]
    for band in AQI_BANDS:
        if aqi <= band.upper:
            return band
    return AQI_BANDS[-1]


def band_index(aqi: float) -> int:
    for i, band in enumerate(AQI_BANDS):
        if aqi <= band.upper:
            return i
    return len(AQI_BANDS) - 1


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
