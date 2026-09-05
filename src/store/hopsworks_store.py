"""Hopsworks feature store and model registry.

Every function here degrades to a no-op when credentials are absent or the
`hopsworks` package is not installed, and every caller checks `is_available()`
before relying on a result. That is deliberate: the local Parquet mirror is
always written first, so a Hopsworks outage costs us the managed store for that
run and nothing else. An automated pipeline whose every stage is a hard
dependency on a third party is a pipeline that stops.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    FEATURE_VIEW, FG_DAILY, FG_HOURLY, FG_VERSION, HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT, MODEL_REGISTRY_NAME,
)

log = logging.getLogger(__name__)

_project = None
_fs = None


def is_configured() -> bool:
    return bool(HOPSWORKS_API_KEY)


def _login():
    """Cached login. Returns None if unavailable, never raises to the caller."""
    global _project, _fs
    if _fs is not None:
        return _fs
    if not is_configured():
        log.info("HOPSWORKS_API_KEY not set; using the local Parquet store only")
        return None
    try:
        import hopsworks
    except ImportError:
        log.warning("hopsworks package not installed; using the local store only")
        return None
    try:
        kwargs: dict[str, Any] = {"api_key_value": HOPSWORKS_API_KEY}
        if HOPSWORKS_PROJECT:
            kwargs["project"] = HOPSWORKS_PROJECT
        _project = hopsworks.login(**kwargs)
        _fs = _project.get_feature_store()
        log.info("connected to Hopsworks project %s", _project.name)
        return _fs
    except Exception as exc:                          # pragma: no cover
        log.warning("Hopsworks login failed (%s); using the local store only", exc)
        return None


def is_available() -> bool:
    return _login() is not None


# ------------------------------------------------------------- feature groups
def _reset_index_for_hopsworks(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    out = df.reset_index()
    if index_name not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": index_name})
    # Hopsworks column names must be lowercase and free of special characters.
    out.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def write_hourly(df: pd.DataFrame) -> bool:
    fs = _login()
    if fs is None or df.empty:
        return False
    try:
        payload = _reset_index_for_hopsworks(df, "ts_utc")
        if "city_id" not in payload.columns:
            from src.config import CITY
            payload["city_id"] = CITY.city_id
        fg = fs.get_or_create_feature_group(
            name=FG_HOURLY, version=FG_VERSION,
            description="Raw hourly pollutant and weather observations with the EPA AQI",
            primary_key=["city_id", "ts_utc"], event_time="ts_utc",
            online_enabled=False,
        )
        fg.insert(payload, write_options={"wait_for_job": False})
        return True
    except Exception as exc:                          # pragma: no cover
        log.warning("Hopsworks hourly write failed: %s", exc)
        return False


def write_daily(df: pd.DataFrame) -> bool:
    fs = _login()
    if fs is None or df.empty:
        return False
    try:
        payload = _reset_index_for_hopsworks(df, "date")
        fg = fs.get_or_create_feature_group(
            name=FG_DAILY, version=FG_VERSION,
            description="Engineered daily features and 3-day AQI targets",
            primary_key=["city_id", "date"], event_time="date",
            online_enabled=False,
        )
        fg.insert(payload, write_options={"wait_for_job": False})
        return True
    except Exception as exc:                          # pragma: no cover
        log.warning("Hopsworks daily write failed: %s", exc)
        return False


def read_daily() -> pd.DataFrame:
    fs = _login()
    if fs is None:
        return pd.DataFrame()
    try:
        fg = fs.get_feature_group(name=FG_DAILY, version=FG_VERSION)
        df = fg.read()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        return df
    except Exception as exc:                          # pragma: no cover
        log.warning("Hopsworks daily read failed: %s", exc)
        return pd.DataFrame()


def get_or_create_feature_view():
    """The single read path shared by training and serving.

    Routing both through one feature view is what makes train/serve consistency
    structural rather than a matter of remembering to keep two code paths in
    step.
    """
    fs = _login()
    if fs is None:
        return None
    try:
        return fs.get_feature_view(name=FEATURE_VIEW, version=FG_VERSION)
    except Exception:
        try:
            fg = fs.get_feature_group(name=FG_DAILY, version=FG_VERSION)
            return fs.create_feature_view(
                name=FEATURE_VIEW, version=FG_VERSION,
                description="Daily features and 3-day AQI targets",
                query=fg.select_all(),
            )
        except Exception as exc:                      # pragma: no cover
            log.warning("could not create feature view: %s", exc)
            return None


# ------------------------------------------------------------- model registry
def register_model(model_dir: Path, metrics: dict, description: str = "",
                   input_example=None) -> bool:
    fs = _login()
    if fs is None or _project is None:
        return False
    try:
        mr = _project.get_model_registry()
        clean = {k: float(v) for k, v in metrics.items()
                 if isinstance(v, (int, float)) and pd.notna(v)}
        model = mr.python.create_model(
            name=MODEL_REGISTRY_NAME,
            metrics=clean,
            description=description or "3-day AQI forecaster",
            input_example=input_example,
        )
        model.save(str(model_dir))
        log.info("registered %s v%s", MODEL_REGISTRY_NAME, model.version)
        return True
    except Exception as exc:                          # pragma: no cover
        log.warning("model registration failed: %s", exc)
        return False


def get_best_model_metrics(metric: str = "rmse_mean") -> dict:
    """Incumbent metrics, used by the promotion rule in the training pipeline."""
    fs = _login()
    if fs is None or _project is None:
        return {}
    try:
        mr = _project.get_model_registry()
        best = mr.get_best_model(MODEL_REGISTRY_NAME, metric, "min")
        return dict(best.training_metrics) if best else {}
    except Exception:                                 # pragma: no cover
        return {}


def download_latest_model(target_dir: Path) -> Path | None:
    fs = _login()
    if fs is None or _project is None:
        return None
    try:
        mr = _project.get_model_registry()
        best = mr.get_best_model(MODEL_REGISTRY_NAME, "rmse_mean", "min")
        if best is None:
            return None
        return Path(best.download())
    except Exception as exc:                          # pragma: no cover
        log.warning("model download failed: %s", exc)
        return None
