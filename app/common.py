"""Shared loading and styling for the dashboard.

Everything the app reads is precomputed by the pipelines. The app fits no
models, computes no SHAP and calls no forecasting APIs -- it renders. That
keeps it inside a Community Cloud memory allowance and makes a page load fast
even on a cold start.

Where the data comes from
-------------------------
The pipelines run on ephemeral GitHub Actions runners and persist their output
to an orphan `data` branch (see scripts/data_branch.sh). The app is deployed
from `main`, where `data/` is gitignored, so there is nothing to read locally:
a Community Cloud deployment would show an empty dashboard forever.

So each loader tries two sources in order:

  1. the local `data/` directory, which is what exists during development and
     inside a pipeline run
  2. the `data` branch over raw.githubusercontent.com, which is what exists on
     Community Cloud

Reading the branch over HTTPS rather than through Hopsworks is deliberate.
Hopsworks holds the feature groups, but the forecast, the experiment log, the
SHAP artefacts and the run summaries are only ever written to the mirror, so
the feature store alone cannot serve this dashboard.

Configure the source in Streamlit's secrets if the repository is renamed:

    DATA_REPO = "owner/repo"
    DATA_BRANCH = "data"
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AQI_BANDS, ARTIFACT_DIR, CITY, DATA_DIR, band_for  # noqa: E402
from src.store import local_store                                        # noqa: E402

TTL = 900  # 15 minutes; the feature pipeline runs hourly
REMOTE_TIMEOUT = 20


def _secret(name: str, default: str) -> str:
    """Streamlit secrets are optional; fall back to the default when unset."""
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def _raw_base() -> str:
    repo = _secret("DATA_REPO", "haris-hk/AQI-Predictor")
    branch = _secret("DATA_BRANCH", "data")
    return f"https://raw.githubusercontent.com/{repo}/{branch}/data"


@st.cache_data(ttl=TTL, show_spinner=False)
def _fetch(relative_path: str) -> bytes | None:
    """Fetch one artefact from the data branch. None when absent or unreachable.

    A 404 is the normal state before the first pipeline run, so it is not an
    error worth surfacing: the caller renders its empty state instead.
    """
    try:
        resp = requests.get(f"{_raw_base()}/{relative_path}", timeout=REMOTE_TIMEOUT)
    except requests.RequestException:
        return None
    return resp.content if resp.status_code == 200 else None


def _read_parquet(name: str) -> pd.DataFrame:
    local = local_store.read_parquet(DATA_DIR / name)
    if not local.empty:
        return local
    payload = _fetch(name)
    if payload is None:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception:
        return pd.DataFrame()


def _read_json(relative_path: str, local_path: Path) -> dict:
    if local_path.exists():
        try:
            return json.loads(local_path.read_text())
        except json.JSONDecodeError:
            pass
    payload = _fetch(relative_path)
    if payload is None:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


# ------------------------------------------------------------------- loaders
@st.cache_data(ttl=TTL, show_spinner=False)
def load_daily() -> pd.DataFrame:
    df = _read_parquet("daily_features.parquet")
    return df.sort_index() if not df.empty else df


@st.cache_data(ttl=TTL, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = _read_parquet("predictions.parquet")
    return df.sort_index() if not df.empty else df


@st.cache_data(ttl=TTL, show_spinner=False)
def load_experiments() -> pd.DataFrame:
    local = local_store.read_experiments()
    if not local.empty:
        return local
    payload = _fetch("experiments.csv")
    if payload is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(payload))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def load_json(name: str) -> dict:
    return _read_json(name, DATA_DIR / name)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_artifact_json(name: str) -> dict:
    return _read_json(f"artifacts/{name}", ARTIFACT_DIR / name)


def data_source_note() -> None:
    """Say plainly where the numbers came from, in the sidebar."""
    local = (DATA_DIR / "daily_features.parquet").exists()
    st.sidebar.caption(
        "Source: local `data/`" if local
        else f"Source: `{_secret('DATA_BRANCH', 'data')}` branch of "
             f"`{_secret('DATA_REPO', 'haris-hk/AQI-Predictor')}`")


# -------------------------------------------------------------------- display
def page_config(title: str) -> None:
    st.set_page_config(page_title=f"{title} | {CITY.name} AQI",
                       page_icon="🌫️", layout="wide")


def latest_forecast() -> pd.DataFrame:
    """Only the newest prediction run, one row per horizon."""
    preds = load_predictions()
    if preds.empty:
        return preds
    if "predicted_at" in preds.columns:
        preds = preds[preds["predicted_at"] == preds["predicted_at"].max()]
    return preds.sort_values("horizon") if "horizon" in preds.columns else preds


def band_card(label: str, value: float, sub: str = "", interval: tuple | None = None) -> str:
    band = band_for(value)
    interval_html = ""
    if interval and all(pd.notna(v) for v in interval):
        interval_html = (f"<div style='font-size:0.78rem;opacity:0.75;margin-top:2px'>"
                         f"likely {interval[0]:.0f} to {interval[1]:.0f}</div>")
    return f"""
    <div style="border-radius:12px;padding:14px 16px;background:{band.color}22;
                border-left:6px solid {band.color};height:100%">
      <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;
                  opacity:0.7">{label}</div>
      <div style="font-size:2.4rem;font-weight:700;line-height:1.1">{value:.0f}</div>
      <div style="font-size:0.95rem;font-weight:600">{band.label}</div>
      {interval_html}
      <div style="font-size:0.78rem;opacity:0.7;margin-top:6px">{sub}</div>
    </div>"""


def band_legend() -> str:
    chips = "".join(
        f"<span style='display:inline-block;padding:2px 8px;margin:2px;border-radius:10px;"
        f"background:{b.color}33;border:1px solid {b.color};font-size:0.72rem'>"
        f"{b.label} {int(b.lower)}-{int(b.upper)}</span>"
        for b in AQI_BANDS)
    return f"<div>{chips}</div>"


def data_freshness_note(daily: pd.DataFrame) -> None:
    data_source_note()
    if daily.empty:
        st.warning(
            "No data yet. The pipelines have not published anything to the "
            "`data` branch. Run the **Historical backfill** workflow in the "
            "repository's Actions tab, then the **Training pipeline**.")
        return
    observed = daily[daily["aqi_mean"].notna()]
    if observed.empty:
        return
    last = observed.index.max()
    age_days = (pd.Timestamp.now().normalize() - last.normalize()).days
    if age_days > 2:
        st.warning(f"Latest observation is {last.date()} ({age_days} days old). "
                   "The hourly feature pipeline may not be running.")
