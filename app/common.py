"""Shared loading and styling for the dashboard.

Everything the app reads is precomputed by the pipelines. The app fits models,
computes SHAP and calls no APIs -- it renders. That keeps it inside a Community
Cloud memory allowance and makes a page load fast even on a cold start.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AQI_BANDS, CITY, band_for  # noqa: E402
from src.store import local_store                  # noqa: E402

TTL = 900  # 15 minutes; the feature pipeline runs hourly


@st.cache_data(ttl=TTL, show_spinner=False)
def load_daily() -> pd.DataFrame:
    df = local_store.read_daily()
    return df.sort_index() if not df.empty else df


@st.cache_data(ttl=TTL, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = local_store.read_predictions()
    return df.sort_index() if not df.empty else df


@st.cache_data(ttl=TTL, show_spinner=False)
def load_experiments() -> pd.DataFrame:
    return local_store.read_experiments()


@st.cache_data(ttl=TTL, show_spinner=False)
def load_json(name: str) -> dict:
    return local_store.read_json(name)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_artifact_json(name: str) -> dict:
    import json
    from src.config import ARTIFACT_DIR
    path = ARTIFACT_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


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
    if daily.empty:
        st.warning("No data yet. Run the backfill workflow to populate the feature store.")
        return
    observed = daily[daily["aqi_mean"].notna()]
    if observed.empty:
        return
    last = observed.index.max()
    age_days = (pd.Timestamp.now().normalize() - last.normalize()).days
    if age_days > 2:
        st.warning(f"Latest observation is {last.date()} ({age_days} days old). "
                   "The hourly feature pipeline may not be running.")
