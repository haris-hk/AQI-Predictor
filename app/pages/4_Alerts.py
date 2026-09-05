"""Page 5: alert configuration and history."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from common import latest_forecast, load_json, page_config
from src.config import ALERT_THRESHOLDS, CITY, band_for

page_config("Alerts")
st.title("Hazard alerts")

state = load_json("alert_state.json")
forecast = latest_forecast()

st.markdown("#### Current thresholds")
cols = st.columns(3)
for col, (name, value) in zip(cols, ALERT_THRESHOLDS.items()):
    band = band_for(value)
    col.markdown(
        f"<div style='border-radius:10px;padding:12px;background:{band.color}22;"
        f"border-left:5px solid {band.color}'>"
        f"<div style='font-size:0.75rem;opacity:0.7;text-transform:uppercase'>"
        f"{name.replace('_', ' ')}</div>"
        f"<div style='font-size:1.8rem;font-weight:700'>AQI ≥ {value:.0f}</div></div>",
        unsafe_allow_html=True)

st.caption("An alert fires only when the forecast is **worse than today**. In a "
           "Karachi winter the index sits above 150 for weeks at a time, and an "
           "alert that repeats every six hours for a month is one nobody reads.")

st.markdown("#### Forecast against thresholds")
if forecast.empty:
    st.info("No forecast available.")
else:
    rows = []
    for target_date, row in forecast.iterrows():
        value = row.get("aqi_max", row.get("aqi_mean"))
        triggered = [n for n, t in ALERT_THRESHOLDS.items() if value >= t]
        rows.append({
            "date": pd.Timestamp(target_date).date(),
            "forecast AQI (max)": round(float(value)),
            "band": band_for(float(value)).label,
            "would trigger": triggered[0] if triggered else "-",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

st.markdown("#### Alert history")
history = state.get("history", []) if state else []
if not history:
    st.info("No alerts sent yet.")
else:
    flat = []
    for entry in reversed(history[-40:]):
        for alert in entry.get("alerts", []):
            flat.append({
                "sent at": entry.get("at", "")[:16].replace("T", " "),
                "for date": alert.get("target_date"),
                "AQI": round(alert.get("aqi", 0)),
                "severity": alert.get("severity"),
                "channels": ", ".join(
                    k for k, ok in (entry.get("channels") or {}).items() if ok) or "none",
            })
    st.dataframe(pd.DataFrame(flat), width='stretch', hide_index=True)

with st.expander("How to receive these"):
    st.markdown("""
Set any of these as repository secrets and the alert job will use them:

| Secret | Channel | Setup |
|---|---|---|
| `NTFY_TOPIC` | Push notification | Pick any topic name, subscribe in the ntfy app. No account. |
| `SLACK_WEBHOOK_URL` | Slack message | Create an incoming webhook in your workspace. |
| `ALERT_EMAIL_TO` + `SMTP_USER` + `SMTP_PASSWORD` | Email | An app password from your mail provider. |

With none configured the alert is still evaluated and recorded here.
    """)
