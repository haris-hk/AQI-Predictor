"""Page 1: the forecast. The landing page and the point of the whole project."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    band_card, band_legend, data_freshness_note, latest_forecast, load_daily,
    load_json, page_config,
)
from src.config import AQI_BANDS, CITY, band_for

page_config("Forecast")

st.title(f"Air quality forecast · {CITY.name}")
st.caption("Three-day US AQI forecast from a serverless pipeline. "
           "Features refresh hourly, models retrain daily.")

daily = load_daily()
forecast = latest_forecast()
data_freshness_note(daily)

# ------------------------------------------------------------------ today
observed = daily[daily["aqi_mean"].notna()] if not daily.empty else pd.DataFrame()
if not observed.empty:
    today = observed.iloc[-1]
    st.markdown("#### Today")
    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        st.markdown(
            band_card("Observed daily mean", float(today["aqi_mean"]),
                      f"{observed.index[-1].date()} · dominant pollutant: "
                      f"{today.get('dominant_pollutant', 'n/a')}"),
            unsafe_allow_html=True)
    cols[1].metric("Daily max", f"{today.get('aqi_max', float('nan')):.0f}")
    cols[2].metric("PM2.5", f"{today.get('pm2_5_mean', float('nan')):.0f} µg/m³")
    cols[3].metric("Wind", f"{today.get('wind_speed_10m_mean', float('nan')):.1f} km/h")
    st.caption(band_for(float(today["aqi_mean"])).guidance)

# --------------------------------------------------------------- the forecast
st.markdown("#### Next three days")
if forecast.empty:
    st.info("No forecast yet. Train a model, then run the inference pipeline.")
else:
    cols = st.columns(len(forecast))
    for col, (target_date, row) in zip(cols, forecast.iterrows()):
        with col:
            interval = (row.get("aqi_mean_lower"), row.get("aqi_mean_upper"))
            st.markdown(
                band_card(
                    pd.Timestamp(target_date).strftime("%a %d %b"),
                    float(row["aqi_mean"]),
                    f"max {row.get('aqi_max', float('nan')):.0f} · "
                    f"{int(row.get('horizon', 0))} day(s) ahead",
                    interval,
                ),
                unsafe_allow_html=True)
    worst = forecast.loc[forecast["aqi_mean"].idxmax()]
    st.caption(f"**{band_for(float(worst['aqi_mean'])).label} peak expected "
               f"{pd.Timestamp(worst.name).strftime('%A')}.** "
               f"{band_for(float(worst['aqi_mean'])).guidance}")

# ------------------------------------------------------------------- chart
st.markdown("#### Recent history and forecast")
if not observed.empty:
    window = observed.tail(30)
    fig = go.Figure()
    for band in AQI_BANDS:
        fig.add_hrect(y0=band.lower, y1=min(band.upper, 500),
                      fillcolor=band.color, opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(
        x=window.index, y=window["aqi_mean"], name="Observed daily mean",
        mode="lines+markers", line=dict(width=2.5, color="#1f2937")))

    if not forecast.empty:
        bridge_x = [window.index[-1]] + list(forecast.index)
        bridge_y = [float(window["aqi_mean"].iloc[-1])] + list(forecast["aqi_mean"])
        if {"aqi_mean_lower", "aqi_mean_upper"} <= set(forecast.columns):
            fig.add_trace(go.Scatter(
                x=list(forecast.index) + list(forecast.index)[::-1],
                y=list(forecast["aqi_mean_upper"]) + list(forecast["aqi_mean_lower"])[::-1],
                fill="toself", fillcolor="rgba(217,70,239,0.18)",
                line=dict(width=0), name="Prediction interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y, name="Forecast", mode="lines+markers",
            line=dict(width=2.5, dash="dash", color="#a21caf")))

    fig.update_layout(height=420, hovermode="x unified", margin=dict(t=10, b=10),
                      yaxis_title="US AQI", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')
    st.markdown(band_legend(), unsafe_allow_html=True)

# ------------------------------------------------------------------- footer
meta = load_json("training_summary.json")
run_meta = load_json("last_feature_run.json")
with st.expander("Pipeline status"):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Last feature run**")
        st.json({k: run_meta.get(k) for k in
                 ("ran_at", "latest_hour", "daily_rows_total", "hopsworks_daily")}
                if run_meta else {"status": "not run yet"})
    with c2:
        st.write("**Last training run**")
        st.json({k: meta.get(k) for k in ("run_id", "promoted", "reason", "rmse_mean")}
                if meta else {"status": "not trained yet"})

st.caption("Data: Open-Meteo (CAMS reanalysis and forecast). AQI computed with "
           "the US EPA breakpoint tables. Forecasts are model output, not an "
           "official advisory.")
