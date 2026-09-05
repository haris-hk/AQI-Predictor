"""Page 2: history and exploratory analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import load_daily, page_config
from src.config import AQI_BANDS, CITY, band_for

page_config("History & EDA")
st.title("History and exploratory analysis")

daily = load_daily()
if daily.empty:
    st.warning("No data yet. Run the backfill workflow.")
    st.stop()

observed = daily[daily["aqi_mean"].notna()]
st.caption(f"{len(observed):,} days · {observed.index.min().date()} to "
           f"{observed.index.max().date()}")

# ------------------------------------------------------------------ overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean AQI", f"{observed['aqi_mean'].mean():.0f}")
c2.metric("Median AQI", f"{observed['aqi_mean'].median():.0f}")
c3.metric("Worst day", f"{observed['aqi_mean'].max():.0f}")
unhealthy = (observed["aqi_mean"] > 150).mean() * 100
c4.metric("Days above 150", f"{unhealthy:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Time series", "Seasonality", "Weather relationships", "Data quality"])

with tab1:
    span = st.select_slider("Window", options=[90, 180, 365, 730, len(observed)],
                            value=min(365, len(observed)),
                            format_func=lambda v: f"{v} days")
    window = observed.tail(span)
    fig = go.Figure()
    for band in AQI_BANDS:
        fig.add_hrect(y0=band.lower, y1=min(band.upper, 500),
                      fillcolor=band.color, opacity=0.09, line_width=0)
    fig.add_trace(go.Scatter(x=window.index, y=window["aqi_mean"],
                             name="Daily mean", line=dict(width=1.6, color="#1f2937")))
    if "aqi_mean_roll30_mean" in window:
        fig.add_trace(go.Scatter(x=window.index, y=window["aqi_mean_roll30_mean"],
                                 name="30-day mean", line=dict(width=3, color="#a21caf")))
    fig.update_layout(height=430, yaxis_title="US AQI", hovermode="x unified",
                      margin=dict(t=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Distribution by AQI band**")
    bands = observed["aqi_mean"].map(lambda v: band_for(v).label).value_counts()
    order = [b.label for b in AQI_BANDS if b.label in bands.index]
    st.plotly_chart(
        px.bar(x=order, y=[bands[o] for o in order],
               color=order, color_discrete_map={b.label: b.color for b in AQI_BANDS},
               labels={"x": "", "y": "days"}).update_layout(
                   height=280, showlegend=False, margin=dict(t=10)),
        width='stretch')

with tab2:
    pivot = observed.pivot_table(index=observed.index.month,
                                 columns=observed.index.year,
                                 values="aqi_mean", aggfunc="mean")
    st.markdown("**Monthly mean AQI by year**")
    st.plotly_chart(
        px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                  labels=dict(x="year", y="month", color="AQI")
                  ).update_layout(height=380, margin=dict(t=10)),
        width='stretch')

    c1, c2 = st.columns(2)
    monthly = observed.groupby(observed.index.month)["aqi_mean"].agg(["mean", "std"])
    c1.plotly_chart(
        px.line(monthly, y="mean", markers=True, labels={"index": "month", "mean": "AQI"},
                title="Seasonal cycle").update_layout(height=320, margin=dict(t=40)),
        width='stretch')
    weekday = observed.groupby(observed.index.dayofweek)["aqi_mean"].mean()
    weekday.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    c2.plotly_chart(
        px.bar(weekday, labels={"index": "", "value": "AQI"}, title="Day of week"
               ).update_layout(height=320, showlegend=False, margin=dict(t=40)),
        width='stretch')

with tab3:
    drivers = [c for c in ("wind_speed_10m_mean", "boundary_layer_height_mean",
                           "ventilation_index", "temperature_2m_mean",
                           "relative_humidity_2m_mean", "precipitation_sum",
                           "surface_pressure_mean", "temp_range")
               if c in observed.columns]
    if drivers:
        corr = observed[["aqi_mean"] + drivers].corr()["aqi_mean"].drop("aqi_mean")
        st.markdown("**Correlation with daily mean AQI**")
        st.plotly_chart(
            px.bar(corr.sort_values(), orientation="h",
                   labels={"value": "correlation", "index": ""},
                   color=corr.sort_values(), color_continuous_scale="RdBu",
                   range_color=[-1, 1]).update_layout(height=340, margin=dict(t=10)),
            width='stretch')
        choice = st.selectbox("Scatter against", drivers, index=0)
        st.plotly_chart(
            px.scatter(observed, x=choice, y="aqi_mean", opacity=0.45,
                       trendline="lowess", color=observed.index.month,
                       color_continuous_scale="Turbo",
                       labels={"color": "month"}).update_layout(height=430, margin=dict(t=10)),
            width='stretch')
        st.caption("Ventilation -- wind speed times mixing-layer depth -- is the "
                   "physical mechanism: pollution concentration is roughly "
                   "emissions divided by the volume available to dilute them.")

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Hourly observations per day**")
        if "hours_observed" in observed.columns:
            st.plotly_chart(
                px.histogram(observed, x="hours_observed", nbins=25
                             ).update_layout(height=300, margin=dict(t=10)),
                width='stretch')
    with c2:
        st.markdown("**Most incomplete features**")
        nulls = (daily.isna().mean() * 100).sort_values(ascending=False).head(15)
        st.dataframe(nulls.rename("% null").round(1), width='stretch')
    expected = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    gaps = expected.difference(observed.index)
    st.metric("Missing calendar days", len(gaps))
    if len(gaps):
        st.caption("Examples: " + ", ".join(str(d.date()) for d in gaps[:12]))
