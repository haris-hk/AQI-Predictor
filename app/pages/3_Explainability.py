"""Page 4: what the model is actually looking at."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import load_artifact_json, load_json, page_config

page_config("Explainability")
st.title("Explainability")
st.caption("SHAP values, precomputed in the training job. Global importance is the "
           "sanity check -- if meteorology does not rank highly, the model has "
           "learned the calendar rather than the air.")

global_shap = load_artifact_json("shap_global.json")
local_shap = load_artifact_json("shap_latest.json")

if not global_shap:
    st.info("No explanations yet. They are written by the training pipeline; "
            "the explain step is skipped when the winning model is not tree-based.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Explained model", global_shap.get("model", "n/a"))
c2.metric("Target", f"{global_shap.get('target')} · h{global_shap.get('horizon')}")
c3.metric("Background rows", global_shap.get("n_background", 0))

st.markdown("#### Global feature importance")
importance = pd.Series(global_shap.get("global_importance", {})).sort_values()
top = importance.tail(20)
st.plotly_chart(
    px.bar(top, orientation="h", labels={"value": "mean |SHAP| (AQI points)", "index": ""},
           color=top, color_continuous_scale="Magma"
           ).update_layout(height=560, showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=10)),
    width='stretch')

with st.expander("How to read this"):
    st.markdown("""
- **Lag features ranking first is expected, not a problem.** AQI is persistent;
  yesterday's value is genuinely the strongest single predictor. The question is
  what ranks *next*.
- **Meteorology ranking second is the good sign.** Ventilation index, wind speed
  and mixing-layer depth are the physical mechanism, and a model using them is
  doing something a persistence baseline cannot.
- **Calendar features dominating would be a warning.** It would mean the model
  had learned "January is bad" rather than "tomorrow is bad".
    """)

if local_shap:
    st.markdown("#### Why this particular forecast")
    st.caption(f"Contributions for {local_shap.get('as_of')} · "
               f"{local_shap.get('target')} at {local_shap.get('horizon')} day(s) ahead")

    contributions = pd.Series(local_shap.get("contributions", {}))
    base = local_shap.get("base_value", 0.0)
    prediction = local_shap.get("prediction", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline (average day)", f"{base:.0f}")
    c2.metric("This forecast", f"{prediction:.0f}")
    c3.metric("Difference", f"{prediction - base:+.0f}")

    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(14)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(ordered),
        x=list(ordered.index),
        y=list(ordered.values),
        increasing=dict(marker_color="#dc2626"),
        decreasing=dict(marker_color="#2563eb"),
    ))
    fig.update_layout(height=460, xaxis_tickangle=-40,
                      yaxis_title="contribution to the forecast (AQI points)",
                      margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption("Red pushes the forecast up, blue pushes it down. Bars sum "
               "(with the remaining features) to the difference from the baseline.")
