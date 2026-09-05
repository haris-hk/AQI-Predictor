"""Page 3: the model leaderboard and error diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_daily, load_experiments, load_json, page_config

page_config("Model performance")
st.title("Model performance")

experiments = load_experiments()
summary = load_json("training_summary.json")

if summary:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest run", str(summary.get("run_id", "n/a")))
    c2.metric("Mean holdout RMSE", f"{summary.get('rmse_mean', float('nan')):.2f}")
    c3.metric("Promoted", "yes" if summary.get("promoted") else "no")
    if not summary.get("promoted"):
        st.warning(f"Model not promoted: {summary.get('reason', 'unknown')}. "
                   "The incumbent stays live.")

    holdout = summary.get("holdout", {})
    if holdout:
        st.markdown("#### Holdout performance against persistence")
        st.caption("The holdout is the final 90 days, chronologically separated and "
                   "evaluated exactly once. A model that fails to beat persistence "
                   "here is not promoted, whatever its R².")
        rows = []
        for key, m in holdout.items():
            target, horizon = key.split("|")
            rows.append({
                "target": target, "horizon": int(horizon),
                "RMSE": m.get("rmse"), "persistence RMSE": m.get("persistence_rmse"),
                "skill": m.get("skill_vs_persistence"),
                "MAE": m.get("mae"), "R²": m.get("r2"),
                "band accuracy": m.get("band_accuracy"),
                "beats persistence": m.get("beats_persistence"),
            })
        table = pd.DataFrame(rows).sort_values(["target", "horizon"])
        st.dataframe(
            table.style.format({
                "RMSE": "{:.2f}", "persistence RMSE": "{:.2f}", "skill": "{:+.3f}",
                "MAE": "{:.2f}", "R²": "{:.3f}", "band accuracy": "{:.1%}"}),
            width='stretch', hide_index=True)

        st.plotly_chart(
            px.bar(table, x="horizon", y="skill", color="target", barmode="group",
                   title="Skill score against persistence (higher is better; 0 means no gain)"
                   ).add_hline(y=0, line_dash="dash").update_layout(
                       height=340, margin=dict(t=50)),
            width='stretch')

if experiments.empty:
    st.info("No experiment log yet. Run the training pipeline.")
    st.stop()

st.markdown("#### Cross-validation leaderboard")
latest_run = experiments["run_id"].max() if "run_id" in experiments else None
scope = st.radio("Scope", ["Latest run", "All runs"], horizontal=True)
frame = experiments[experiments["run_id"] == latest_run] if scope == "Latest run" and latest_run \
    else experiments

target = st.selectbox("Target", sorted(frame["target"].dropna().unique()))
horizon = st.selectbox("Horizon (days ahead)", sorted(frame["horizon"].dropna().unique()))
subset = frame[(frame["target"] == target) & (frame["horizon"] == horizon)]
subset = subset.sort_values("rmse")

cols = [c for c in ("model", "rmse", "rmse_std", "mae", "r2", "skill_vs_persistence",
                    "band_accuracy", "band_within_one", "n_folds")
        if c in subset.columns]
st.dataframe(subset[cols].style.format({
    "rmse": "{:.2f}", "rmse_std": "{:.2f}", "mae": "{:.2f}", "r2": "{:.3f}",
    "skill_vs_persistence": "{:+.3f}", "band_accuracy": "{:.1%}",
    "band_within_one": "{:.1%}"}), width='stretch', hide_index=True)

baselines = {"persistence", "seasonal_naive", "climatology", "drifted_persistence"}
subset = subset.assign(kind=np.where(subset["model"].isin(baselines), "baseline", "learned"))
st.plotly_chart(
    px.bar(subset, x="model", y="rmse", color="kind", error_y="rmse_std" if "rmse_std" in subset else None,
           color_discrete_map={"baseline": "#94a3b8", "learned": "#a21caf"},
           title=f"Cross-validated RMSE · {target} at {int(horizon)} day(s)"
           ).update_layout(height=420, xaxis_tickangle=-35, margin=dict(t=50)),
    width='stretch')

st.caption("Grey bars are the reference models. A learned model that does not sit "
           "clearly below them has not earned its complexity.")
