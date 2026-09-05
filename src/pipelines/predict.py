"""Inference. Runs hourly, right after the feature pipeline.

Predictions are computed here and written to the store, not computed inside the
Streamlit app. Two reasons: Community Cloud apps are memory-constrained and a
model load per page view is wasteful, and a stored prediction with a timestamp
is auditable -- we can go back and score what we actually said against what
happened, which is what the live skill tracking on the dashboard needs.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, CITY, TARGETS, ensure_dirs
from src.models.bundle import ModelBundle
from src.store import local_store

log = logging.getLogger(__name__)


def latest_features() -> pd.DataFrame:
    df = local_store.read_daily()
    if df.empty:
        raise SystemExit("no features available -- run the feature pipeline first")
    return df.sort_index()


def run(save: bool = True) -> dict:
    ensure_dirs()
    bundle = ModelBundle.load(ARTIFACT_DIR)
    if bundle is None:
        raise SystemExit("no trained model -- run the training pipeline first")

    features = latest_features()
    # The most recent day with an observed AQI is the forecast origin. Later
    # rows exist (the feature frame extends into the forecast window) but have
    # no observed target and are not a valid origin.
    observed = features[features["aqi_mean"].notna()]
    forecast = bundle.predict_row(observed)

    if forecast.empty:
        raise SystemExit("model produced no forecast")

    forecast["city_id"] = CITY.city_id
    if save:
        local_store.write_predictions(forecast)

    score = score_past_predictions(features)
    summary = {
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(observed.index.max().date()),
        "forecast": json.loads(forecast.reset_index().to_json(
            orient="records", date_format="iso")),
        "live_scoring": score,
        "model_trained_at": bundle.metadata.get("trained_at"),
    }
    local_store.write_json("latest_forecast.json", summary)
    for _, row in forecast.iterrows():
        log.info("%s  mean %.0f  max %.0f  %s",
                 row.name.date(), row.get("aqi_mean", np.nan),
                 row.get("aqi_max", np.nan), row.get("band", ""))
    return summary


def score_past_predictions(features: pd.DataFrame) -> dict:
    """Grade stored forecasts against what actually happened.

    Backtest metrics are an estimate of live performance. This is live
    performance, and the gap between them is the most informative number in the
    whole project -- it is where any remaining train/serve skew shows up.
    """
    preds = local_store.read_predictions()
    if preds.empty or features.empty:
        return {}
    joined = preds.join(features[list(TARGETS)], how="inner", rsuffix="_actual")
    out: dict = {}
    for target in TARGETS:
        actual_col = f"{target}_actual" if f"{target}_actual" in joined.columns else target
        if target not in joined.columns or actual_col not in joined.columns:
            continue
        pair = joined[[target, actual_col, "horizon"]].dropna()
        if len(pair) < 5:
            continue
        for horizon, group in pair.groupby("horizon"):
            err = group[target] - group[actual_col]
            out[f"{target}|h{int(horizon)}"] = {
                "n": int(len(group)),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "mae": float(err.abs().mean()),
                "bias": float(err.mean()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the 3-day AQI forecast")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(save=not args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
