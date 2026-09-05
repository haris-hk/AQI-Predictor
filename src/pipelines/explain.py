"""Explanation pipeline. Runs after training.

Separated from the training job so explanations can be refreshed without
retraining, and so a SHAP failure can never fail a training run that otherwise
succeeded.
"""
from __future__ import annotations

import argparse
import json
import logging

from src.config import ARTIFACT_DIR, HORIZONS, TARGETS, ensure_dirs
from src.explain.shap_explain import explain
from src.features.build import feature_columns
from src.models.bundle import ModelBundle
from src.store import local_store

log = logging.getLogger(__name__)


def run() -> dict:
    ensure_dirs()
    bundle = ModelBundle.load(ARTIFACT_DIR)
    if bundle is None:
        return {"status": "no model bundle"}
    df = local_store.read_daily()
    if df.empty:
        return {"status": "no features"}

    X = df[feature_columns(df)]
    out: dict = {}

    # The dashboard shows one explanation: daily mean AQI, one day ahead. Later
    # pairs are tried only if the preferred one is not explainable, so a
    # non-tree winner at h1 does not leave the page empty.
    preferred = [("aqi_mean", h) for h in HORIZONS] + \
                [(t, h) for t in TARGETS for h in HORIZONS if t != "aqi_mean"]

    for target, horizon in preferred:
        if (target, horizon) not in bundle.models:
            continue
        result = explain(bundle, X, target, horizon)
        if result:
            out[f"{target}|{horizon}"] = list(result["global_importance"])[:10]
            break

    if not out:
        # Model-agnostic fallback so the page is never blank when the winner is
        # a linear or neural model that TreeExplainer cannot handle.
        out = _permutation_fallback(bundle, df)
    return out or {"status": "no explanations produced"}


def _permutation_fallback(bundle, df) -> dict:
    import json

    from src.config import target_col
    from src.explain.shap_explain import GLOBAL_FILE, permutation_fallback

    for target in TARGETS:
        for horizon in HORIZONS:
            model = bundle.models.get((target, horizon))
            col = target_col(target, horizon)
            if model is None or col not in df.columns:
                continue
            usable = df[df[col].notna()].tail(400)
            if len(usable) < 60:
                continue
            importance = permutation_fallback(
                model, bundle.align(usable), usable[col])
            if not importance:
                continue
            payload = {
                "target": target, "horizon": horizon,
                "model": type(model).__name__,
                "method": "permutation_importance",
                "n_background": int(len(usable)),
                "global_importance": importance,
            }
            (ARTIFACT_DIR / GLOBAL_FILE).write_text(json.dumps(payload, indent=2))
            log.info("wrote permutation importance fallback")
            return {f"{target}|{horizon}": list(importance)[:10]}
    return {}


def main() -> None:
    argparse.ArgumentParser(description="Refresh SHAP explanations").parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
