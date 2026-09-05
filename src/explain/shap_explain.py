"""SHAP explanations, precomputed.

Computed in the training job and cached to disk, never inside the Streamlit
app: a TreeExplainer over a few hundred rows will blow through a Community
Cloud memory allowance, and the app would recompute it on every page view.

Two artefacts are produced. Global importance answers "what does this model pay
attention to?", which is the sanity check -- if wind speed and mixing height do
not rank highly, the model has learned the calendar instead of the meteorology.
Per-day values answer "why is Thursday forecast to be bad?", which is the
question a dashboard reader actually has.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR

log = logging.getLogger(__name__)

GLOBAL_FILE = "shap_global.json"
LOCAL_FILE = "shap_latest.json"

try:
    import shap
    HAS_SHAP = True
except ImportError:                                   # pragma: no cover
    HAS_SHAP = False


# TreeExplainer supports these. Detecting by type name rather than by the
# presence of `feature_importances_`: HistGradientBoostingRegressor is a tree
# ensemble and is fully supported by SHAP, but does not expose that attribute,
# so an attribute check silently skips the model that most often wins here.
TREE_MODELS = {
    "RandomForestRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
}


def is_tree_model(estimator) -> bool:
    return type(estimator).__name__ in TREE_MODELS


def _unwrap(model):
    """Reach the tree estimator inside a scikit-learn Pipeline."""
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def _transform(model, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's preprocessing so SHAP sees what the tree sees."""
    if not hasattr(model, "named_steps"):
        return X
    out = X
    for name, step in model.named_steps.items():
        if name == "model":
            break
        out = pd.DataFrame(step.transform(out), columns=X.columns, index=X.index)
    return out


def explain(bundle, X: pd.DataFrame, target: str = "aqi_mean", horizon: int = 1,
            sample: int = 300, directory: Path | None = None) -> dict:
    """Compute and cache SHAP for one (target, horizon). Safe to call blind."""
    directory = Path(directory or ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    if not HAS_SHAP:
        log.info("shap not installed; skipping explanations")
        return {}
    model = bundle.models.get((target, horizon))
    if model is None or X.empty:
        return {}

    aligned = bundle.align(X)
    estimator = _unwrap(model)
    if not is_tree_model(estimator):
        log.info("%s is not tree-based; TreeExplainer does not apply",
                 type(estimator).__name__)
        return {}

    background = aligned.tail(sample)
    transformed = _transform(model, background)

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed, check_additivity=False)
    except Exception as exc:                          # pragma: no cover
        log.warning("SHAP failed: %s", exc)
        return {}

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]

    mean_abs = np.abs(values).mean(axis=0)
    ranking = (
        pd.Series(mean_abs, index=list(aligned.columns))
        .sort_values(ascending=False)
    )

    payload = {
        "target": target,
        "horizon": horizon,
        "model": type(estimator).__name__,
        "n_background": int(len(background)),
        "global_importance": {k: float(v) for k, v in ranking.head(40).items()},
    }
    (directory / GLOBAL_FILE).write_text(json.dumps(payload, indent=2))

    last_idx = -1
    local = {
        "target": target,
        "horizon": horizon,
        "as_of": str(background.index[last_idx].date()),
        "base_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        "prediction": float(model.predict(background.iloc[[last_idx]])[0]),
        "contributions": {
            k: float(v) for k, v in sorted(
                zip(aligned.columns, values[last_idx]),
                key=lambda kv: abs(kv[1]), reverse=True)[:20]
        },
    }
    (directory / LOCAL_FILE).write_text(json.dumps(local, indent=2))
    log.info("SHAP written; top features: %s", list(ranking.head(5).index))
    return payload


def load_global(directory: Path | None = None) -> dict:
    path = Path(directory or ARTIFACT_DIR) / GLOBAL_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def load_local(directory: Path | None = None) -> dict:
    path = Path(directory or ARTIFACT_DIR) / LOCAL_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def permutation_fallback(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> dict:
    """Model-agnostic importance, for the tiers TreeExplainer cannot handle."""
    from sklearn.inspection import permutation_importance
    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42, scoring="neg_root_mean_squared_error")
    except Exception as exc:                          # pragma: no cover
        log.warning("permutation importance failed: %s", exc)
        return {}
    ranking = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    return {k: float(v) for k, v in ranking.head(30).items()}
