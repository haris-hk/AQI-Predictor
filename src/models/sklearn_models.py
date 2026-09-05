"""The learned model ladder, tier 1 and tier 2.

Every estimator is wrapped in a pipeline that handles missing values, because
the feature frame legitimately contains them: lag-14 is undefined for the first
fortnight, and the API occasionally drops an hour. Imputing inside the pipeline
means the imputer is fitted on the training fold only, never on the validation
data -- the same discipline as the split itself.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def _linear(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def _tree(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def ridge(alpha: float = 10.0):
    return _linear(Ridge(alpha=alpha, random_state=RANDOM_SEED))


def elastic_net(alpha: float = 0.5, l1_ratio: float = 0.5):
    return _linear(ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                              random_state=RANDOM_SEED, max_iter=5000))


def random_forest(n_estimators: int = 400, max_depth: int | None = 14,
                  min_samples_leaf: int = 2):
    return _tree(RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, n_jobs=-1,
        random_state=RANDOM_SEED,
    ))


def extra_trees(n_estimators: int = 400, max_depth: int | None = 16):
    return _tree(ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1,
        random_state=RANDOM_SEED,
    ))


def hist_gradient_boosting(max_iter: int = 400, learning_rate: float = 0.05,
                           max_depth: int | None = 6, min_samples_leaf: int = 15,
                           l2_regularization: float = 1.0):
    """Handles NaN natively, so it gets no imputer -- letting it use missingness
    as a signal rather than papering over it with a median."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        early_stopping=True, validation_fraction=0.15, random_state=RANDOM_SEED,
    )


def quantile_gbm(quantile: float = 0.9, max_iter: int = 300):
    """Direct quantile regression, for prediction intervals that widen where the
    model is genuinely less certain instead of applying one global band."""
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=max_iter,
        learning_rate=0.05, max_depth=6, random_state=RANDOM_SEED,
        early_stopping=True, validation_fraction=0.15,
    )


def model_factories() -> dict[str, callable]:
    return {
        "ridge": ridge,
        "elastic_net": elastic_net,
        "random_forest": random_forest,
        "extra_trees": extra_trees,
        "hist_gradient_boosting": hist_gradient_boosting,
    }


# Small, fixed randomised search spaces. Kept deliberately modest: with roughly
# 1,500 daily rows an aggressive search overfits the validation folds, and the
# gain is smaller than the gain from one more good feature.
SEARCH_SPACES: dict[str, list[dict]] = {
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 50.0, 200.0)],
    "random_forest": [
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 4},
        {"n_estimators": 400, "max_depth": 14, "min_samples_leaf": 2},
        {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 3},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.05, "max_depth": 4, "max_iter": 400, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_depth": 6, "max_iter": 400, "l2_regularization": 1.0},
        {"learning_rate": 0.03, "max_depth": 8, "max_iter": 600, "l2_regularization": 5.0},
    ],
}
