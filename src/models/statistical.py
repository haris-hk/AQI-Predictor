"""Tier 3: classical time-series models.

The brief asks for a range from statistical modelling to deep learning. SARIMAX
is the statistical leg. It is fitted on the AQI series itself and updated
observation by observation, forecasting `horizon` steps from each origin --
which is exactly how it would run in production, rather than being handed the
whole validation window at once.

statsmodels is an optional dependency. If it is absent the factory returns
nothing and the training pipeline records the model as skipped rather than
failing the run.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

log = logging.getLogger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:                                   # pragma: no cover
    HAS_STATSMODELS = False


class SarimaxForecaster(BaseEstimator, RegressorMixin):
    """SARIMAX on the base AQI series, forecasting `horizon` days ahead.

    Assumes X is chronologically ordered with a DatetimeIndex, which the
    walk-forward splitter guarantees. Weekly seasonality is the default period:
    the annual cycle is left to the calendar features in the tabular models
    rather than asked of a model fitted on a few hundred daily points.
    """

    def __init__(self, base_column: str = "aqi_mean", horizon: int = 1,
                 order=(2, 0, 2), seasonal_order=(1, 0, 1, 7), trend: str = "c"):
        self.base_column = base_column
        self.horizon = horizon
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, X: pd.DataFrame, y=None):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for SarimaxForecaster")
        self.fallback_ = float(np.nanmean(y)) if y is not None and len(y) else 100.0
        series = pd.Series(
            pd.to_numeric(X[self.base_column], errors="coerce").to_numpy(dtype=float)
        ).interpolate(limit_direction="both")
        self.history_ = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.result_ = SARIMAX(
                    series, order=self.order, seasonal_order=self.seasonal_order,
                    trend=self.trend, enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=200)
            except Exception as exc:                  # pragma: no cover
                log.warning("SARIMAX fit failed (%s); falling back to persistence", exc)
                self.result_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = pd.to_numeric(X[self.base_column], errors="coerce").fillna(
            self.fallback_).to_numpy(dtype=float)
        if getattr(self, "result_", None) is None:
            return base
        preds = np.empty(len(X), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = self.result_
            for i, observed in enumerate(base):
                try:
                    preds[i] = float(state.forecast(steps=self.horizon)[-1])
                except Exception:                     # pragma: no cover
                    preds[i] = observed
                try:
                    # Extend the filter with the newly observed day without
                    # re-estimating parameters -- the production update path.
                    state = state.append([observed], refit=False)
                except Exception:                     # pragma: no cover
                    pass
        return np.clip(np.nan_to_num(preds, nan=self.fallback_), 0, 500)


def statistical_factories(target: str, horizon: int) -> dict[str, callable]:
    if not HAS_STATSMODELS:
        log.info("statsmodels unavailable; skipping the statistical tier")
        return {}
    return {
        "sarimax": lambda: SarimaxForecaster(base_column=target, horizon=horizon),
    }
