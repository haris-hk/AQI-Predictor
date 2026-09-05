"""Baselines -- the bar every learned model has to clear.

AQI is highly persistent: tomorrow usually resembles today. Any model that
cannot beat that is not adding information, and a headline R-squared of 0.9
against a naive split is exactly what a persistence-dominated series produces.
These estimators are scikit-learn compatible so they run through the identical
cross-validation path as everything else -- same folds, same metrics, no
special-casing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ColumnPersistence(BaseEstimator, RegressorMixin):
    """y_hat(t+h) = x(t). Tomorrow is today.

    Falls back to the training mean when the column is missing or null, so a
    single bad row never produces a NaN forecast.
    """

    def __init__(self, column: str = "aqi_mean"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        self.fallback_ = float(np.nanmean(y)) if y is not None and len(y) else 100.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            return np.full(len(X), self.fallback_)
        return X[self.column].fillna(self.fallback_).to_numpy(dtype=float)


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """y_hat(t+h) = y(t+h-7). The same weekday one week earlier.

    Expressed exactly using the lag features rather than approximated, which is
    why LAG_DAYS carries 4, 5 and 6.
    """

    def __init__(self, base_column: str = "aqi_mean", horizon: int = 1, period: int = 7):
        self.base_column = base_column
        self.horizon = horizon
        self.period = period

    def fit(self, X: pd.DataFrame, y=None):
        self.fallback_ = float(np.nanmean(y)) if y is not None and len(y) else 100.0
        lag = self.period - self.horizon
        self.column_ = f"{self.base_column}_lag{lag}" if lag > 0 else self.base_column
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = self.column_ if self.column_ in X.columns else self.base_column
        if col not in X.columns:
            return np.full(len(X), self.fallback_)
        return X[col].fillna(self.fallback_).to_numpy(dtype=float)


class Climatology(BaseEstimator, RegressorMixin):
    """The historical average for this time of year, smoothed across day-of-year.

    Captures the seasonal cycle and nothing else. A model that beats persistence
    but loses to climatology has learned the calendar rather than the weather.
    """

    def __init__(self, doy_column: str = "day_of_year", window: int = 15):
        self.doy_column = doy_column
        self.window = window

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float))
        self.global_mean_ = float(y.mean())
        if self.doy_column not in X.columns:
            self.table_ = {}
            return self
        doy = pd.Series(np.asarray(X[self.doy_column]), dtype=float).round().astype(int)
        frame = pd.DataFrame({"doy": doy.to_numpy(), "y": y.to_numpy()})
        means = frame.groupby("doy")["y"].mean().reindex(range(1, 367))
        # Wrap the smoothing window around the year boundary.
        tripled = pd.concat([means, means, means]).interpolate(limit_direction="both")
        smoothed = tripled.rolling(self.window, center=True, min_periods=1).mean()
        self.table_ = smoothed.iloc[366:732].to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.doy_column not in X.columns or len(getattr(self, "table_", [])) == 0:
            return np.full(len(X), self.global_mean_)
        doy = pd.Series(np.asarray(X[self.doy_column]), dtype=float)
        doy = doy.fillna(1).round().clip(1, 366).astype(int).to_numpy()
        out = self.table_[doy - 1]
        return np.where(np.isfinite(out), out, self.global_mean_)


class DriftedPersistence(BaseEstimator, RegressorMixin):
    """Persistence plus the recent weekly trend, damped by horizon.

    A slightly stronger reference than plain persistence; if a learned model
    only beats naive persistence but not this, the gain is just trend-following.
    """

    def __init__(self, column: str = "aqi_mean", slope_column: str = "aqi_mean_slope7",
                 horizon: int = 1, damping: float = 0.5):
        self.column = column
        self.slope_column = slope_column
        self.horizon = horizon
        self.damping = damping

    def fit(self, X: pd.DataFrame, y=None):
        self.fallback_ = float(np.nanmean(y)) if y is not None and len(y) else 100.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            return np.full(len(X), self.fallback_)
        base = X[self.column].fillna(self.fallback_).to_numpy(dtype=float)
        if self.slope_column in X.columns:
            slope = X[self.slope_column].fillna(0.0).to_numpy(dtype=float)
            base = base + slope * self.horizon * self.damping
        return np.clip(base, 0, 500)


def baseline_factories(target: str, horizon: int) -> dict[str, callable]:
    """The reference set, built for one (target, horizon) pair."""
    return {
        "persistence": lambda: ColumnPersistence(column=target),
        "seasonal_naive": lambda: SeasonalNaive(base_column=target, horizon=horizon),
        "climatology": lambda: Climatology(),
        "drifted_persistence": lambda: DriftedPersistence(column=target, horizon=horizon),
    }
