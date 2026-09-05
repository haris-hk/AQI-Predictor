"""Evaluation protocol.

The single most important file in the modelling code, because it is where this
kind of project usually goes wrong. Hourly AQI is strongly autocorrelated; a
random split puts hour 14 in train and hour 15 in test and reports a
magnificent R-squared for a model that has learned nothing. Everything here is
built to prevent that:

  * expanding-window walk-forward splits, never shuffled
  * a gap of at least `horizon` days between train end and validation start, so
    the last training row's label cannot fall inside the validation window
  * a chronological holdout that the search never sees
  * a skill score against persistence reported next to every headline metric
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import AQI_BANDS, CV_GAP_DAYS, CV_MIN_TRAIN_DAYS, CV_N_SPLITS, CV_VAL_DAYS, band_index

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- metrics
def band_accuracy(y_true, y_pred) -> float:
    """Share of days placed in the correct AQI band.

    This is what a dashboard reader actually consumes -- "Unhealthy tomorrow"
    rather than "168.4 tomorrow" -- so it belongs next to RMSE, not below it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    t = np.array([band_index(v) for v in y_true])
    p = np.array([band_index(v) for v in y_pred])
    return float((t == p).mean())


def band_within_one(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    t = np.array([band_index(v) for v in y_true])
    p = np.array([band_index(v) for v in y_pred])
    return float((np.abs(t - p) <= 1).mean())


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {k: float("nan") for k in
                ("rmse", "mae", "r2", "bias", "band_accuracy", "band_within_one", "n")}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "band_accuracy": band_accuracy(y_true, y_pred),
        "band_within_one": band_within_one(y_true, y_pred),
        "n": int(len(y_true)),
    }


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """1 means perfect, 0 means no better than the reference, negative is worse.

    The reference is persistence. A model that cannot clear 0 here has not
    earned deployment however good its R-squared looks.
    """
    if not np.isfinite(rmse_reference) or rmse_reference == 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_reference)


# -------------------------------------------------------------------- splits
@dataclass
class Split:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = CV_N_SPLITS,
                        min_train: int = CV_MIN_TRAIN_DAYS, val_days: int = CV_VAL_DAYS,
                        gap: int = CV_GAP_DAYS) -> list[Split]:
    """Expanding-window splits, oldest data always in train.

    Shrinks the requested configuration rather than failing when the series is
    short, and returns an empty list only when even one honest fold is
    impossible.
    """
    n = len(index)
    if n < 30:
        return []
    while n_splits > 1 and min_train + gap + val_days * n_splits > n:
        if val_days > 14:
            val_days = max(14, val_days // 2)
        elif min_train > 60:
            min_train = max(60, int(min_train * 0.6))
        else:
            n_splits -= 1
    if min_train + gap + val_days > n:
        min_train = max(30, n - gap - val_days)
        if min_train + gap + val_days > n:
            return []

    splits: list[Split] = []
    for i in range(n_splits):
        val_end = n - (n_splits - 1 - i) * val_days
        val_start = val_end - val_days
        train_end = val_start - gap
        if train_end < min_train or val_start >= val_end:
            continue
        splits.append(Split(
            fold=len(splits),
            train_idx=np.arange(0, train_end),
            val_idx=np.arange(val_start, val_end),
            train_end=index[train_end - 1],
            val_start=index[val_start],
            val_end=index[val_end - 1],
        ))
    return splits


def holdout_split(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological holdout. Touched once, at the very end, never in search."""
    if len(df) <= holdout_days + 30:
        cut = max(1, int(len(df) * 0.8))
        return df.iloc[:cut], df.iloc[cut:]
    return df.iloc[:-holdout_days], df.iloc[-holdout_days:]


def assert_no_leakage(splits: Sequence[Split], horizon: int) -> None:
    """Guard rail invoked by the tests and by the training pipeline."""
    for s in splits:
        assert s.train_idx.max() < s.val_idx.min(), "train overlaps validation"
        gap_days = int(s.val_idx.min() - s.train_idx.max() - 1)
        assert gap_days >= horizon, (
            f"fold {s.fold}: gap of {gap_days} days is smaller than the "
            f"{horizon}-day horizon, so training labels leak into validation"
        )


# ------------------------------------------------------------------ CV driver
@dataclass
class CVResult:
    name: str
    target: str
    horizon: int
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    residuals: np.ndarray | None = None

    def row(self, extra: dict | None = None) -> dict:
        row = {"model": self.name, "target": self.target, "horizon": self.horizon}
        row |= {k: v for k, v in self.metrics.items()}
        row |= (extra or {})
        return row


def cross_validate(model_factory: Callable[[], object], X: pd.DataFrame, y: pd.Series,
                   splits: Sequence[Split], name: str, target: str, horizon: int
                   ) -> CVResult:
    """Fit a fresh model on each expanding window and score the next block."""
    fold_metrics: list[dict[str, float]] = []
    residuals: list[np.ndarray] = []

    for split in splits:
        X_tr, y_tr = X.iloc[split.train_idx], y.iloc[split.train_idx]
        X_va, y_va = X.iloc[split.val_idx], y.iloc[split.val_idx]
        if len(X_tr) < 20 or len(X_va) < 5:
            continue
        try:
            model = model_factory()
            model.fit(X_tr, y_tr)
            pred = np.asarray(model.predict(X_va), dtype=float)
        except Exception as exc:                      # pragma: no cover
            log.warning("%s failed on fold %d: %s", name, split.fold, exc)
            continue
        m = regression_metrics(y_va, pred)
        m["fold"] = split.fold
        fold_metrics.append(m)
        residuals.append(np.asarray(y_va, float) - pred)

    if not fold_metrics:
        return CVResult(name, target, horizon,
                        {k: float("nan") for k in ("rmse", "mae", "r2")}, [])

    keys = ("rmse", "mae", "r2", "bias", "band_accuracy", "band_within_one")
    agg = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys}
    agg["rmse_std"] = float(np.nanstd([f["rmse"] for f in fold_metrics]))
    agg["n_folds"] = len(fold_metrics)
    agg["n"] = int(sum(f["n"] for f in fold_metrics))
    return CVResult(name, target, horizon, agg, fold_metrics,
                    np.concatenate(residuals) if residuals else None)


def residual_quantiles(residuals: np.ndarray | None,
                       levels=(0.1, 0.25, 0.75, 0.9)) -> dict[str, float]:
    """Empirical prediction intervals.

    A bare point forecast of "AQI 168" claims a precision the model does not
    have. The dashboard shows a band, built from the out-of-fold residual
    distribution for that horizon.
    """
    if residuals is None or len(residuals) < 20:
        return {}
    finite = residuals[np.isfinite(residuals)]
    return {f"q{int(q * 100)}": float(np.quantile(finite, q)) for q in levels}
