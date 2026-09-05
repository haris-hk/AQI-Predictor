"""Tier 4: neural models.

Included because the brief asks for the statistical-to-deep-learning range, and
kept honest about what it can do. With roughly 1,500 daily rows an LSTM is
being asked to learn from very little; gradient boosting on good features is
the expected winner. Reporting that the deep model was built, evaluated on the
identical folds and lost is a stronger result than quietly omitting it.

The LSTM is given a genuine sequence rather than the same flat feature vector,
so the comparison is fair to it: it sees `seq_len` days of the raw daily series
and has to learn the temporal structure that the tabular models get handed as
explicit lag features.

torch is optional. Absent, the factory returns nothing and the run records the
tier as skipped.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.config import RANDOM_SEED

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:                                   # pragma: no cover
    HAS_TORCH = False

SEQUENCE_COLUMNS = [
    "aqi_mean", "aqi_max", "pm2_5_mean", "pm10_mean",
    "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "boundary_layer_height_mean", "doy_sin", "doy_cos",
]


def _standardise(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (a - mean) / np.where(std > 1e-8, std, 1.0)


class _TorchRegressor(BaseEstimator, RegressorMixin):
    """Shared training loop: Adam, early stopping on a chronological tail split."""

    def __init__(self, epochs: int = 200, lr: float = 1e-3, batch_size: int = 64,
                 patience: int = 20, hidden: int = 64, dropout: float = 0.2,
                 verbose: bool = False):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.hidden = hidden
        self.dropout = dropout
        self.verbose = verbose

    def _build(self, n_features: int):
        raise NotImplementedError

    def _prepare(self, X, y=None, fitting=False):
        raise NotImplementedError

    def _train(self, xt: "torch.Tensor", yt: "torch.Tensor"):
        torch.manual_seed(RANDOM_SEED)
        n = len(xt)
        # Chronological validation tail -- never a random split, for the same
        # reason the outer CV is never random.
        cut = max(1, int(n * 0.85))
        x_tr, y_tr, x_va, y_va = xt[:cut], yt[:cut], xt[cut:], yt[cut:]
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        best, best_state, bad = float("inf"), None, 0

        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(x_tr))
            for i in range(0, len(x_tr), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(x_tr[idx]).squeeze(-1), y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                if len(x_va):
                    val = float(loss_fn(self.net_(x_va).squeeze(-1), y_va))
                else:
                    val = float(loss_fn(self.net_(x_tr).squeeze(-1), y_tr))
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()


class MLPForecaster(_TorchRegressor):
    """Feed-forward network on the same tabular features the tree models see."""

    def _build(self, n_features: int):
        return nn.Sequential(
            nn.Linear(n_features, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden // 2, 1),
        )

    def _prepare(self, X: pd.DataFrame, y=None, fitting=False):
        arr = X.to_numpy(dtype=float)
        if fitting:
            self.x_mean_ = np.nanmean(arr, axis=0)
            self.x_std_ = np.nanstd(arr, axis=0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = _standardise(arr, np.nan_to_num(self.x_mean_), np.nan_to_num(self.x_std_, nan=1.0))
        return np.nan_to_num(arr, nan=0.0)

    def fit(self, X: pd.DataFrame, y):
        if not HAS_TORCH:
            raise ImportError("torch is required for MLPForecaster")
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = float(np.nanmean(y)), float(np.nanstd(y) or 1.0)
        arr = self._prepare(X, fitting=True)
        self.net_ = self._build(arr.shape[1])
        self._train(torch.tensor(arr, dtype=torch.float32),
                    torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        arr = self._prepare(X)
        with torch.no_grad():
            out = self.net_(torch.tensor(arr, dtype=torch.float32)).squeeze(-1).numpy()
        return np.clip(out * self.y_std_ + self.y_mean_, 0, 500)


class LSTMForecaster(_TorchRegressor):
    """LSTM over a rolling window of the raw daily series.

    Sequences are reconstructed from consecutive rows of X, which is valid
    because the walk-forward splitter always hands over contiguous, ordered
    blocks. The first `seq_len - 1` rows of any block have no full history, so
    their windows are left-padded with the earliest available row.
    """

    def __init__(self, seq_len: int = 21, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def _build(self, n_features: int):
        class Net(nn.Module):
            def __init__(self, n_in, hidden, dropout):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :]))

        return Net(n_features, self.hidden, self.dropout)

    def _columns(self, X: pd.DataFrame) -> list[str]:
        return [c for c in SEQUENCE_COLUMNS if c in X.columns] or list(X.columns[:8])

    def _sequences(self, X: pd.DataFrame, fitting=False) -> np.ndarray:
        cols = self.cols_ if not fitting else self._columns(X)
        if fitting:
            self.cols_ = cols
        arr = X[cols].to_numpy(dtype=float)
        arr = np.nan_to_num(arr, nan=np.nan)
        if fitting:
            self.x_mean_ = np.nanmean(arr, axis=0)
            self.x_std_ = np.nanstd(arr, axis=0)
        arr = np.where(np.isfinite(arr), arr, np.nan_to_num(self.x_mean_))
        arr = _standardise(arr, np.nan_to_num(self.x_mean_), np.nan_to_num(self.x_std_, nan=1.0))
        arr = np.nan_to_num(arr, nan=0.0)

        pad = np.repeat(arr[:1], self.seq_len - 1, axis=0)
        padded = np.vstack([pad, arr])
        windows = np.stack([padded[i:i + self.seq_len] for i in range(len(arr))])
        return windows

    def fit(self, X: pd.DataFrame, y):
        if not HAS_TORCH:
            raise ImportError("torch is required for LSTMForecaster")
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = float(np.nanmean(y)), float(np.nanstd(y) or 1.0)
        seq = self._sequences(X, fitting=True)
        self.net_ = self._build(seq.shape[2])
        self._train(torch.tensor(seq, dtype=torch.float32),
                    torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        seq = self._sequences(X)
        with torch.no_grad():
            out = self.net_(torch.tensor(seq, dtype=torch.float32)).squeeze(-1).numpy()
        return np.clip(out * self.y_std_ + self.y_mean_, 0, 500)


def deep_factories(target: str, horizon: int) -> dict[str, callable]:
    if not HAS_TORCH:
        log.info("torch unavailable; skipping the deep-learning tier")
        return {}
    return {
        "mlp": lambda: MLPForecaster(epochs=200, hidden=64),
        "lstm": lambda: LSTMForecaster(seq_len=21, epochs=150, hidden=48),
    }
