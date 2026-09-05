"""Split integrity, metric behaviour and the promotion rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import Climatology, ColumnPersistence, SeasonalNaive
from src.models.evaluate import (
    assert_no_leakage, band_accuracy, cross_validate, holdout_split,
    regression_metrics, residual_quantiles, skill_score, walk_forward_splits,
)
from src.pipelines.train import should_promote


class TestSplits:
    def test_splits_are_chronological_and_gapped(self):
        idx = pd.date_range("2022-01-01", periods=1200, freq="D")
        splits = walk_forward_splits(idx)
        assert splits
        for s in splits:
            assert s.train_idx.max() < s.val_idx.min()
        assert_no_leakage(splits, horizon=3)

    def test_gap_is_at_least_the_horizon(self):
        idx = pd.date_range("2022-01-01", periods=1200, freq="D")
        splits = walk_forward_splits(idx, gap=3)
        for s in splits:
            assert int(s.val_idx.min() - s.train_idx.max() - 1) >= 3

    def test_a_zero_gap_configuration_is_rejected(self):
        idx = pd.date_range("2022-01-01", periods=1200, freq="D")
        splits = walk_forward_splits(idx, gap=0)
        with pytest.raises(AssertionError):
            assert_no_leakage(splits, horizon=3)

    def test_training_windows_expand(self):
        idx = pd.date_range("2022-01-01", periods=1200, freq="D")
        sizes = [len(s.train_idx) for s in walk_forward_splits(idx)]
        assert sizes == sorted(sizes)

    def test_short_series_degrades_rather_than_crashing(self):
        idx = pd.date_range("2024-01-01", periods=150, freq="D")
        assert walk_forward_splits(idx)

    def test_very_short_series_returns_nothing(self):
        assert walk_forward_splits(pd.date_range("2024-01-01", periods=20, freq="D")) == []

    def test_holdout_is_the_tail(self):
        df = pd.DataFrame({"x": range(500)},
                          index=pd.date_range("2024-01-01", periods=500, freq="D"))
        train, hold = holdout_split(df, 90)
        assert len(hold) == 90
        assert train.index.max() < hold.index.min()


class TestMetrics:
    def test_perfect_prediction(self):
        m = regression_metrics([10, 20, 30], [10, 20, 30])
        assert m["rmse"] == pytest.approx(0)
        assert m["r2"] == pytest.approx(1)

    def test_band_accuracy_uses_bands_not_values(self):
        # 45 and 49 are both Good; 45 and 60 are not.
        assert band_accuracy([45], [49]) == 1.0
        assert band_accuracy([45], [60]) == 0.0

    def test_skill_score_signs(self):
        assert skill_score(8, 10) == pytest.approx(0.2)
        assert skill_score(10, 10) == pytest.approx(0.0)
        assert skill_score(12, 10) < 0

    def test_residual_quantiles_span_zero(self):
        rng = np.random.default_rng(0)
        q = residual_quantiles(rng.normal(0, 10, 500))
        assert q["q10"] < 0 < q["q90"]


class TestBaselinesThroughCV:
    def test_persistence_runs_through_the_cv_harness(self):
        n = 400
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.default_rng(3)
        series = pd.Series(100 + np.cumsum(rng.normal(0, 4, n)), index=idx).clip(10, 400)
        X = pd.DataFrame({"aqi_mean": series, "day_of_year": idx.dayofyear}, index=idx)
        y = series.shift(-1).ffill()
        splits = walk_forward_splits(idx, n_splits=3, min_train=120, val_days=40, gap=1)
        result = cross_validate(lambda: ColumnPersistence("aqi_mean"), X, y, splits,
                                "persistence", "aqi_mean", 1)
        assert np.isfinite(result.metrics["rmse"])
        assert result.metrics["n_folds"] == 3

    def test_climatology_learns_the_seasonal_shape(self):
        idx = pd.date_range("2022-01-01", periods=1000, freq="D")
        seasonal = 100 + 60 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365.25)
        X = pd.DataFrame({"day_of_year": idx.dayofyear}, index=idx)
        model = Climatology().fit(X, pd.Series(seasonal, index=idx))
        pred = model.predict(X)
        assert np.corrcoef(pred, seasonal)[0, 1] > 0.95


class TestPromotionRule:
    def test_rejects_a_model_that_loses_to_persistence(self):
        report = {"aqi_mean|1": {"rmse": 20, "beats_persistence": True},
                  "aqi_mean|2": {"rmse": 30, "beats_persistence": False}}
        promote, reason = should_promote(report, {})
        assert not promote and "persistence" in reason

    def test_rejects_a_material_regression_against_the_incumbent(self):
        report = {"aqi_mean|1": {"rmse": 30, "beats_persistence": True}}
        promote, reason = should_promote(report, {"rmse_mean": 20})
        assert not promote and "incumbent" in reason

    def test_accepts_a_genuine_improvement(self):
        report = {"aqi_mean|1": {"rmse": 18, "beats_persistence": True},
                  "aqi_mean|2": {"rmse": 22, "beats_persistence": True}}
        promote, _ = should_promote(report, {"rmse_mean": 25})
        assert promote

    def test_rejects_an_empty_report(self):
        assert should_promote({}, {})[0] is False
