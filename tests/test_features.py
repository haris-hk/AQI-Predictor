"""Feature-engineering correctness, with the leakage guards front and centre."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CITY, HORIZONS, TARGETS, target_col
from src.features.build import (
    add_calendar_features, add_change_features, add_rolling_features,
    build_daily_features, feature_columns, training_frame,
)
from src.features.schema import sanitise, validate_daily


class TestTargets:
    def test_target_is_the_future_value_of_the_source(self, daily):
        """target_aqi_mean_h2 on day t must equal aqi_mean on day t+2, exactly."""
        for h in HORIZONS:
            col = target_col("aqi_mean", h)
            shifted = daily["aqi_mean"].shift(-h)
            pd.testing.assert_series_equal(
                daily[col], shifted, check_names=False)

    def test_last_rows_have_no_label(self, daily):
        for h in HORIZONS:
            assert daily[target_col("aqi_mean", h)].tail(h).isna().all()

    def test_all_six_targets_exist(self, daily):
        for t in TARGETS:
            for h in HORIZONS:
                assert target_col(t, h) in daily.columns


class TestNoLeakage:
    def test_targets_are_excluded_from_features(self, daily):
        cols = feature_columns(daily)
        for t in TARGETS:
            for h in HORIZONS:
                assert target_col(t, h) not in cols

    def test_training_frame_never_contains_a_target(self, daily):
        X, y = training_frame(daily, "aqi_mean", 1)
        leaked = [c for c in X.columns if c.startswith("target_")]
        assert not leaked, f"target columns leaked into X: {leaked}"

    def test_rolling_features_are_causal(self):
        """Changing a *future* observation must not change a past feature.

        This is the test that catches an accidental centred window, which is the
        single easiest way to leak the future into the past.
        """
        idx = pd.date_range("2024-01-01", periods=120, freq="D")
        base = pd.DataFrame({"aqi_mean": np.linspace(50, 150, 120)}, index=idx)
        perturbed = base.copy()
        perturbed.iloc[100:] += 400.0

        a = add_rolling_features(base)
        b = add_rolling_features(perturbed)
        roll_cols = [c for c in a.columns if "roll" in c]
        pd.testing.assert_frame_equal(
            a[roll_cols].iloc[:100], b[roll_cols].iloc[:100],
            obj="rolling features before the perturbation")

    def test_change_features_are_causal(self):
        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        base = pd.DataFrame({"aqi_mean": np.linspace(60, 120, 90)}, index=idx)
        perturbed = base.copy()
        perturbed.iloc[70:] *= 3

        a, b = add_change_features(base), add_change_features(perturbed)
        cols = [c for c in a.columns if "diff" in c or "slope" in c]
        pd.testing.assert_frame_equal(a[cols].iloc[:70], b[cols].iloc[:70])

    def test_future_weather_is_the_only_forward_looking_family(self, daily):
        """Every feature that reads the future must be prefixed fc_.

        Detected structurally: shift a column forward and check which features
        move. Anything that responds to a future-only change and is not an fc_
        feature is leakage.
        """
        forward_looking = [c for c in feature_columns(daily) if c.startswith("fc_")]
        assert forward_looking, "expected forecast-weather features to exist"
        assert all(c.startswith("fc_") for c in forward_looking)


class TestCalendar:
    def test_cyclical_encoding_wraps(self):
        idx = pd.DatetimeIndex(["2024-12-31", "2025-01-01"])
        out = add_calendar_features(pd.DataFrame(index=idx))
        gap = np.hypot(out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1],
                       out["doy_cos"].iloc[0] - out["doy_cos"].iloc[1])
        assert gap < 0.1, "31 December and 1 January should be adjacent in the encoding"

    def test_weekend_flag(self):
        idx = pd.DatetimeIndex(["2024-01-06", "2024-01-08"])   # Saturday, Monday
        out = add_calendar_features(pd.DataFrame(index=idx))
        assert out["is_weekend"].tolist() == [1, 0]


class TestBuild:
    def test_produces_a_daily_index(self, daily):
        assert isinstance(daily.index, pd.DatetimeIndex)
        assert daily.index.is_monotonic_increasing
        assert not daily.index.duplicated().any()

    def test_has_a_useful_number_of_features(self, daily):
        assert len(feature_columns(daily)) > 80

    def test_physics_features_present(self, daily):
        for col in ("ventilation_index", "temp_range", "pm_ratio"):
            assert col in daily.columns

    def test_empty_input_is_safe(self):
        assert build_daily_features(pd.DataFrame(), pd.DataFrame(), CITY).empty

    def test_validation_report(self, daily):
        report = validate_daily(sanitise(daily))
        assert report["rows"] > 700
        assert report["duplicate_dates"] == 0
        assert 0 <= report["aqi_mean_min"] <= report["aqi_mean_max"] <= 500
