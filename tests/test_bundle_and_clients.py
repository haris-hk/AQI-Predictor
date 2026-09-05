"""Serving-side integrity and API response parsing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clients.open_meteo import _hourly_frame, date_chunks
from src.config import HORIZONS, band_for, band_index
from src.models.bundle import ModelBundle


class _Echo:
    """Returns the first column, so alignment errors are visible in the output."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X.iloc[:, 0], dtype=float)


def _bundle():
    b = ModelBundle(feature_names=["a", "b", "c"])
    for h in HORIZONS:
        b.models[("aqi_mean", h)] = _Echo()
        b.models[("aqi_max", h)] = _Echo()
        b.residual_quantiles[f"aqi_mean|{h}"] = {"q10": -12.0, "q90": 15.0}
    return b


class TestBundleAlignment:
    def test_reorders_columns_to_the_training_order(self):
        b = _bundle()
        X = pd.DataFrame({"c": [3.0], "b": [2.0], "a": [1.0]})
        assert list(b.align(X).columns) == ["a", "b", "c"]
        assert b.align(X).iloc[0, 0] == 1.0

    def test_missing_columns_become_nan_rather_than_shifting_the_matrix(self):
        b = _bundle()
        X = pd.DataFrame({"a": [1.0], "c": [3.0]})
        aligned = b.align(X)
        assert list(aligned.columns) == ["a", "b", "c"]
        assert pd.isna(aligned["b"].iloc[0])

    def test_extra_columns_are_dropped(self):
        b = _bundle()
        X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "surprise": [9.0]})
        assert "surprise" not in b.align(X).columns


class TestBundlePrediction:
    def test_produces_one_row_per_horizon_with_dates_and_bands(self):
        b = _bundle()
        idx = pd.date_range("2026-01-01", periods=3, freq="D")
        X = pd.DataFrame({"a": [160.0] * 3, "b": [1.0] * 3, "c": [2.0] * 3}, index=idx)
        out = b.predict_row(X)
        assert len(out) == len(HORIZONS)
        # The forecast origin is the LAST row of X (2026-01-03), so the target
        # dates are the three days after it.
        assert list(out.index) == [pd.Timestamp("2026-01-04"),
                                   pd.Timestamp("2026-01-05"),
                                   pd.Timestamp("2026-01-06")]
        assert out["as_of_date"].iloc[0] == pd.Timestamp("2026-01-03")
        assert out["band"].iloc[0] == "Unhealthy"

    def test_intervals_bracket_the_point_forecast(self):
        b = _bundle()
        idx = pd.date_range("2026-01-01", periods=2, freq="D")
        X = pd.DataFrame({"a": [100.0] * 2, "b": [0.0] * 2, "c": [0.0] * 2}, index=idx)
        out = b.predict_row(X)
        row = out.iloc[0]
        assert row["aqi_mean_lower"] < row["aqi_mean"] < row["aqi_mean_upper"]

    def test_predictions_are_clipped_to_the_valid_range(self):
        b = _bundle()
        idx = pd.date_range("2026-01-01", periods=2, freq="D")
        X = pd.DataFrame({"a": [-50.0, 9000.0], "b": [0.0] * 2, "c": [0.0] * 2}, index=idx)
        out = b.predict_row(X)
        assert 0 <= out["aqi_mean"].iloc[0] <= 500

    def test_roundtrip_through_disk(self, tmp_path):
        b = _bundle()
        b.metadata = {"trained_at": "2026-01-01T00:00:00Z"}
        b.save(tmp_path)
        loaded = ModelBundle.load(tmp_path)
        assert loaded is not None
        assert loaded.feature_names == b.feature_names
        assert set(loaded.models) == set(b.models)

    def test_load_from_an_empty_directory_returns_none(self, tmp_path):
        assert ModelBundle.load(tmp_path) is None


class TestResponseParsing:
    def test_hourly_block_becomes_a_utc_indexed_frame(self):
        payload = {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                              "pm2_5": [35.5, 40.1], "us_aqi": [100, 112]}}
        df = _hourly_frame(payload)
        assert df.index.name == "ts_utc"
        assert str(df.index.tz) == "UTC"
        assert df["pm2_5"].tolist() == [35.5, 40.1]

    def test_nulls_survive_as_nan(self):
        payload = {"hourly": {"time": ["2024-01-01T00:00"], "pm2_5": [None]}}
        assert pd.isna(_hourly_frame(payload)["pm2_5"].iloc[0])

    def test_an_empty_payload_is_safe(self):
        assert _hourly_frame({}).empty
        assert _hourly_frame({"hourly": {}}).empty

    def test_chunks_are_contiguous_and_inclusive(self):
        from datetime import date, timedelta
        chunks = list(date_chunks(date(2022, 6, 1), date(2023, 6, 1), 90))
        assert chunks[0][0] == date(2022, 6, 1)
        assert chunks[-1][1] == date(2023, 6, 1)
        for a, b in zip(chunks, chunks[1:]):
            assert b[0] == a[1] + timedelta(days=1)


class TestBands:
    @pytest.mark.parametrize("aqi,label", [
        (0, "Good"), (50, "Good"), (51, "Moderate"), (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"), (250, "Very Unhealthy"), (400, "Hazardous"), (900, "Hazardous"),
    ])
    def test_band_boundaries(self, aqi, label):
        assert band_for(aqi).label == label

    def test_band_index_is_monotonic(self):
        values = [10, 60, 120, 180, 250, 400]
        assert [band_index(v) for v in values] == sorted(band_index(v) for v in values)
