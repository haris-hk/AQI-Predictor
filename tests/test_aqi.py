"""AQI correctness tests, against hand-computed EPA worked examples."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features import aqi


class TestBreakpointTables:
    def test_tables_are_contiguous_and_ordered(self):
        for pollutant, table in aqi.BREAKPOINTS.items():
            for i in range(len(table) - 1):
                _, c_hi, _, i_hi = table[i]
                c_lo_next, _, i_lo_next, _ = table[i + 1]
                assert c_lo_next > c_hi, f"{pollutant} concentration bands overlap"
                assert i_lo_next == i_hi + 1, f"{pollutant} index bands are not contiguous"

    def test_every_pollutant_has_truncation_and_window(self):
        for pollutant in aqi.BREAKPOINTS:
            assert pollutant in aqi.TRUNCATION
            assert pollutant in aqi.AVERAGING_HOURS


class TestSubIndex:
    @pytest.mark.parametrize("conc,expected", [
        (0.0, 0.0),
        (9.0, 50.0),        # exact top of the Good band under the 2024 revision
        (9.1, 51.0),        # exact bottom of Moderate
        (35.4, 100.0),
        (35.5, 101.0),
        (225.5, 301.0),
    ])
    def test_pm25_breakpoint_edges(self, conc, expected):
        assert aqi.sub_index(conc, "pm2_5") == pytest.approx(expected, abs=0.01)

    def test_pm25_interpolation_worked_example(self):
        # 35.9 ug/m3 -> (150-101)/(55.4-35.5)*(35.9-35.5)+101
        assert aqi.sub_index(35.9, "pm2_5") == pytest.approx(101.985, abs=0.01)

    def test_ozone_interpolation_worked_example(self):
        # 0.078 ppm -> (150-101)/(0.085-0.071)*(0.078-0.071)+101 = 125.5
        assert aqi.sub_index(0.078, "ozone") == pytest.approx(125.5, abs=0.01)

    def test_co_interpolation_worked_example(self):
        # 5.0 ppm -> (100-51)/(9.4-4.5)*(5.0-4.5)+51 = 56.0
        assert aqi.sub_index(5.0, "carbon_monoxide") == pytest.approx(56.0, abs=0.01)

    def test_truncation_changes_the_band(self):
        # 9.19 truncates to 9.1, which sits at the bottom of Moderate, not above it.
        assert aqi.sub_index(9.19, "pm2_5") == pytest.approx(51.0, abs=0.01)

    def test_above_table_clamps_to_500(self):
        assert aqi.sub_index(9_999.0, "pm2_5") == 500.0

    def test_missing_is_nan(self):
        assert math.isnan(aqi.sub_index(float("nan"), "pm2_5"))
        assert math.isnan(aqi.sub_index(None, "pm2_5"))

    def test_unknown_pollutant_raises(self):
        with pytest.raises(KeyError):
            aqi.sub_index(1.0, "unobtainium")


class TestUnitConversion:
    def test_ozone_ugm3_to_ppb(self):
        # 100 ug/m3 O3 -> 100 * 24.45/48 = 50.94 ppb
        assert aqi.ugm3_to_ppb(100.0, "ozone") == pytest.approx(50.94, abs=0.01)

    def test_co_ugm3_to_ppm(self):
        # 1000 ug/m3 CO -> 1000 * 24.45/28.01 / 1000 = 0.873 ppm
        assert aqi.ugm3_to_ppm(1000.0, "carbon_monoxide") == pytest.approx(0.8729, abs=0.001)

    def test_particulates_are_not_converted(self):
        assert aqi.to_epa_units(42.0, "pm2_5") == 42.0

    def test_ozone_uses_ppm_not_ppb(self):
        # Regression guard. The ozone breakpoint table is in ppm, like CO, not
        # ppb like NO2 and SO2. Converting ozone to ppb inflates its sub-index
        # a thousandfold and makes ozone dominant on every row.
        assert aqi.EPA_UNITS["ozone"] == "ppm"
        assert aqi.to_epa_units(150.0, "ozone") == pytest.approx(0.0764, abs=0.0005)
        assert aqi.sub_index(aqi.to_epa_units(150.0, "ozone"), "ozone") < 120

    def test_every_pollutant_declares_its_epa_unit(self):
        assert set(aqi.EPA_UNITS) == set(aqi.BREAKPOINTS)


class TestRollingAverage:
    def test_incomplete_window_returns_nan(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        s = pd.Series(np.arange(10, dtype=float), index=idx)
        out = aqi.rolling_average(s, hours=24, min_fraction=0.75)
        assert out.isna().all(), "a 24h mean must not be reported from 10 hours"

    def test_complete_window_averages(self):
        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        s = pd.Series(np.full(48, 20.0), index=idx)
        out = aqi.rolling_average(s, hours=24)
        assert out.iloc[-1] == pytest.approx(20.0)

    def test_one_hour_window_is_identity(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        s = pd.Series([1.0, 2, 3, 4, 5], index=idx)
        pd.testing.assert_series_equal(aqi.rolling_average(s, 1), s.astype(float))


class TestComputeHourlyAQI:
    @staticmethod
    def _frame(hours=72, pm25=30.0):
        idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
        return pd.DataFrame({
            "pm2_5": np.full(hours, pm25),
            "pm10": np.full(hours, 60.0),
            "ozone": np.full(hours, 60.0),
            "nitrogen_dioxide": np.full(hours, 30.0),
            "sulphur_dioxide": np.full(hours, 10.0),
            "carbon_monoxide": np.full(hours, 300.0),
        }, index=idx)

    def test_produces_index_and_dominant_pollutant(self):
        out = aqi.compute_hourly_aqi(self._frame())
        assert "us_aqi_epa" in out and "dominant_pollutant" in out
        tail = out.iloc[-1]
        assert tail["us_aqi_epa"] > 0
        assert tail["dominant_pollutant"] in aqi.POLLUTANTS

    def test_overall_index_is_the_max_subindex(self):
        out = aqi.compute_hourly_aqi(self._frame())
        sub_cols = [c for c in out.columns if c.startswith("sub_")]
        row = out.iloc[-1]
        assert row["us_aqi_epa"] == pytest.approx(row[sub_cols].max())

    def test_high_pm25_dominates(self):
        out = aqi.compute_hourly_aqi(self._frame(pm25=180.0))
        assert out.iloc[-1]["dominant_pollutant"] == "pm2_5"
        assert out.iloc[-1]["us_aqi_epa"] > 200

    def test_empty_input_is_safe(self):
        assert aqi.compute_hourly_aqi(pd.DataFrame()).empty


class TestDailyAQI:
    def test_local_calendar_grouping(self):
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
        s = pd.Series(np.arange(72, dtype=float), index=idx)
        daily = aqi.daily_aqi(s, "Asia/Karachi")
        assert {"aqi_mean", "aqi_max", "hours_observed"} <= set(daily.columns)
        assert daily["hours_observed"].sum() == 72
