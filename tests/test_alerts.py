"""Alert logic: thresholds, the deterioration rule, and de-duplication."""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.alerts import build_alerts, deduplicate, severity_of


class TestSeverity:
    @pytest.mark.parametrize("aqi,expected", [
        (40, None), (120, None), (151, "unhealthy"),
        (205, "very_unhealthy"), (350, "hazardous"),
    ])
    def test_thresholds(self, aqi, expected):
        assert severity_of(aqi) == expected


def _forecast(values):
    idx = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])[:len(values)]
    return pd.DataFrame(
        {"aqi_max": values, "aqi_mean": [v * 0.9 for v in values],
         "horizon": list(range(1, len(values) + 1))},
        index=idx)


class TestBuildAlerts:
    def test_clean_air_produces_nothing(self):
        assert build_alerts(_forecast([40, 60, 80]), today_aqi=50) == []

    def test_high_forecast_produces_alerts(self):
        alerts = build_alerts(_forecast([160, 210, 330]), today_aqi=80)
        assert [a["severity"] for a in alerts] == [
            "unhealthy", "very_unhealthy", "hazardous"]

    def test_a_sustained_bad_spell_is_not_a_deterioration(self):
        """Already Unhealthy today, forecast stays Unhealthy -- not news."""
        alerts = build_alerts(_forecast([160, 165, 158]), today_aqi=170)
        assert all(not a["deteriorating"] for a in alerts)

    def test_worsening_within_a_bad_spell_is_a_deterioration(self):
        alerts = build_alerts(_forecast([160, 260, 165]), today_aqi=170)
        by_date = {a["target_date"]: a for a in alerts}
        assert by_date["2026-01-03"]["deteriorating"] is True

    def test_missing_values_are_skipped(self):
        assert build_alerts(_forecast([float("nan"), 200]), today_aqi=50)[0]["aqi"] == 200


class TestDeduplication:
    def test_same_severity_is_suppressed_on_a_second_pass(self):
        alerts = build_alerts(_forecast([210]), today_aqi=50)
        state = {"sent": {}}
        first = deduplicate(alerts, state)
        assert len(first) == 1
        state["sent"]["2026-01-02"] = "very_unhealthy"
        assert deduplicate(alerts, state) == []

    def test_an_escalation_is_not_suppressed(self):
        state = {"sent": {"2026-01-02": "unhealthy"}}
        alerts = build_alerts(_forecast([330]), today_aqi=50)
        assert len(deduplicate(alerts, state)) == 1

    def test_a_de_escalation_is_suppressed(self):
        state = {"sent": {"2026-01-02": "hazardous"}}
        alerts = build_alerts(_forecast([160]), today_aqi=50)
        assert deduplicate(alerts, state) == []
