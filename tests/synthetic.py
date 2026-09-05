"""Synthetic Karachi-like data.

The live APIs are unreachable from the development sandbox, so the entire chain
is verified against a generator that reproduces the structure that matters:

  * a strong winter pollution season (November to February)
  * a diurnal emissions cycle and a weekday/weekend difference
  * dispersion driven by wind speed and mixing-layer depth, so meteorology
    carries genuine predictive information beyond persistence
  * autocorrelated synoptic weather, so the series is persistent but not
    trivially so
  * occasional dust events and rain washout

This is a test fixture, not a data source. It exists so that a broken feature
transform or a leaking split fails here rather than in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_hourly(days: int = 1460, seed: int = 7, start: str = "2022-06-01"
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    dow = idx.dayofweek.to_numpy()

    # --- synoptic weather: red noise, so weather persists for days ----------
    def red_noise(scale: float, rho: float = 0.995) -> np.ndarray:
        e = rng.normal(0, scale, n)
        out = np.empty(n)
        out[0] = e[0]
        for i in range(1, n):
            out[i] = rho * out[i - 1] + e[i]
        return out

    synoptic = red_noise(1.0)
    synoptic /= (synoptic.std() or 1.0)

    season = np.cos(2 * np.pi * (doy - 15) / 365.25)          # +1 in mid-January

    temperature = 27 + 7 * np.sin(2 * np.pi * (doy - 110) / 365.25) \
        + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + 1.5 * synoptic + rng.normal(0, 0.8, n)
    wind = np.clip(9 + 4.5 * np.sin(2 * np.pi * (doy - 170) / 365.25)
                   + 2.0 * np.sin(2 * np.pi * (hour - 14) / 24)
                   + 3.2 * synoptic + rng.normal(0, 1.0, n), 0.3, None)
    blh = np.clip(500 + 550 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
                  + 220 * (1 - season) + 160 * synoptic + rng.normal(0, 70, n), 60, None)
    humidity = np.clip(62 - 0.8 * (temperature - 27) + 8 * synoptic + rng.normal(0, 6, n), 8, 100)
    pressure = 1008 + 3.5 * season + 2.5 * synoptic + rng.normal(0, 1.2, n)
    cloud = np.clip(35 + 22 * synoptic + rng.normal(0, 18, n), 0, 100)
    rain_prob = np.clip(0.004 + 0.05 * np.exp(-((doy - 210) ** 2) / (2 * 30 ** 2)), 0, 1)
    precipitation = rng.gamma(1.2, 1.6, n) * (rng.random(n) < rain_prob)

    # --- emissions ----------------------------------------------------------
    diurnal = 1 + 0.42 * np.sin(2 * np.pi * (hour - 8) / 24) \
        + 0.28 * np.sin(4 * np.pi * (hour - 7) / 24)
    weekly = np.where(dow >= 5, 0.84, 1.0)
    emissions = 27 * diurnal * weekly * (1 + 0.30 * season)

    # --- dispersion: concentration ~ emissions / ventilation ----------------
    ventilation = np.clip(wind * blh, 200, None)
    pm25 = emissions * (9000.0 / ventilation) ** 0.70

    # dust intrusions, more likely in the pre-monsoon months
    dust_events = rng.random(n) < (0.00035 * (1 + 2 * np.exp(-((doy - 150) ** 2) / (2 * 40 ** 2))))
    dust_boost = np.zeros(n)
    for i in np.flatnonzero(dust_events):
        span = slice(i, min(i + rng.integers(12, 60), n))
        dust_boost[span] += rng.uniform(25, 80)

    # rain washout, with a short memory
    washout = np.ones(n)
    decay = 0.0
    for i in range(n):
        decay = max(decay * 0.94, min(precipitation[i] * 0.30, 0.7))
        washout[i] = 1 - decay

    ar = red_noise(1.0, rho=0.88)
    ar /= (ar.std() or 1.0)

    pm25 = np.clip((pm25 + dust_boost) * washout * np.exp(0.34 * ar), 2, 700)
    pm10 = np.clip(pm25 * rng.uniform(1.9, 2.5, n) + dust_boost * 1.7 + rng.normal(0, 9, n), 4, None)

    air_quality = pd.DataFrame({
        "pm2_5": pm25,
        "pm10": pm10,
        "ozone": np.clip(58 + 26 * np.sin(2 * np.pi * (hour - 14) / 24)
                         + 14 * np.sin(2 * np.pi * (doy - 120) / 365.25)
                         + rng.normal(0, 7, n), 1, None),
        "nitrogen_dioxide": np.clip(pm25 * 0.32 + rng.normal(0, 6, n), 1, None),
        "sulphur_dioxide": np.clip(11 + pm25 * 0.05 + rng.normal(0, 3, n), 0.5, None),
        "carbon_monoxide": np.clip(230 + pm25 * 3.6 + rng.normal(0, 45, n), 40, None),
        "dust": np.clip(14 + dust_boost * 0.55 + rng.normal(0, 5, n), 0, None),
        "aerosol_optical_depth": np.clip(0.30 + pm25 / 320 + rng.normal(0, 0.06, n), 0.01, None),
        "uv_index": np.clip(7 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
                            * (1 - cloud / 220), 0, None),
        "us_aqi": np.nan,
    }, index=idx)

    weather = pd.DataFrame({
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "dew_point_2m": temperature - (100 - humidity) / 5.0,
        "apparent_temperature": temperature + 0.9,
        "precipitation": precipitation,
        "surface_pressure": pressure,
        "cloud_cover": cloud,
        "wind_speed_10m": wind,
        "wind_direction_10m": (180 + 60 * synoptic + rng.normal(0, 25, n)) % 360,
        "wind_gusts_10m": wind * rng.uniform(1.3, 2.0, n),
        "boundary_layer_height": blh,
    }, index=idx)

    return air_quality, weather


def make_daily_features(days: int = 1460, seed: int = 7) -> pd.DataFrame:
    from src.config import CITY
    from src.features.build import build_daily_features
    aq, wx = make_hourly(days=days, seed=seed)
    return build_daily_features(aq, wx, CITY)
