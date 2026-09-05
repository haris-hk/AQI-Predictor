"""Raw hourly observations -> the daily modelling frame.

Every feature here is *causal*: computed from information available at the end
of day t, used to predict days t+1..t+3. The one exception is forecast weather,
which is legitimately available for future days at prediction time -- see
`add_future_weather` and PROJECT_PLAN.md section 3.2 for why that distinction
matters and how the train/serve skew is handled.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CITY, HORIZONS, TARGETS, City, target_col
from src.features.aqi import compute_hourly_aqi, daily_aqi

log = logging.getLogger(__name__)

WEATHER_MEAN_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
    "boundary_layer_height",
]
POLLUTANT_COLS = [
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide", "dust", "aerosol_optical_depth",
]
# 4-6 are here so the seasonal-naive baseline (y[t+h] = y[t+h-7]) can be
# expressed exactly rather than approximated.
LAG_DAYS = (1, 2, 3, 4, 5, 6, 7, 14)
ROLL_WINDOWS = (3, 7, 14, 30)
CHANGE_WINDOWS = (1, 3, 7)

META_COLS = {"city_id", "date", "hours_observed", "dominant_pollutant", "ingested_at"}


# ------------------------------------------------------------------ hourly join
def merge_hourly(air_quality: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join the two hourly sources on the UTC index and attach the EPA AQI."""
    if air_quality.empty:
        return pd.DataFrame()
    merged = air_quality.join(weather, how="outer", rsuffix="_wx") if not weather.empty \
        else air_quality.copy()
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    aqi_frame = compute_hourly_aqi(merged)
    return merged.join(aqi_frame, how="left")


# ----------------------------------------------------------- daily aggregation
def aggregate_daily(hourly: pd.DataFrame, city: City = CITY) -> pd.DataFrame:
    """Collapse the hourly frame to the local calendar day."""
    if hourly.empty or "us_aqi_epa" not in hourly:
        return pd.DataFrame()

    daily = daily_aqi(hourly["us_aqi_epa"].dropna(), city.timezone)

    local = hourly.tz_convert(city.timezone)
    key = local.index.date

    for col in POLLUTANT_COLS:
        if col in local.columns:
            grouped = local[col].groupby(key)
            daily[f"{col}_mean"] = grouped.mean().values if len(grouped) == len(daily) \
                else grouped.mean().reindex(daily.index.date).values
            daily[f"{col}_max"] = grouped.max().reindex(daily.index.date).values

    for col in WEATHER_MEAN_COLS:
        if col in local.columns:
            g = local[col].groupby(key)
            daily[f"{col}_mean"] = g.mean().reindex(daily.index.date).values
            if col in ("temperature_2m", "wind_speed_10m", "boundary_layer_height"):
                daily[f"{col}_min"] = g.min().reindex(daily.index.date).values
                daily[f"{col}_max"] = g.max().reindex(daily.index.date).values

    if "precipitation" in local.columns:
        daily["precipitation_sum"] = (
            local["precipitation"].groupby(key).sum().reindex(daily.index.date).values
        )
    if "wind_gusts_10m" in local.columns:
        daily["wind_gusts_10m_max"] = (
            local["wind_gusts_10m"].groupby(key).max().reindex(daily.index.date).values
        )
    if "wind_direction_10m" in local.columns:
        # Wind direction is circular: the mean of 350 deg and 10 deg is 0, not 180.
        rad = np.deg2rad(local["wind_direction_10m"])
        u = pd.Series(np.sin(rad).values, index=local.index).groupby(key).mean()
        v = pd.Series(np.cos(rad).values, index=local.index).groupby(key).mean()
        daily["wind_dir_sin"] = u.reindex(daily.index.date).values
        daily["wind_dir_cos"] = v.reindex(daily.index.date).values

    if "dominant_pollutant" in hourly.columns:
        dom = local["dominant_pollutant"].groupby(key).agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else None
        )
        daily["dominant_pollutant"] = dom.reindex(daily.index.date).values

    daily.insert(0, "city_id", city.city_id)
    return daily


# ------------------------------------------------------------------- calendar
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["day_of_week"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["day_of_year"] = idx.dayofyear
    out["week_of_year"] = idx.isocalendar().week.astype(int).values
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Cyclical encodings so that 31 December sits next to 1 January rather than
    # 364 units away from it.
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


# ------------------------------------------------------- lags / rolling / change
def add_lag_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    columns = columns or [c for c in ("aqi_mean", "aqi_max", "pm2_5_mean", "pm10_mean")
                          if c in out.columns]
    for col in columns:
        for lag in LAG_DAYS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Trailing statistics. `closed='right'` keeps day t in its own window, which
    is correct: at prediction time on day t, day t is observed."""
    out = df.copy()
    columns = columns or [c for c in ("aqi_mean", "aqi_max", "pm2_5_mean") if c in out.columns]
    for col in columns:
        for w in ROLL_WINDOWS:
            roll = out[col].rolling(window=w, min_periods=max(2, w // 2))
            out[f"{col}_roll{w}_mean"] = roll.mean()
            out[f"{col}_roll{w}_std"] = roll.std()
            if col == "aqi_mean":
                out[f"{col}_roll{w}_min"] = roll.min()
                out[f"{col}_roll{w}_max"] = roll.max()
    return out


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """AQI change rate -- explicitly required by the brief."""
    out = df.copy()
    for col in [c for c in ("aqi_mean", "aqi_max", "pm2_5_mean") if c in out.columns]:
        for w in CHANGE_WINDOWS:
            out[f"{col}_diff{w}"] = out[col].diff(w)
            out[f"{col}_pct_change{w}"] = out[col].pct_change(w).replace(
                [np.inf, -np.inf], np.nan)
        # Slope of a least-squares line through the trailing week, in AQI/day.
        out[f"{col}_slope7"] = (
            out[col].rolling(7, min_periods=4)
            .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=True)
        )
    if {"aqi_mean", "aqi_mean_roll30_mean"} <= set(out.columns):
        out["aqi_vs_month_mean"] = out["aqi_mean"] - out["aqi_mean_roll30_mean"]
    return out


def add_pollutant_mix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Composition features. In Karachi these separate dust intrusions from
    combustion smog, which behave differently and persist differently."""
    out = df.copy()
    if {"pm2_5_mean", "pm10_mean"} <= set(out.columns):
        out["pm_ratio"] = out["pm2_5_mean"] / out["pm10_mean"].replace(0, np.nan)
        out["pm_coarse"] = (out["pm10_mean"] - out["pm2_5_mean"]).clip(lower=0)
    if "dominant_pollutant" in out.columns:
        for pol in ("pm2_5", "pm10", "ozone", "nitrogen_dioxide"):
            out[f"dominant_is_{pol}"] = (out["dominant_pollutant"] == pol).astype(int)
    return out


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap, physically motivated dispersion proxies.

    Pollution concentration is roughly emissions divided by ventilation, and
    ventilation is mixing-layer depth times wind speed. These consistently rank
    high in the SHAP attributions, which is a good sign that the model has
    learned meteorology rather than memorised the calendar.
    """
    out = df.copy()
    blh = out.get("boundary_layer_height_mean")
    wind = out.get("wind_speed_10m_mean")
    if blh is not None and wind is not None:
        out["ventilation_index"] = blh * wind
        out["stagnation"] = ((wind < 2.0) & (blh < 500)).astype(int)
    if blh is not None:
        out["blh_range"] = out.get("boundary_layer_height_max", blh) - \
                           out.get("boundary_layer_height_min", blh)
    if {"temperature_2m_max", "temperature_2m_min"} <= set(out.columns):
        # A large diurnal range implies clear skies and strong nocturnal
        # radiative cooling, which favours a surface inversion that traps
        # pollution overnight.
        out["temp_range"] = out["temperature_2m_max"] - out["temperature_2m_min"]
    if {"temperature_2m_mean", "dew_point_2m_mean"} <= set(out.columns):
        out["dew_point_depression"] = out["temperature_2m_mean"] - out["dew_point_2m_mean"]
    if "precipitation_sum" in out.columns:
        out["rained"] = (out["precipitation_sum"] > 0.2).astype(int)
        out["precip_3d"] = out["precipitation_sum"].rolling(3, min_periods=1).sum()
    return out


# --------------------------------------------------------------- future weather
FUTURE_WEATHER_COLS = [
    "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "boundary_layer_height_mean", "precipitation_sum", "surface_pressure_mean",
    "cloud_cover_mean", "ventilation_index",
]


def add_future_weather(df: pd.DataFrame, horizons=HORIZONS,
                       source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weather for days t+1..t+3 as features on row t.

    This is the only genuinely forward-looking signal in the model, and it is
    legitimate: a weather forecast for t+3 really is available on day t.

    `source` supplies those values. In production it is the forecast frame. In
    training it should ideally be the *historical forecast* -- weather as it was
    predicted at the time -- so that training and serving see the same quality
    of input. Passing None falls back to the observed record, which is
    optimistic; the fallback is recorded in the run metadata so the report can
    state which regime produced a given model.
    """
    out = df.copy()
    src = source if source is not None else df
    for h in horizons:
        for col in FUTURE_WEATHER_COLS:
            if col in src.columns:
                shifted = src[col].shift(-h)
                out[f"fc_{col}_h{h}"] = shifted.reindex(out.index)
    return out


# --------------------------------------------------------------------- targets
def add_targets(df: pd.DataFrame, horizons=HORIZONS, targets=TARGETS) -> pd.DataFrame:
    """Direct multi-horizon targets: no recursion, no compounding error."""
    out = df.copy()
    for t in targets:
        if t not in out.columns:
            continue
        for h in horizons:
            out[target_col(t, h)] = out[t].shift(-h)
    return out


# ----------------------------------------------------------------- orchestration
def build_daily_features(air_quality: pd.DataFrame, weather: pd.DataFrame,
                         city: City = CITY,
                         future_weather_source: pd.DataFrame | None = None,
                         with_targets: bool = True) -> pd.DataFrame:
    """The single transform used by backfill, the hourly pipeline and serving.

    Training and inference call this same function, which is what makes the
    feature definitions structurally identical in both paths.
    """
    hourly = merge_hourly(air_quality, weather)
    if hourly.empty:
        return pd.DataFrame()

    daily = aggregate_daily(hourly, city)
    if daily.empty:
        return pd.DataFrame()

    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_change_features(daily)
    daily = add_pollutant_mix_features(daily)
    daily = add_physics_features(daily)

    fut_src = future_weather_source
    if fut_src is None:
        fut_src = daily
    else:
        fut_src = add_physics_features(fut_src)
    daily = add_future_weather(daily, source=fut_src)

    if with_targets:
        daily = add_targets(daily)

    daily.index.name = "date"
    return daily


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything numeric that is not a target or metadata.

    Deriving this from the frame rather than hardcoding a list means a feature
    added in one place cannot be silently missing at serving time.
    """
    target_names = {target_col(t, h) for t in TARGETS for h in HORIZONS}
    cols = []
    for c in df.columns:
        if c in target_names or c in META_COLS:
            continue
        if c in TARGETS:          # the present-day value of a target is a valid lag-0 feature
            cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def training_frame(df: pd.DataFrame, target: str, horizon: int
                   ) -> tuple[pd.DataFrame, pd.Series]:
    """X, y for one (target, horizon), with rows lacking the label dropped."""
    col = target_col(target, horizon)
    if col not in df.columns:
        raise KeyError(f"{col} not in frame; build with with_targets=True")
    usable = df[df[col].notna()]
    return usable[feature_columns(usable)], usable[col]
