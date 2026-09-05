"""US EPA Air Quality Index, computed properly.

Open-Meteo returns a convenience `us_aqi` field, but the EPA index is not an
hourly quantity: it is defined over pollutant-specific averaging windows -- a
24-hour mean for particulates, an 8-hour mean for ozone and carbon monoxide,
and 1-hour values for the two remaining gases. The index is then the *maximum*
sub-index across pollutants, and the pollutant attaining it is the "dominant"
one. We implement that here and validate against the API's field in the EDA.

Two things break naive implementations, both handled below:

1. Units. Open-Meteo reports O3, NO2, SO2 and CO in ug/m3. The EPA breakpoints
   are in ppb (O3, NO2, SO2) and ppm (CO). Converting requires the molecular
   weight and a reference temperature and pressure.
2. Truncation. The EPA specifies truncating each concentration to a set number
   of decimals *before* interpolating. Skipping this shifts values across
   breakpoint edges and changes the reported band.

Breakpoint provenance: US EPA "Technical Assistance Document for the Reporting
of Daily Air Quality -- the Air Quality Index (AQI)". The PM2.5 breakpoints
were revised in 2024 (the Good band upper bound moved from 12.0 to 9.0 ug/m3);
the revised table is used here. Re-verify against the current EPA document
before submission and record the edition in docs/DATA_CARD.md.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# (C_low, C_high, I_low, I_high)
Breakpoint = tuple[float, float, float, float]

BREAKPOINTS: dict[str, list[Breakpoint]] = {
    # PM2.5, 24-hour mean, ug/m3 (2024 revision)
    "pm2_5": [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
    ],
    # PM10, 24-hour mean, ug/m3
    "pm10": [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500),
    ],
    # Ozone, 8-hour mean, ppm
    "ozone": [
        (0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300),
    ],
    # Carbon monoxide, 8-hour mean, ppm
    "carbon_monoxide": [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500),
    ],
    # Sulphur dioxide, 1-hour, ppb
    "sulphur_dioxide": [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500),
    ],
    # Nitrogen dioxide, 1-hour, ppb
    "nitrogen_dioxide": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
}

# EPA truncation: decimal places each concentration is truncated to.
TRUNCATION: dict[str, int] = {
    "pm2_5": 1, "pm10": 0, "ozone": 3,
    "carbon_monoxide": 1, "sulphur_dioxide": 0, "nitrogen_dioxide": 0,
}

# Averaging window in hours for each pollutant's sub-index.
AVERAGING_HOURS: dict[str, int] = {
    "pm2_5": 24, "pm10": 24, "ozone": 8,
    "carbon_monoxide": 8, "sulphur_dioxide": 1, "nitrogen_dioxide": 1,
}

# Molar volume of an ideal gas at 25 C and 1013.25 hPa, in L/mol.
MOLAR_VOLUME = 24.45
MOLECULAR_WEIGHT = {
    "ozone": 48.00, "nitrogen_dioxide": 46.0055,
    "sulphur_dioxide": 64.066, "carbon_monoxide": 28.010,
}

POLLUTANTS: tuple[str, ...] = tuple(BREAKPOINTS)


def ugm3_to_ppb(values, pollutant: str):
    """ug/m3 -> ppb at 25 C, 1 atm."""
    return values * (MOLAR_VOLUME / MOLECULAR_WEIGHT[pollutant])


def ugm3_to_ppm(values, pollutant: str):
    """ug/m3 -> ppm at 25 C, 1 atm."""
    return values * (MOLAR_VOLUME / MOLECULAR_WEIGHT[pollutant]) / 1000.0


# The unit each pollutant's breakpoint table is expressed in. Getting ozone
# wrong here is the classic failure: its table is in ppm like CO, not ppb like
# the other two gases, and treating it as ppb inflates every ozone sub-index by
# a factor of 1000 and makes ozone spuriously dominant on every single row.
EPA_UNITS: dict[str, str] = {
    "pm2_5": "ug/m3", "pm10": "ug/m3",
    "ozone": "ppm", "carbon_monoxide": "ppm",
    "nitrogen_dioxide": "ppb", "sulphur_dioxide": "ppb",
}


def to_epa_units(values, pollutant: str):
    """Convert an Open-Meteo concentration into the units the EPA table expects."""
    unit = EPA_UNITS[pollutant]
    if unit == "ug/m3":
        return values
    if unit == "ppm":
        return ugm3_to_ppm(values, pollutant)
    return ugm3_to_ppb(values, pollutant)


def _truncate(value: float, decimals: int) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def sub_index(concentration: float, pollutant: str) -> float:
    """EPA sub-index for one already-averaged, already-converted concentration.

    Returns NaN for missing input. Concentrations above the top breakpoint are
    clamped to 500 rather than extrapolated -- the EPA does not define the index
    beyond its table, and extrapolating invents precision that does not exist.
    """
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return float("nan")
    if pollutant not in BREAKPOINTS:
        raise KeyError(f"unknown pollutant {pollutant!r}")

    conc = _truncate(float(concentration), TRUNCATION[pollutant])
    table = BREAKPOINTS[pollutant]
    if conc < table[0][0]:
        return 0.0
    for c_lo, c_hi, i_lo, i_hi in table:
        if c_lo <= conc <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo
    return 500.0


def sub_index_series(concentrations: pd.Series, pollutant: str) -> pd.Series:
    return concentrations.map(lambda v: sub_index(v, pollutant)).astype(float)


def rolling_average(series: pd.Series, hours: int, min_fraction: float = 0.75) -> pd.Series:
    """Trailing mean over `hours`, requiring `min_fraction` of the window present.

    EPA guidance requires a minimum data completeness before an average may be
    reported; a 24-hour PM2.5 mean built from three observations is not a
    24-hour mean. Windows that fall short return NaN rather than a value that
    looks authoritative and is not.
    """
    if hours <= 1:
        return series.astype(float)
    min_periods = max(1, int(round(hours * min_fraction)))
    return series.rolling(window=hours, min_periods=min_periods).mean()


def compute_hourly_aqi(df: pd.DataFrame, min_fraction: float = 0.75) -> pd.DataFrame:
    """Full EPA AQI from a UTC-indexed hourly frame of Open-Meteo concentrations.

    Expects the Open-Meteo column names (pm2_5, pm10, ozone, nitrogen_dioxide,
    sulphur_dioxide, carbon_monoxide) in their native ug/m3, sorted ascending
    and on a regular hourly index. Returns the sub-indices, the overall index
    and the dominant pollutant.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)
    available: list[str] = []

    for pollutant in POLLUTANTS:
        if pollutant not in df.columns:
            continue
        raw = pd.to_numeric(df[pollutant], errors="coerce")
        averaged = rolling_average(raw, AVERAGING_HOURS[pollutant], min_fraction)
        converted = to_epa_units(averaged, pollutant)
        out[f"sub_{pollutant}"] = sub_index_series(converted, pollutant)
        available.append(pollutant)

    if not available:
        return out

    sub_cols = [f"sub_{p}" for p in available]
    subs = out[sub_cols]
    out["us_aqi_epa"] = subs.max(axis=1, skipna=True)
    # idxmax over columns gives the dominant pollutant; NaN-only rows give NaN.
    dominant = subs.where(subs.notna().any(axis=1)).idxmax(axis=1, skipna=True)
    out["dominant_pollutant"] = dominant.str.replace("sub_", "", regex=False)
    return out


def daily_aqi(hourly_aqi: pd.Series, tz: str) -> pd.DataFrame:
    """Collapse an hourly AQI series into local-calendar daily mean and max."""
    local = hourly_aqi.tz_convert(tz)
    grouped = local.groupby(local.index.date)
    frame = pd.DataFrame({
        "aqi_mean": grouped.mean(),
        "aqi_max": grouped.max(),
        "aqi_min": grouped.min(),
        "aqi_std": grouped.std(),
        "hours_observed": grouped.count(),
    })
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame


def band_label(aqi: float) -> str:
    from src.config import band_for
    return band_for(aqi).label
