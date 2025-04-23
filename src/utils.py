"""Common utility functions used across modules."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List

# ---------------- basic feature helpers ----------------

def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Cumulative rolling sum with minimum 1 period."""
    return series.rolling(window=window, min_periods=1).sum()

def lag(series: pd.Series, lag_days: int) -> pd.Series:
    """Shift series forward by *lag_days*."""
    return series.shift(lag_days)

def moving_avg(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def diff_ratio(short_ma: pd.Series, long_ma: pd.Series) -> tuple[pd.Series, pd.Series]:
    diff = short_ma - long_ma
    ratio = short_ma / long_ma
    return diff, ratio

# ---------------- custom rush trend ----------------

def make_rush_trend(df: pd.DataFrame) -> pd.Series:
    """Synthetic exponential bump between 2020‑06‑01 and 2020‑09‑30.

    * week‑end / holiday → 0
    * before 2019‑06‑01 → 0.01 (flat)
    * after 2020‑10‑01 → 0
    """
    start, end, zero = pd.to_datetime("2020-06-01"), pd.to_datetime("2020-09-30"), pd.to_datetime("2020-10-01")
    factor = 0.115

    trend = pd.Series(0.01, index=df.index, name="rush_trend")
    mask = (df.index >= start) & (df.index <= end)
    trend.loc[mask] = 0.01 * (factor ** (df.index[mask] - start).days)
    trend.loc[df.index >= zero] = 0
    if "dow" in df.columns:
        trend.loc[df["dow"].isin([5, 6])] = 0
    if "holiday_flag" in df.columns:
        trend.loc[df["holiday_flag"].fillna(False)] = 0
    return trend
