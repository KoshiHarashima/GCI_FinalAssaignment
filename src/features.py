"""Feature engineering layer.

Takes merged DataFrame from data_prep and adds lag/moving‑average, interactions, etc.
"""
from __future__ import annotations

import pandas as pd
from .utils import rolling_sum, lag, moving_avg, diff_ratio, make_rush_trend


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # cm flag windows
    for w in [90, 14, 3]:
        out[f"cm_flg_{w}d"] = rolling_sum(out["cm_flg"], w)

    # lags
    for d in [90, 30, 7]:
        out[f"search_cnt_lag_{d}d"] = lag(out["search_cnt"], d)
    for d in [30, 14, 7]:
        out[f"acc_get_cnt_lag_{d}d"] = lag(out["acc_get_cnt"], d)
    for d in [1, 3, 7, 14, 30]:
        out[f"call_num_lag_{d}d"] = lag(out["call_num"], d)

    # moving averages
    for w in [3, 7, 30, 90]:
        out[f"search_cnt_ma_{w}d"] = moving_avg(out["search_cnt"], w)
        out[f"acc_get_cnt_ma_{w}d"] = moving_avg(out["acc_get_cnt"], w)
    out["call_num_ma_30d"] = moving_avg(out["call_num"], 30)
    out["call_num_ma_90d"] = moving_avg(out["call_num"], 90)

    out["search_cnt_ma_diff"], out["search_cnt_ma_ratio"] = diff_ratio(
        out["search_cnt_ma_7d"], out["search_cnt_ma_30d"]
    )
    out["acc_get_cnt_ma_diff"], out["acc_get_cnt_ma_ratio"] = diff_ratio(
        out["acc_get_cnt_ma_7d"], out["acc_get_cnt_ma_30d"]
    )

    # interactions
    out["search_cnt_x_acc_get_cnt"] = out["search_cnt"] * out["acc_get_cnt"]
    out["search_cnt_x_cm_flg"] = out["search_cnt"] * out["cm_flg"]
    out["acc_get_cnt_x_cm_flg"] = out["acc_get_cnt"] * out["cm_flg"]

    # rush trend
    out["rush_trend"] = make_rush_trend(out.set_index("cdr_date"))

    # flag announce
    out["flag_anounce"] = (
        (out["cdr_date"] >= "2018-11-01") & (out["cdr_date"] <= "2018-11-15")
    ).astype(int)

    return out
