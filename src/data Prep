"""Raw CSV → merged DataFrame (+ minimal cleaning).

Meant to be called by notebooks or feature pipeline.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from .utils import rolling_sum

RAW_FILES = {
    "calendar": "calender_data.csv",
    "acc": "regi_acc_get_data_transform.csv",
    "cm": "cm_data.csv",
    "call": "regi_call_data_transform.csv",
    "gt": "gt_service_name.csv",
}

START = "2018-03-01"
END = "2020-03-31"


def read_csv(p: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(p / RAW_FILES[name], parse_dates=["cdr_date"] if name != "gt" else ["week"])


def build_merged(raw_dir: str | Path, *, cache: bool = False) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    cal = read_csv(raw_dir, "calendar")
    acc = read_csv(raw_dir, "acc")
    cm = read_csv(raw_dir, "cm")
    call = read_csv(raw_dir, "call")
    gt = read_csv(raw_dir, "gt")

    idx = pd.date_range(START, END, freq="D")
    df = pd.DataFrame({"cdr_date": idx})

    def _ext(base: pd.DataFrame) -> pd.DataFrame:
        return df.merge(base, on="cdr_date", how="left")

    merged = (
        _ext(cal)
        .merge(_ext(acc), on="cdr_date", how="left")
        .merge(cm, on="cdr_date", how="left")
        .merge(_ext(call), on="cdr_date", how="left")
    )

    # calendar derived cols
    merged["dow"] = merged["cdr_date"].dt.dayofweek
    merged["woy"] = merged["cdr_date"].dt.isocalendar().week
    merged["wom"] = (merged["cdr_date"].dt.day - 1) // 7 + 1
    merged["doy"] = merged["cdr_date"].dt.dayofyear

    # week join with google trends
    merged["week"] = merged["cdr_date"] - pd.to_timedelta(merged["cdr_date"].dt.weekday, unit="D")
    gt["week"] = gt["week"].dt.to_period("W").dt.start_time
    merged = merged.merge(gt, on="week", how="left").drop(columns=["week", "holiday_name"], errors="ignore")

    # search_cnt interpolate
    if {"search_cnt_x", "search_cnt_y"}.issubset(merged.columns):
        merged["search_cnt"] = merged["search_cnt_y"].combine_first(merged["search_cnt_x"])
        merged.drop(columns=["search_cnt_x", "search_cnt_y"], inplace=True)
    merged["search_cnt"] = merged["search_cnt"].interpolate("linear").bfill().ffill()

    # sample cumulative cm flags as fast features
    merged["cm_flg_90d"] = rolling_sum(merged["cm_flg"], 90)

    if cache:
        out = Path("data/processed/merged_df_base.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False)
        print(f"Saved base merged to {out}")
    return merged
