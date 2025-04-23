"""01_feature_engineering.py
Creates merged_df_cleaned.csv with engineered features described in notebook.
Usage:
    python 01_feature_engineering.py --raw_dir data/raw --out_path data/processed/merged_df_cleaned.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW_FILES = {
    "calendar": "calender_data.csv",
    "acc": "regi_acc_get_data_transform.csv",
    "cm": "cm_data.csv",
    "call": "regi_call_data_transform.csv",
    "gt": "gt_service_name.csv",
}

START = "2018-03-01"
END = "2020-03-31"


def rolling_sum(df, col, days):
    return df[col].rolling(window=days, min_periods=1).sum()


def lag(df, col, days):
    return df[col].shift(days)


def moving_avg(df, col, window):
    return df[col].rolling(window=window, min_periods=1).mean()


def build_custom_trend(df):
    start, end, zero = pd.to_datetime("2020-06-01"), pd.to_datetime("2020-09-30"), pd.to_datetime("2020-10-01")
    factor = 0.115
    df["rush_trend"] = 0.01
    mask = (df["cdr_date"] >= start) & (df["cdr_date"] <= end)
    days = (df.loc[mask, "cdr_date"] - start).dt.days
    df.loc[mask, "rush_trend"] = 0.01 * (factor ** days)
    df.loc[df["cdr_date"] >= zero, "rush_trend"] = 0
    df.loc[df["dow"].isin([5, 6]), "rush_trend"] = 0
    df.loc[df["holiday_flag"], "rush_trend"] = 0


def main(args):
    raw_dir = Path(args.raw_dir)
    # ---------------- read -----------------
    cal = pd.read_csv(raw_dir / RAW_FILES["calendar"], parse_dates=["cdr_date"])
    acc = pd.read_csv(raw_dir / RAW_FILES["acc"], parse_dates=["cdr_date"])
    cm = pd.read_csv(raw_dir / RAW_FILES["cm"], parse_dates=["cdr_date"])
    call = pd.read_csv(raw_dir / RAW_FILES["call"], parse_dates=["cdr_date"])

    date_range = pd.date_range(START, END, freq="D")
    # calendar extend
    cal_ext = pd.DataFrame({"cdr_date": date_range}).merge(cal, on="cdr_date", how="left")
    cal_ext["dow"] = cal_ext["cdr_date"].dt.dayofweek
    cal_ext["woy"] = cal_ext["cdr_date"].dt.isocalendar().week
    cal_ext["wom"] = (cal_ext["cdr_date"].dt.day - 1) // 7 + 1
    cal_ext["doy"] = cal_ext["cdr_date"].dt.dayofyear
    # holiday nan for early period
    mask = cal_ext["cdr_date"].dt.strftime("%Y-%m").isin(["2018-03", "2018-04", "2018-05"])
    cal_ext.loc[mask, ["holiday_flag", "day_before_holiday_flag", "holiday_name"]] = np.nan

    # extend other frames
    def extend(df):
        return pd.DataFrame({"cdr_date": date_range}).merge(df, on="cdr_date", how="left")

    acc_ext = extend(acc)
    cm_ext = cm.copy()
    call_ext = extend(call)

    merged = cal_ext.merge(acc_ext, on="cdr_date", how="left") \
                    .merge(cm_ext, on="cdr_date", how="left") \
                    .merge(call_ext, on="cdr_date", how="left")

    # week merge with google trends service name
    merged["week"] = merged["cdr_date"] - pd.to_timedelta(merged["cdr_date"].dt.weekday, unit="D")
    gt = pd.read_csv(raw_dir / RAW_FILES["gt"], parse_dates=["week"])
    gt["week"] = gt["week"].dt.to_period("W").dt.start_time
    merged = merged.merge(gt, on="week", how="left").drop(columns=["week", "holiday_name"], errors="ignore")

    # ---------------- feature engineering -----------------
    merged["cm_flg_90d"] = rolling_sum(merged, "cm_flg", 90)
    merged["cm_flg_14d"] = rolling_sum(merged, "cm_flg", 14)
    merged["cm_flg_3d"] = rolling_sum(merged, "cm_flg", 3)

    for d in [90, 30, 7]:
        merged[f"search_cnt_lag_{d}d"] = lag(merged, "search_cnt", d)
    for d in [30, 14, 7]:
        merged[f"acc_get_cnt_lag_{d}d"] = lag(merged, "acc_get_cnt", d)
    for d in [1, 3, 7, 14, 30]:
        merged[f"call_num_lag_{d}d"] = lag(merged, "call_num", d)

    for w in [3, 7, 30, 90]:
        merged[f"search_cnt_ma_{w}d"] = moving_avg(merged, "search_cnt", w)
        merged[f"acc_get_cnt_ma_{w}d"] = moving_avg(merged, "acc_get_cnt", w)
    merged["call_num_ma_30d"] = moving_avg(merged, "call_num", 30)
    merged["call_num_ma_90d"] = moving_avg(merged, "call_num", 90)

    # diff & ratio
    merged["search_cnt_ma_diff"] = merged["search_cnt_ma_7d"] - merged["search_cnt_ma_30d"]
    merged["search_cnt_ma_ratio"] = merged["search_cnt_ma_7d"] / merged["search_cnt_ma_30d"]
    merged["acc_get_cnt_ma_diff"] = merged["acc_get_cnt_ma_7d"] - merged["acc_get_cnt_ma_30d"]
    merged["acc_get_cnt_ma_ratio"] = merged["acc_get_cnt_ma_7d"] / merged["acc_get_cnt_ma_30d"]

    # interaction
    merged["search_cnt_x_acc_get_cnt"] = merged["search_cnt"] * merged["acc_get_cnt"]
    merged["search_cnt_x_cm_flg"] = merged["search_cnt"] * merged["cm_flg"]
    merged["acc_get_cnt_x_cm_flg"] = merged["acc_get_cnt"] * merged["cm_flg"]

    # custom trend & flags
    build_custom_trend(merged)
    merged["flag_anounce"] = ((merged["cdr_date"] >= "2018-11-01") & (merged["cdr_date"] <= "2018-11-15")).astype(int)

    # cleanup
    merged.drop(columns=["dow_name"], inplace=True, errors="ignore")
    merged.to_csv(args.out_path, index=False)
    print(f"Saved engineered data to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_path", default="data/processed/merged_df_cleaned.csv")
    main(parser.parse_args())
