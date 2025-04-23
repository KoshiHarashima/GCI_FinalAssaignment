"""02_model_VAR_XGB.py
VAR for exogenous feature forecasting + XGBoost final regression.
Scenarios:
    normal  -> 2020-03 forecast window (stable)
    peak    -> 2019-09 forecast window (rush)
Usage:
    python 02_model_VAR_XGB.py --scenario normal --data data/processed/merged_df_cleaned.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

EXCLUDE_COLS = {
    "meta": ["cdr_date", "wom", "financial_year"],
    "date_only": ["dow", "woy", "wom", "doy", "day_before_holiday_flag", "holiday_flag"],
    "leak": [
        "cm_flg_90d", "cm_flg_14d", "cm_flg_3d", "rush_trend", "flag_anounce",  # allowed exogenous
    ],
}

SCENARIOS = {
    "normal": dict(forecast_start="2020-03-01", forecast_end="2020-03-31", test_days=31),
    "peak": dict(forecast_start="2020-09-01", forecast_end="2020-09-30", test_days=30),
}


def fit_var(df, maxlags=10):
    var = smt.VAR(df)
    return var.fit(maxlags=maxlags)


def auto_grid_search(dtrain):
    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "seed": 42,
    }
    best_rmse, best_params = 1e9, None
    for max_depth in [4, 6, 8]:
        for eta in [0.01, 0.05]:
            for subsample in [0.6, 0.8, 1.0]:
                for colsample in [0.6, 0.8, 1.0]:
                    p = {**base, "max_depth": max_depth, "eta": eta, "subsample": subsample, "colsample_bytree": colsample}
                    cv = xgb.cv(p, dtrain, num_boost_round=500, nfold=5, early_stopping_rounds=30, verbose_eval=False)
                    rmse = cv["test-rmse-mean"].min()
                    if rmse < best_rmse:
                        best_rmse, best_params = rmse, p
    return best_params


def main(args):
    cfg = SCENARIOS[args.scenario]
    df = pd.read_csv(args.data, parse_dates=["cdr_date"])
    # ---------------- create VAR input -----------------
    train_df = df[df["cdr_date"] < cfg["forecast_start"]]

    # select non-leak exogenous features
    var_feats = train_df.drop(columns=[*EXCLUDE_COLS["meta"], "call_num"], errors="ignore")
    maxlags = min(10, len(var_feats) // 10)
    var_res = fit_var(var_feats, maxlags=maxlags)

    steps = cfg["test_days"]
    var_input = var_feats.values[-var_res.k_ar:]
    forecast = var_res.forecast(var_input, steps=steps)
    forecast_idx = pd.date_range(cfg["forecast_start"], periods=steps, freq="D")
    forecast_df = pd.DataFrame(forecast, columns=var_feats.columns, index=forecast_idx)

    # insert forecast into copy of original df
    df2 = df.copy()
    mask = (df2["cdr_date"] >= cfg["forecast_start"]) & (df2["cdr_date"] <= cfg["forecast_end"])
    df2.loc[mask, forecast_df.columns] = forecast_df.values

    # rebuild lag/moving features that depend on replaced cols
    for d in [1, 3, 7, 14, 30]:
        df2[f"call_num_lag_{d}d"] = df2["call_num"].shift(d)
    for d in [90, 30, 7]:
        df2[f"search_cnt_lag_{d}d"] = df2["search_cnt"].shift(d)
    for d in [30, 14, 7]:
        df2[f"acc_get_cnt_lag_{d}d"] = df2["acc_get_cnt"].shift(d)

    # fill na
    df2.fillna(method="ffill", inplace=True)

    # split train / test
    test_len = cfg["test_days"]
    X = df2.drop(columns=["call_num", *EXCLUDE_COLS["meta"]])
    y = df2["call_num"]
    X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]
    y_train, y_test = y.iloc[:-test_len], y.iloc[-test_len:]

    dtrain, dtest = xgb.DMatrix(X_train, y_train), xgb.DMatrix(X_test, y_test)

    params = auto_grid_search(dtrain)
    model = xgb.train(params, dtrain, num_boost_round=800, early_stopping_rounds=50, evals=[(dtest, "val")], verbose_eval=False)

    preds = model.predict(dtest)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    model.save_model(out_dir / f"final_xgb_{args.scenario}.json")

    pd.DataFrame({"y_test": y_test, "y_pred": preds}, index=y_test.index).to_csv(out_dir / f"pred_{args.scenario}.csv")
    print(f"{args.scenario} -> MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="normal")
    parser.add_argument("--data", default="data/processed/merged_df_cleaned.csv")
    main(parser.parse_args())
