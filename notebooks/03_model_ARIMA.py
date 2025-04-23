"""03_model_ARIMA.py
Simple ARIMA baseline on call_num.
Usage:
    python 03_model_ARIMA.py --data data/processed/merged_df_cleaned.csv --train_end 2020-02-29 --test_days 31
"""
import argparse
from pathlib import Path

import json
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main(args):
    df = pd.read_csv(args.data, parse_dates=["cdr_date"])
    ts = df.set_index("cdr_date")["call_num"].dropna()

    train = ts[: args.train_end]
    test = ts[args.test_days * -1 :]

    model = ARIMA(train, order=(4, 1, 0)).fit()
    preds = model.predict(start=train.index[-1] + pd.Timedelta(days=1), end=ts.index[-1])

    mae = mean_absolute_error(test, preds)
    rmse = mean_squared_error(test, preds, squared=False)

    metrics = {"MAE": mae, "RMSE": rmse}
    (Path("reports") / "metrics").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics/arima_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/merged_df_cleaned.csv")
    parser.add_argument("--train_end", default="2020-02-29")
    parser.add_argument("--test_days", type=int, default=31)
    main(parser.parse_args())
