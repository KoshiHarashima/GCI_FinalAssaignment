"""00_EDA.py
Exploratory Data Analysis – weekly trend plots for call_num, acc_get_cnt, cm_flg, search_cnt.
Usage:
    python 00_EDA.py --data_dir data/processed --out_dir reports/figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PLOTS = [
    ("regi_call_data_transform.csv", "call_num", "Weekly Sum of call_num"),
    ("regi_acc_get_data_transform.csv", "acc_get_cnt", "Weekly Sum of acc_get_cnt"),
    ("cm_data.csv", "cm_flg", "Monthly Sum of cm_flg", "M"),
    ("merged_df_cleaned.csv", "search_cnt", "Weekly Mean of search_cnt"),
]

def plot_timeseries(df, date_col, value_col, freq, title, out_path):
    ts = df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum()
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, col, title, *freq in PLOTS:
        freq = freq[0] if freq else "W"
        df = pd.read_csv(Path(args.data_dir) / fname, parse_dates=["cdr_date"])
        plot_timeseries(df, "cdr_date", col, freq, title, out_dir / f"{col}_{freq}.png")
    print(f"Figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="reports/figures")
    main(parser.parse_args())
