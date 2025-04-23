# GCI_FinalAssaignment
GCIでの最終課題に用いたコード

# AirREGI Call Center – Call Volume Forecast

予測対象: **AirREGI** コールセンターの 1 日あたり `call_num`  
手法: VAR による外生変数予測 ➜ XGBoostによる最終予測  
目的: 通常期と駆け込み需要期（増税前）のオペレーション最適化

---

## 1. ディレクトリ構成

GCI_FinalAssaignment/
├── notebooks/ # 実験用 (ipynb)
├── src/ # 再利用可能な Python モジュール
├── data/ │ 
├── raw/ # ここに生データを置く (Git 管理外) 
│ └── processed/ # 01_feature_engineering.py が生成 
├── models/ # 学習済みモデル (Git 管理外) 
├── reports/ │
├── figures/ # EDA・結果グラフ
│ └── metrics/ # JSON メトリクス 
├── requirements.txt 
└── README.md

---

## 2. セットアップ

```bash
# clone
git clone https://github.com/your-name/air-callnum-forecast.git
cd air-callnum-forecast

# Python 仮想環境 (conda 例)
conda env create -f environment.yml
conda activate air-callnum-forecast
pip 派の方は python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt をどうぞ。

3. データ取得手順
data/raw/ に配置してください(機密管理のためNG）

calender_data.csv

regi_acc_get_data_transform.csv

cm_data.csv

regi_call_data_transform.csv

gt_service_name.csv

フォルダ構成が正しいことを確認したら前処理を実行します:


python src/01_feature_engineering.py \
       --raw_dir data/raw \
       --out_path data/processed/merged_df_cleaned.csv

4. 再現方法
4-1. EDA
python src/00_EDA.py \
       --data_dir data/processed \
       --out_dir reports/figures

4-2. 通常期モデル (2020-03)
python src/02_model_VAR_XGB.py --scenario normal

4-3. 駆け込み期モデル (2019-09)
python src/02_model_VAR_XGB.py --scenario peak

4-4. ARIMA ベースライン
python src/03_model_ARIMA.py
実行後:

学習済みモデル → models/

予測 CSV → models/pred_<scenario>.csv

メトリクス JSON → reports/metrics/

グラフ → reports/figures/
