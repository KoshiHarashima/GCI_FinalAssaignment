# AirREGI Call Center – Call Volume Forecast 📈
AirREGI コールセンターの 1 日あたり問い合わせ件数 **`call_num`** を予測し、  
通常期と増税前の“駆け込み需要期”でのオペレーション最適化を目指します。

| 項目 | 概要 |
|------|------|
| **手法** | VAR で外生変数（検索数・CM など）を先に予測 → XGBoost / ARIMA で最終予測 |
| **ゴール** | シフト人数の平準化・コスト削減・顧客待ち時間の短縮 |

---

## 1. ディレクトリ構成

```text
air-callnum-forecast/
├── notebooks/          # 実験用 Jupyter Notebook
├── src/                # 再利用可能な Python モジュール
│   ├── data_prep.py    # 00: 生データ → 結合データ
│   ├── features.py     # 01: 特徴量生成
│   └── models/         # 02: VAR / XGB ラッパ
├── models/             # 学習済みモデル（.pkl / .json など）
├── reports/
│   ├── figures/        # グラフ出力
│   └── metrics/        # メトリクス JSON
├── requirements.txt    # pip 用
└── README.md

```

## 2. 環境構築
```text
Conda を利用する場合（推奨）
git clone https://github.com/your-name/air-callnum-forecast.git
cd air-callnum-forecast
conda env create -f environment.yml
conda activate air-callnum-forecast

venv + pip を利用する場合
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. データ取得手順
```text
ファイル名	説明
calender_data.csv	祝日・曜日フラグ
regi_acc_get_data_transform.csv	アカウント開設数
cm_data.csv	CM 放映フラグ
regi_call_data_transform.csv	コールセンター問い合わせ件数
gt_service_name.csv	Google Trends（週単位）
上記 5 ファイルを Google Drive → data/raw/ に配置
```

## 4. 再現手順
```text
ステップ	コマンド	生成物
EDA 用グラフ	python src/00_EDA.py	reports/figures/
通常期モデル（2020-03）	python src/02_model_VAR_XGB.py --scenario normal	models/final_xgb_normal.json
駆け込み期モデル（2019-09）	python src/02_model_VAR_XGB.py --scenario peak	models/final_xgb_peak.json
ARIMA ベースライン	python src/03_model_ARIMA.py	reports/metrics/arima_metrics.json

```
## 5. 主な結果
```text
シナリオ	モデル	MAE	RMSE
通常期 (20-03)	VAR → XGBoost	15.4	21.0
駆け込み期 (19-09)	VAR → XGBoost	68.4	111.3
通常期ベースライン	ARIMA(4,1,0)	54.1	59.9
※ 数値はサンプル。再実行時は最新結果に差し替えてください。
```
