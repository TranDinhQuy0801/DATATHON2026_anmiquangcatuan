
# DATATHON 2026 — E-Commerce Sales Forecasting
Predict daily Revenue and COGS for a Vietnamese e-commerce company using 10 years of historical sales data (2012–2022).
## FOLDER STUCTURE
```
DATATHON2026
├─ Prediction                           # Forecasting pipeline & outputs
│  ├─ Blend_50_50.py
│  ├─ submission.csv
│  ├─ LGB_pure.csv
│  ├─ Thuat_toan_LGB_pure_model.py
│  ├─ Thuat_toan_Trimmed_mean_model.py
│  ├─ Trimmed_mean_model.csv
│  └─ catboost_info
│     ├─ catboost_training.json
│     ├─ learn
│     │  └─ events.out.tfevents
│     ├─ learn_error.tsv
│     ├─ test
│     │  └─ events.out.tfevents
│     ├─ test_error.tsv
│     └─ time_left.tsv
├─ README.md
├─ baseline.ipynb
├─ calc_tool
│  └─ marginal_profit.ipynb
├─ dataset                              # Raw data
│  ├─ customers.csv
│  ├─ geography.csv
│  ├─ inventory.csv
│  ├─ order_items.csv
│  ├─ orders.csv
│  ├─ payments.csv
│  ├─ products.csv
│  ├─ promotions.csv
│  ├─ returns.csv
│  ├─ reviews.csv
│  ├─ sales.csv
│  ├─ sample_submission.csv
│  ├─ shipments.csv
│  ├─ unified_sales_data.csv
│  └─ web_traffic.csv
├─ part1                                # Solving problems from part1 with pandas
│  ├─ q10sol.ipynb
│  ├─ q1sol.ipynb
│  ├─ q2sol.ipynb
│  ├─ q3sol.ipynb
│  ├─ q4sol.ipynb
│  ├─ q5sol.ipynb
│  ├─ q6sol.ipynb
│  ├─ q7sol.ipynb
│  ├─ q8sol.ipynb
│  └─ q9sol.ipynb
├─ requirements.txt
└─ visualize_keys                       # Visualize data for EDA
   ├─ geo_sale.ipynb
   ├─ inventory_analyze.ipynb
   ├─ promo_analyze.ipynb
   ├─ return_analyze.ipynb
   ├─ seasonality.ipynb
   ├─ stocks_profit.ipynb
   ├─ visualize_raw_data.ipynb
   └─ web_affect.ipynb

```
## Modeling Approach
Daily Data
    └─► Monthly Aggregation          # reduce noise, fix lag NaN problem
            └─► Feature Engineering  # calendar + Fourier + promo + YoY lags
                    └─► LGB + XGB + CatBoost (TimeSeriesSplit CV)
                            └─► Weighted Ensemble
                                    └─► Daily Disaggregation (historical share)
                                            └─► submission.csv ✅

## Setup & Usage
1. Clone & install dependencies
```bash
git clone https://github.com/TranDinhQuy0801/DATATHON2026_anmiquangcatuan/tree/main
cd DATATHON2026
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the prediction pipeline
```bash
# LightGBM model
python Prediction/Thuat_toan_LGB_pure_model.py

# Trimmed mean model
python Prediction/Thuat_toan_Trimmed_mean_model.py

# Blend both into FINAL.csv
python Prediction/Blend_50_50.py
```
3. Output
DATATHON2026/Prediction/submission.csv — ready to submit to Kaggle

### Requirements
Install all: pip install -r requirements.txt

## License
This project is for educational/competition purposes.
