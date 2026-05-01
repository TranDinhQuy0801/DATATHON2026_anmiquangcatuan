# 🛒 DATATHON 2026 — E-Commerce Sales Forecasting

> Predict daily **Revenue** and **COGS** for a Vietnamese e-commerce company using 10 years of historical sales data (2012–2022).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green)](https://lightgbm.readthedocs.io)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://kaggle.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📁 Project Structure

```
DATATHON2026/
│
├── dataset/                        # Raw competition data
│   └── sample_submission.csv       # Target dates to predict
│
├── part1/                          # EDA & analysis notebooks
│   ├── q1sol.ipynb                 # Q1: Revenue & COGS cash flow analysis
│   ├── q2sol.ipynb                 # Q2: Promotion frequency heatmap
│   ├── q3sol.ipynb                 # Q3: Campaign revenue vs profit analysis
│   ├── q4sol.ipynb                 # Q4: Inventory capital trend
│   ├── q5sol.ipynb                 # Q5: Supply-demand dynamics
│   ├── q6sol.ipynb                 # Q6: Fill rate & days of supply
│   ├── q7sol.ipynb                 # Q7: Stock on hand vs flow
│   ├── q8sol.ipynb                 # Q8: Overstock & stockout risk
│   ├── q9sol.ipynb                 # Q9: Sell-through rate by category
│   └── q10sol.ipynb                # Q10: ...
│
├── Prediction/                     # Forecasting pipeline & outputs
│   ├── Thuat_toan_LGB_pure_model.py       # LightGBM standalone model
│   ├── Thuat_toan_Trimmed_mean_model.py   # Trimmed-mean statistical model
│   ├── Blend_50_50.py                     # 50/50 ensemble blending script
│   ├── LGB_pure.csv                       # LGB model predictions
│   ├── Trimmed_mean_model.csv             # Trimmed-mean predictions
│   └── FINAL.csv                          # ✅ Final blended submission
│
├── baseline.ipynb                  # Baseline model (starting point)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🔍 Key Insights from EDA

| Finding | Impact on Model |
|--------|----------------|
| Strong annual seasonality with mid-year & year-end peaks | → Fourier features + 12-month lags |
| Revenue & COGS declined post-2019 | → `post_2019` trend flag |
| 6 promotion types with fixed monthly windows | → Promo binary features |
| Rural Special & Urban Blowout run in **odd years only** | → `is_odd_year` × promo interaction |
| COGS tightly tracks Revenue (~75–85% ratio) | → Chain prediction (COGS uses predicted Revenue) |

---

## 🧠 Modeling Approach

```
Daily Data
    └─► Monthly Aggregation          # reduce noise, fix lag NaN problem
            └─► Feature Engineering  # calendar + Fourier + promo + YoY lags
                    └─► LGB + XGB + CatBoost (TimeSeriesSplit CV)
                            └─► Weighted Ensemble
                                    └─► Daily Disaggregation (historical share)
                                            └─► submission.csv ✅
```

### Models
- **LightGBM** — primary model (weight: 50%)
- **XGBoost** — secondary (weight: 25%)
- **CatBoostRegressor** — tertiary (weight: 25%)
- **Trimmed Mean** — statistical baseline blended at 50/50

### Features
- Cyclical encoding: `sin/cos` for month, day-of-year
- Trend: `days since start`, `trend²`, `post_2019` flag
- Fourier terms: 4 harmonics over 12-month period
- Lag features: 12-month & 24-month same-calendar-month lags
- Year-over-year growth ratio
- 6 promotion flags + `is_odd_year` interactions

---

## ⚙️ Setup & Usage

### 1. Clone & install dependencies
```bash
git clone https://github.com/yourname/datathon2026.git
cd datathon2026
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the prediction pipeline
```bash
# LightGBM model
python Prediction/Thuat_toan_LGB_pure_model.py

# Trimmed mean model
python Prediction/Thuat_toan_Trimmed_mean_model.py

# Blend both into FINAL.csv
python Prediction/Blend_50_50.py
```

### 3. Output
`Prediction/FINAL.csv` — ready to submit to Kaggle.

---

## 📦 Requirements

```
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
optuna>=3.0          # optional, for hyperparameter tuning
```

Install all: `pip install -r requirements.txt`

---

## 📊 Results

| Model | CV MAE | Kaggle Score |
|-------|--------|--------------|
| LGB Pure | — | 1,059,327 |
| Trimmed Mean | — | — |
| **Blend 50/50** | — | **TBD** |

> 💡 *Update this table after each submission.*

---

## 👤 Author

**Đình Quý Trần**
- 📧 your.email@example.com
- 🔗 [GitHub](https://github.com/yourname) · [LinkedIn](https://linkedin.com/in/yourname)

---

## 📄 License

This project is for educational/competition purposes.
