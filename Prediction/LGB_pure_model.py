import pandas as pd
import numpy as np
import os
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')

# Đường dẫn tương đối
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
UNIFIED_PATH = os.path.join(ROOT_DIR, 'dataset', 'unified_sales_data.csv')
TEST_PATH    = os.path.join(ROOT_DIR, 'dataset', 'sample_submission.csv')
OUTPUT_DIR   = os.path.join(ROOT_DIR, 'Prediction')

def make_features(df):
    df = df.copy()
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Day']          = df['Date'].dt.day
    df['DOW']          = df['Date'].dt.dayofweek
    df['DOY']          = df['Date'].dt.dayofyear
    df['Quarter']      = df['Date'].dt.quarter
    df['WeekOfYear']   = df['Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend']    = (df['DOW'] >= 5).astype(int)
    df['IsMonthEnd']   = (df['Day'] >= 28).astype(int)
    df['IsMonthEnd2']  = (df['Day'] >= 25).astype(int)
    df['IsMonthStart'] = (df['Day'] <= 3).astype(int)
    df['DayOfMonthSin']= np.sin(2 * np.pi * df['Day'] / 31)
    df['DayOfMonthCos']= np.cos(2 * np.pi * df['Day'] / 31)
    for k in [1, 2, 3]:
        df[f'Sin_DOY_{k}'] = np.sin(2 * np.pi * k * df['DOY'] / 365.25)
        df[f'Cos_DOY_{k}'] = np.cos(2 * np.pi * k * df['DOY'] / 365.25)
    df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Sin_DOW']   = np.sin(2 * np.pi * df['DOW'] / 7)
    df['Cos_DOW']   = np.cos(2 * np.pi * df['DOW'] / 7)

    return df

FEATURE_COLS = [
    'Month', 'Day', 'DOW', 'DOY', 'Quarter', 'WeekOfYear',
    'IsWeekend', 'IsMonthEnd', 'IsMonthEnd2', 'IsMonthStart',
    'DayOfMonthSin', 'DayOfMonthCos',
    'Sin_DOY_1', 'Cos_DOY_1', 'Sin_DOY_2', 'Cos_DOY_2', 'Sin_DOY_3', 'Cos_DOY_3',
    'Sin_Month', 'Cos_Month', 'Sin_DOW', 'Cos_DOW',
]

def train_lgb(X_tr, y_tr, X_val, y_val):
    params = {
        'objective': 'regression_l1',
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model

def main():
    print("=" * 60)
    print(" LIGHTGBM PURE MODEL (TRAINING ONLY) ")
    print("=" * 60)

    df = pd.read_csv(UNIFIED_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df_all = df[~df['Date'].dt.year.isin([2020, 2021])].copy()
    df_all = make_features(df_all)

    # Train/Val Split (2022 làm val)
    df_tr  = df_all[df_all['Year'] != 2022].copy()
    df_val = df_all[df_all['Year'] == 2022].copy()

    # Normalize
    ym_rev = df_all.groupby('Year')['Revenue'].mean()
    ym_cogs = df_all.groupby('Year')['COGS'].mean()
    for s in [df_tr, df_val, df_all]:
        s['YM_Rev'] = s['Year'].map(ym_rev); s['YM_COGS'] = s['Year'].map(ym_cogs)
        s['NormRev'] = s['Revenue'] / s['YM_Rev']; s['NormCOGS'] = s['COGS'] / s['YM_COGS']

    # Train
    model_rev = train_lgb(df_tr[FEATURE_COLS], df_tr['NormRev'], df_val[FEATURE_COLS], df_val['NormRev'])
    model_cogs = train_lgb(df_tr[FEATURE_COLS], df_tr['NormCOGS'], df_val[FEATURE_COLS], df_val['NormCOGS'])

    # Predict Test
    df_test = pd.read_csv(TEST_PATH)
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test = make_features(df_test)
    
    # Anchor: Trích xuất mốc mùa cao điểm bằng Median-Split
    latest_year = df_all['Year'].max()
    df_latest = df_all[df_all['Year'] == latest_year]
    
    # Lấy trung bình nửa trên của năm gần nhất (Ra ~4.47M)
    q50_threshold = df_latest['Revenue'].median()
    base_revenue = df_latest[df_latest['Revenue'] >= q50_threshold]['Revenue'].mean()
    
    # Tự động trích xuất tỷ lệ COGS 0.872
    base_cogs_ratio = df_latest['COGS'].sum() / df_latest['Revenue'].sum()
    base_cogs = base_revenue * base_cogs_ratio
    
    # Neo kết quả của LightGBM vào mốc thực tế vừa tính
    pred_rev = model_rev.predict(df_test[FEATURE_COLS])
    pred_cogs = model_cogs.predict(df_test[FEATURE_COLS])
    
    scale_rev = base_revenue / pred_rev.mean()
    scale_cogs = base_cogs / pred_cogs.mean()

    out = pd.DataFrame({
        'Date': df_test['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': np.maximum(pred_rev * scale_rev, 0),
        'COGS': np.maximum(pred_cogs * scale_cogs, 0)
    })
    
    out_path = os.path.join(OUTPUT_DIR, 'LGB_pure.csv')
    out.to_csv(out_path, index=False, float_format='%.6f')
    print(f"Success! Pure LGB results saved to: {out_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()