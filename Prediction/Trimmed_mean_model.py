import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Sử dụng đường dẫn tương đối để đảm bảo tính tái lập
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

UNIFIED_PATH = os.path.join(ROOT_DIR, 'dataset', 'unified_sales_data.csv')
TEST_PATH    = os.path.join(ROOT_DIR, 'dataset', 'sample_submission.csv')
OUTPUT_DIR   = os.path.join(ROOT_DIR, 'Prediction')

def trimmed_mean(group):
    vals = group.sort_values()
    if len(vals) >= 4:
        # Thuật toán Trimmed Mean: Bỏ nhiễu Min/Max
        return vals.iloc[1:-1].mean()
    return vals.mean()

def main():
    print("=" * 60)
    print(" ALGORITHM: TRIMMED MEAN PROFILING (Month/Day) ")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(UNIFIED_PATH)
    df['Date']  = pd.to_datetime(df['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day']   = df['Date'].dt.day
    
    # Loại bỏ năm COVID để lấy profile sạch
    df = df[~df['Year'].isin([2020, 2021])].copy()

    # 2. Normalize các năm về cùng mặt bằng 2022
    ref_mean = df[df['Year'] == 2022]['Revenue'].mean()
    yearly_means = df.groupby('Year')['Revenue'].mean()
    df = df.merge(yearly_means.rename('YM'), on='Year')
    df['NR'] = df['Revenue'] * (ref_mean / df['YM'])

    # 3. Tạo Seasonal Shape dựa trên Month/Day
    profile = df.groupby(['Month', 'Day'])['NR'].apply(trimmed_mean).reset_index()
    profile.columns = ['Month', 'Day', 'BaseShape']

    # 4. Dự báo trên Test Set
    df_test = pd.read_csv(TEST_PATH)
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test['Month'] = df_test['Date'].dt.month
    df_test['Day']   = df_test['Date'].dt.day

    final = df_test.merge(profile, on=['Month', 'Day'], how='left')
    final['BaseShape'] = final['BaseShape'].fillna(final['BaseShape'].mean())

    # 5. 
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year]
    
    # Kỹ thuật Median-Split: Lọc lấy 50% các ngày có doanh thu cao nhất năm gần nhất
    q50_threshold = df_latest['Revenue'].median()
    df_high_season = df_latest[df_latest['Revenue'] >= q50_threshold]
    
    # Tự động tính Scale (Sẽ ra xấp xỉ 4.47M)
    dynamic_scale = df_high_season['Revenue'].mean()
    
    # Tự động tính hệ số COGS của năm gần nhất (Sẽ ra đúng 0.872)
    dynamic_cogs_ratio = df_latest['COGS'].sum() / df_latest['Revenue'].sum()

    # Áp dụng hệ số vào dự báo
    mult = dynamic_scale / final['BaseShape'].mean()
    
    out = pd.DataFrame()
    out['Date']    = final['Date'].dt.strftime('%Y-%m-%d')
    out['Revenue'] = final['BaseShape'] * mult
    out['COGS']    = out['Revenue'] * dynamic_cogs_ratio
    
    out_path = os.path.join(OUTPUT_DIR, 'Trimmed_mean_model.csv')
    out.to_csv(out_path, index=False, float_format='%.6f')
    
    print(f"Success! Prediction saved to: {out_path}")
    print(f"Algorithm applied: Trimmed Mean (1-min, 1-max excluded)")
    print(f"Target Scale: {dynamic_scale:,.0f} VND/day")
    print("=" * 60)

if __name__ == '__main__':
    main()