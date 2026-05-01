import pandas as pd
import os

def main():
    print("=" * 60)
    print(" BLENDING: 50% Trimmed + 50% LIGHTGBM ")
    print("=" * 60)

    # Đường dẫn
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_ok  = os.path.join(BASE_DIR, 'Trimmed_mean_model.csv')
    file_lgb = os.path.join(BASE_DIR, 'LGB_pure.csv')
    out_file = os.path.join(BASE_DIR, 'FINAL_BLEND_50_50.csv')

    # Kiểm tra file
    if not os.path.exists(file_ok) or not os.path.exists(file_lgb):
        print("❌ LỖI: Không tìm thấy file đầu vào!")
        print(f"   Tìm OK tại: {file_ok}")
        print(f"   Tìm LGB tại: {file_lgb}")
        return

    # Đọc dữ liệu
    df_ok  = pd.read_csv(file_ok)
    df_lgb = pd.read_csv(file_lgb)

    # Trộn 50/50
    df_final = df_ok.copy()
    df_final['Revenue'] = (df_ok['Revenue'] * 0.5) + (df_lgb['Revenue'] * 0.5)
    df_final['COGS']    = (df_ok['COGS']    * 0.5) + (df_lgb['COGS']    * 0.5)

    # Xuất file
    df_final.to_csv(out_file, index=False, float_format='%.6f')
    
    print(f"✅ Đã trộn xong!")
    print(f"   Bản đồ kết hợp: 50% Tính ổn định (OK) + 50% Thông minh (LGB)")
    print(f"   File kết quả: {out_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
