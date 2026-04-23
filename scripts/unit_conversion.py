import pandas as pd
import numpy as np
import glob
import os

# 換算係數：kmol/hr -> m³/hr，依據莫耳密度 0.05637 kmol/m³
# 1 / 0.05637 ≈ 17.740
KMOL_TO_M3_FACTOR = 1.0 / 0.05637   # ≈ 17.740

# 換算係數：kg/hr -> m³/hr (air_SP 的情況)，依據截圖 mass density 1.58083 kg/m³
KG_TO_M3_FACTOR = 1.0 / 1.58083     # ≈ 0.63258

# 要換算的欄位 (分為兩類)
COLS_KMOL_TO_M3 = ['acidgas_Fm', 'air']
COLS_KG_TO_M3 = ['air_SP']

# 轉換後的變數重新命名對應表
RENAME_MAP = {
    'acidgas_Fm': 'acidgas_Fv',
    'air': 'air_m3',
    'air_SP': 'air_SP_m3'
}

# Step change 資料夾
STEP_CHANGE_DIRS = [
    'data/Claus_dynamic/step_change/in_training_distribution',
    'data/Claus_dynamic/step_change/out_of_training_distribution',
    'data/Claus_dynamic/step_change/acidgas_fm=170',
]

# 訓練資料原始檔 (只抓非 _converted)
TRAIN_DATA_PATTERN = 'data/Claus_dynamic/Test_dataform_change_air2_R=*.csv'


def convert_files(file_list, label=''):
    print(f"\n{'='*60}")
    print(f"{label}  ({len(file_list)} 個原始檔)")
    print(f"{'='*60}")
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            converted_cols = []
            acidgas_fv_exists = 'acidgas_Fv' in df.columns
            
            # kmol/hr -> m3/hr
            for col in COLS_KMOL_TO_M3:
                if col == 'acidgas_Fm' and acidgas_fv_exists:
                    continue
                if col in df.columns:
                    df[col] = df[col] * KMOL_TO_M3_FACTOR
                    converted_cols.append(col)
                    
            # kg/hr -> m3/hr
            for col in COLS_KG_TO_M3:
                if col in df.columns:
                    df[col] = df[col] * KG_TO_M3_FACTOR
                    converted_cols.append(col)

            # 依據 RENAME_MAP 對轉換後的欄位重新命名
            renamed_cols = {}
            for old_col, new_col in RENAME_MAP.items():
                if old_col == 'acidgas_Fm' and acidgas_fv_exists:
                    continue
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    renamed_cols[old_col] = new_col

            if not converted_cols:
                print(f"  [SKIP] {os.path.basename(file_path)}: 找不到目標欄位")
                continue
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_converted{ext}"
            df.to_csv(output_path, index=False)
            print(f"  [OK] {os.path.basename(file_path)}  ->  {os.path.basename(output_path)}"
                  f"  (換算: {converted_cols})")
        except Exception as e:
            print(f"  [ERROR] {file_path}: {e}")


def main():
    print(f"換算係數 (kmol/hr -> m³/hr): {KMOL_TO_M3_FACTOR:.6f}")
    print(f"換算欄位 (kmol): {COLS_KMOL_TO_M3}")
    print(f"換算係數 (kg/hr -> m³/hr): {KG_TO_M3_FACTOR:.6f}")
    print(f"換算欄位 (kg): {COLS_KG_TO_M3}")

    # 1. 訓練資料
    all_train = sorted(glob.glob(TRAIN_DATA_PATTERN))
    orig_train = [f for f in all_train if '_converted' not in os.path.basename(f)]
    convert_files(orig_train, label='訓練資料 Test_dataform')

    # 2. Step change 資料
    for data_dir in STEP_CHANGE_DIRS:
        all_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        orig_files = [f for f in all_files if '_converted' not in os.path.basename(f)]
        convert_files(orig_files, label=f'Step change: {data_dir}')

    print("\n完成！  ⚠️  訓練資料已重新換算，需重新訓練模型。")


if __name__ == "__main__":
    main()
