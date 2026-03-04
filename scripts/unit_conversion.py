import pandas as pd
import numpy as np
import glob
import os

# 換算係數：kmol/hr -> m³/hr，依據莫耳密度 0.05637 kmol/m³
# 1 / 0.05637 ≈ 17.740
FLOW_FACTOR = 1.0 / 0.05637   # ≈ 17.740

# 要換算的欄位
COLS_TO_CONVERT = ['acidgas_Fm', 'air']

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
            for col in COLS_TO_CONVERT:
                if col in df.columns:
                    df[col] = df[col] * FLOW_FACTOR
                    converted_cols.append(col)
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
    print(f"換算係數: 1 / 0.05637 = {FLOW_FACTOR:.6f}  (kmol/hr -> m³/hr)")
    print(f"換算欄位: {COLS_TO_CONVERT}")

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
