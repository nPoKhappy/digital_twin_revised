# 這份腳本的目的是分析訓練數據（Aspen模擬）和現場數據（Plant Data）的分佈差異，並生成統計摘要和比較圖表。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml


def resolve_path(path_str):
    """Resolve file path robustly across different working directories."""
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    candidates = [
        path_str,  # current working directory
        os.path.join(os.path.dirname(__file__), path_str),  # scripts/
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path_str),  # repo root
    ]

    for p in candidates:
        p_norm = os.path.normpath(p)
        if os.path.exists(p_norm):
            return p_norm

    raise FileNotFoundError(f"找不到檔案: {path_str}\n已嘗試路徑: {candidates}")

# 設定中文字型 (以免亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_and_compare(config_path, plant_data_path):
    config_path = resolve_path(config_path)
    plant_data_path = resolve_path(plant_data_path)

    # 1. Load Config to find Training Data
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    cfg_data = config['data']
    data_path = cfg_data['path']
    train_files = cfg_data['training_files']
    
    print("載入訓練數據 (Training Data)...")
    train_dfs = []
    for fname in train_files:
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Apply same simple preprocessing if needed (e.g. dropna)
            df.dropna(inplace=True)
            train_dfs.append(df)
            print(f"  Loaded {fname}: {len(df)}")
    
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_train['Source'] = 'Training Data'
    
    # Convert Training Data (Aspen) to Plant Units for comparison
    print("Converting Training Data to Plant Units...")

    # AcidGas Flow:
    # NOTE: In current datasets, Training and Plant acidgas_Fm are already on comparable scale
    # (roughly 2,600~3,000). Do NOT apply 18.86 conversion here.
        
    # Air Flow: User confirmed Training Data is already m3/hr? 
    # Or at least requested no conversion for Air.
    # Training (223) vs Plant (857). No 18.86 factor applied.
    
    # Pressure: keep unit in bar for direct engineering interpretation.
    # Aspen training data is assumed absolute pressure (barA).

    # 2. Load Plant Data
    print(f"\n載入現場數據 (Plant Data): {plant_data_path}...")
    df_plant = pd.read_csv(plant_data_path)
    df_plant.dropna(inplace=True)
    df_plant['Source'] = 'Plant Data (W251)'

    # Column mapping (Plant -> Aspen naming)
    # Plant file uses cat2_input_temp as the counterpart of HEATER2_output_T_PV
    if 'HEATER2_output_T_PV' not in df_plant.columns and 'cat2_input_temp' in df_plant.columns:
        df_plant['HEATER2_output_T_PV'] = pd.to_numeric(df_plant['cat2_input_temp'], errors='coerce')
        print("  [Column Mapping] cat2_input_temp -> HEATER2_output_T_PV")
    
    # Data Cleaning for Plant Data: Convert to Numeric just in case
    # 比對欄位：新增 acidgas_P（表壓 + 1.01325 => 絕對壓）
    cols_to_compare = [
        'acidgas_Fm', 
        'acidgas_T', 
        'acidgas_P',
        'second_air2', 
        'HEATER2_output_T_PV'
    ]
    
    for c in cols_to_compare:
        if c in df_plant.columns:
            df_plant[c] = pd.to_numeric(df_plant[c], errors='coerce')

    # Plant acid gas pressure is gauge pressure. Convert to absolute bar (barA).
    # User confirmed plant_data unit is mbar(g): barA = (mbar_g + 1013.25) / 1000
    if 'acidgas_P' in df_plant.columns:
        p_series = pd.to_numeric(df_plant['acidgas_P'], errors='coerce')
        df_plant['acidgas_P'] = (p_series + 1013.25) / 1000.0
        print("  [Pressure Unit] Plant acidgas_P treated as mbar(g), converted to barA")

    existing_compare_cols_plant = [c for c in cols_to_compare if c in df_plant.columns]
    missing_compare_cols_plant = [c for c in cols_to_compare if c not in df_plant.columns]
    if missing_compare_cols_plant:
        print(f"  [Plant Missing Columns] {missing_compare_cols_plant}")

    missing_compare_cols_train = [c for c in cols_to_compare if c not in df_train.columns]
    if missing_compare_cols_train:
        print(f"  [Training Missing Columns] {missing_compare_cols_train}")

    df_plant.dropna(subset=existing_compare_cols_plant, inplace=True)
    print(f"  Plant Data valid rows: {len(df_plant)}")

    # 3. Combine for Plotting
    # Filter only relevant columns + Source
    common_cols = [c for c in cols_to_compare if c in df_train.columns and c in df_plant.columns]
    print(f"  Common compare columns: {common_cols}")
    
    df_combined = pd.concat([
        df_train[common_cols + ['Source']], 
        df_plant[common_cols + ['Source']]
    ], ignore_index=True)
    
    # 4. Statistical Summary
    print("\n========== 統計數據比對 (Mean ± Std) ==========")
    summary_train = df_train[common_cols].describe().T[['mean', 'std', 'min', 'max', '50%']]
    summary_plant = df_plant[common_cols].describe().T[['mean', 'std', 'min', 'max', '50%']]
    
    summary_compare = pd.concat([summary_train, summary_plant], axis=1, keys=['Training', 'Plant'])
    print(summary_compare)
    summary_compare.to_csv("distribution_summary_compare.csv")
    print("統計表已保存至 distribution_summary_compare.csv")

    # 5. Plotting Boxplots (Normalized / Raw)
    print("\n正在繪製分佈圖...")
    n_cols = 4
    n_rows = (len(common_cols) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, col in enumerate(common_cols):
        plt.subplot(n_rows, n_cols, i+1)
        sns.boxplot(x='Source', y=col, data=df_combined, showfliers=False) # Hide outliers for better scale
        plt.title(f'{col} 分佈比較')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('distribution_comparison_boxplot.png')
    print("分佈圖已保存至 distribution_comparison_boxplot.png")
    
    # 6. Plotting KDE (Density)
    plt.figure(figsize=(20, 5 * n_rows))
    for i, col in enumerate(common_cols):
        plt.subplot(n_rows, n_cols, i+1)
        sns.kdeplot(data=df_combined, x=col, hue='Source', fill=True, common_norm=False, alpha=0.3)
        plt.title(f'{col} 密度分佈 (KDE)')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('distribution_comparison_kde.png')
    print("密度圖已保存至 distribution_comparison_kde.png")

if __name__ == "__main__":
    config_file = 'configs/transformer_layerwise_AT_Rolling_Aligned_GRU_Interval10_median.yaml'
    # Correct path to the processed plant data
    plant_file = 'data/Claus_dynamic/Claus_plant_data/W251_500area_processed_Training_Data.csv'
    
    analyze_and_compare(config_file, plant_file)
