import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# 設定中文字型 (以免亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_and_compare(config_path, plant_data_path):
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
    flow_factor = 18.86
    
    # AcidGas Flow: kmol/hr -> Nm3/hr (using 18.86 factor)
    if 'acidgas_Fm' in df_train.columns:
        df_train['acidgas_Fm'] = df_train['acidgas_Fm'] * flow_factor
        
    # Air Flow: User confirmed Training Data is already m3/hr? 
    # Or at least requested no conversion for Air.
    # Training (223) vs Plant (857). No 18.86 factor applied.
    
    # Pressure: bar -> kPa
    if 'acidgas_P' in df_train.columns:
        df_train['acidgas_P'] = df_train['acidgas_P'] * 100.0
        
    # Composition: Mole Frac -> %
    for c in ['B35_H2S', 'B35_SO2']:
        if c in df_train.columns:
            df_train[c] = df_train[c] * 100.0
            
    # 2. Load Plant Data
    print(f"\n載入現場數據 (Plant Data): {plant_data_path}...")
    df_plant = pd.read_csv(plant_data_path)
    df_plant.dropna(inplace=True)
    df_plant['Source'] = 'Plant Data (W251)'
    
    # Data Cleaning for Plant Data: Convert to Numeric just in case
    # User requested 5 specific variables (AcidGas Flow/Temp/Pres, Air2, T2)
    cols_to_compare = [
        'acidgas_Fm', 
        'acidgas_T', 
        'second_air2', 
        'HEATER2_output_T_PV'
    ]
    
    for c in cols_to_compare:
        df_plant[c] = pd.to_numeric(df_plant[c], errors='coerce')
        
    df_plant.dropna(subset=cols_to_compare, inplace=True)
    print(f"  Plant Data valid rows: {len(df_plant)}")

    # 3. Combine for Plotting
    # Filter only relevant columns + Source
    common_cols = [c for c in cols_to_compare if c in df_train.columns and c in df_plant.columns]
    
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
    config_file = 'configs/GRU_Aligned_Transformer_Interval10_median.yaml'
    # Correct path to the processed plant data
    plant_file = 'data/Claus_dynamic/Claus_plant_data/W251_500area_processed_for_GRU.csv'
    
    analyze_and_compare(config_file, plant_file)
