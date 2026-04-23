import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` works when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_data, calculate_zscore_stats, apply_zscore
from src.variable_selection import variable_selection

def analyze_distribution():
    # 1. 設定檔案路徑
    base_dir = "./data/Claus_dynamic/"
    
    # 定義四個獨立的檔案
    # 定義檔案配置
    file_configs = [
        {"name": "Train (R5)", "file": "Test_dataform_change_air2_R=5_converted.csv", "group": "Train", "color": "blue", "marker": "circle"},
        # OOD Datasets
        {"name": "Train (R5-1)", "file": "Test_dataform_change_air2_R=5-1_converted.csv", "group": "OOD", "color": "red", "marker": "diamond"},
        {"name": "Train (R5-2)", "file": "Test_dataform_change_air2_R=5-2_converted.csv", "group": "OOD", "color": "orange", "marker": "diamond"},
        {"name": "Train (R5-6)", "file": "Test_dataform_change_air2_R=5-6_converted.csv", "group": "OOD", "color": "brown", "marker": "diamond"},
        {"name": "OOD (R5-7)", "file": "Test_dataform_change_air2_R=5-7_converted.csv", "group": "OOD", "color": "purple", "marker": "diamond"},
        {"name": "OOD (R5-8)", "file": "Test_dataform_change_air2_R=5-8_converted.csv", "group": "OOD", "color": "black", "marker": "diamond"},
        # OOD Step Datasets (out_of_training_distribution)
        {"name": "OOD_Step (80_190 air2)", "file": "step_change/out_of_training_distribution/air2_80_t2_190_air2_change_10_converted.csv", "group": "OOD_Step", "color": "darkblue", "marker": "x"},
        {"name": "OOD_Step (80_190 TR2)", "file": "step_change/out_of_training_distribution/air2_80_t2_190_TR2_change_10_converted.csv", "group": "OOD_Step", "color": "blue", "marker": "x"},
        {"name": "OOD_Step (100_190 air2)", "file": "step_change/out_of_training_distribution/air2_100_t2_190_air2_change_10_converted.csv", "group": "OOD_Step", "color": "darkred", "marker": "x"},
        {"name": "OOD_Step (100_190 TR2)", "file": "step_change/out_of_training_distribution/air2_100_t2_190_TR2_change_10_converted.csv", "group": "OOD_Step", "color": "red", "marker": "x"},
        {"name": "OOD_Step (400_190 air2)", "file": "step_change/out_of_training_distribution/air2_400_t2_190_air2_change_10_converted.csv", "group": "OOD_Step", "color": "darkgreen", "marker": "x"},
        {"name": "OOD_Step (400_190 TR2)", "file": "step_change/out_of_training_distribution/air2_400_t2_190_TR2_change_10_converted.csv", "group": "OOD_Step", "color": "green", "marker": "x"},
        {"name": "OOD_Step (500_190 air2)", "file": "step_change/out_of_training_distribution/air2_500_t2_190_air2_change_10_converted.csv", "group": "OOD_Step", "color": "darkorange", "marker": "x"},
        {"name": "OOD_Step (500_190 TR2)", "file": "step_change/out_of_training_distribution/air2_500_t2_190_TR2_change_10_converted.csv", "group": "OOD_Step", "color": "orange", "marker": "x"},
        # ID Step Datasets (in_training_distribution)
        {"name": "ID_Step (180_150 air2)", "file": "step_change/in_training_distribution/air2_180_t2_150_air2_change_10_converted.csv", "group": "ID_Step", "color": "green", "marker": "cross"},
        {"name": "ID_Step (180_150 TR2)", "file": "step_change/in_training_distribution/air2_180_t2_150_TR2_change_10_converted.csv", "group": "ID_Step", "color": "darkgreen", "marker": "cross"},
        {"name": "ID_Step (190_155 air2)", "file": "step_change/in_training_distribution/air2_190_t2_155_air2_change_-5_converted.csv", "group": "ID_Step", "color": "magenta", "marker": "cross"},
        {"name": "ID_Step (190_155 t2)", "file": "step_change/in_training_distribution/air2_190_t2_155_t2_change_-5_converted.csv", "group": "ID_Step", "color": "darkmagenta", "marker": "cross"},
        {"name": "ID_Step (190_230 air2)", "file": "step_change/in_training_distribution/air2_190_t2_230_air2_change_-5_converted.csv", "group": "ID_Step", "color": "cyan", "marker": "cross"},
        {"name": "ID_Step (190_230 t2)", "file": "step_change/in_training_distribution/air2_190_t2_230_t2_change_-5_converted.csv", "group": "ID_Step", "color": "darkcyan", "marker": "cross"},
        {"name": "ID_Step (200_210 air2)", "file": "step_change/in_training_distribution/air2_200_t2_210_air2_change_10_converted.csv", "group": "ID_Step", "color": "purple", "marker": "cross"},
        {"name": "ID_Step (200_210 TR2)", "file": "step_change/in_training_distribution/air2_200_t2_210_TR2_change_10_converted.csv", "group": "ID_Step", "color": "indigo", "marker": "cross"},
        {"name": "ID_Step (240_170 air2)", "file": "step_change/in_training_distribution/air2_240_t2_170_air2_change_10_converted.csv", "group": "ID_Step", "color": "orange", "marker": "cross"},
        {"name": "ID_Step (240_170 TR2)", "file": "step_change/in_training_distribution/air2_240_t2_170_TR2_change_10_converted.csv", "group": "ID_Step", "color": "darkorange", "marker": "cross"},
        {"name": "ID_Step (270_155 air2)", "file": "step_change/in_training_distribution/air2_270_t2_155_air2_change_-5_converted.csv", "group": "ID_Step", "color": "yellow", "marker": "cross"},
        {"name": "ID_Step (270_155 t2)", "file": "step_change/in_training_distribution/air2_270_t2_155_t2_change_-5_converted.csv", "group": "ID_Step", "color": "gold", "marker": "cross"},
        {"name": "ID_Step (270_230 air2)", "file": "step_change/in_training_distribution/air2_270_t2_230_air2_change_-5_converted.csv", "group": "ID_Step", "color": "lime", "marker": "cross"},
        {"name": "ID_Step (270_230 t2)", "file": "step_change/in_training_distribution/air2_270_t2_230_t2_change_-5_converted.csv", "group": "ID_Step", "color": "greenyellow", "marker": "cross"},
        {"name": "ID_Step (280_210 air2)", "file": "step_change/in_training_distribution/air2_280_t2_210_air2_change_10_converted.csv", "group": "ID_Step", "color": "pink", "marker": "cross"},
        {"name": "ID_Step (280_210 TR2)", "file": "step_change/in_training_distribution/air2_280_t2_210_TR2_change_10_converted.csv", "group": "ID_Step", "color": "hotpink", "marker": "cross"},
    ]
    
    # 2. 變數定義 (切換為 72 變數)
    variables_num = 72
    try:
        de_mv, y_sv, _, en_mv_and_sv = variable_selection(variables_num)
    except:
        # Fallback if 72 is not defined or error, use a hardcoded list for analysis
        en_mv_and_sv = ['acidgas_Fm', 'acidgas_T', 'acidgas_P', 'HEATER1_output_T_PV', 
                        'HEATER2_output_T_PV', 'second_air2', 'B35_H2S', 'B35_SO2']

    all_vars = en_mv_and_sv 
    print(f"分析變數 ({len(all_vars)}個): {all_vars}")

    # 3. 載入數據
    data_store = {} # 用來存原始 df
    
    print("\n[Step 1] Loading separate files...")
    for cfg in file_configs:
        path = os.path.join(base_dir, cfg['file'])
        print(f"Loading {cfg['name']} from {path}...")
        try:
            # Using updated load_data from utils which supports xlsx
            df = load_data(path)
        except Exception as e:
            print(f"Error loading {path} with load_data: {e}")
            try:
                if path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
            except Exception as e2:
                print(f"Failed to load {path}: {e2}")
                continue
                
        # Remap columns for plant_simulated_data to match 72 variables config
        rename_map = {
            'air_SP': 'air_SP_m3',
            'air': 'air_m3',
            'acidgas_Fm': 'acidgas_Fv'
        }
        df = df.rename(columns=rename_map)
        
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure all columns exist
        missing_cols = [c for c in all_vars if c not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols} in {cfg['name']}. Skipping.")
            continue
            
        df = df[all_vars].dropna()
        if len(df) > 10000: # Sampling for visualization performance
            df = df.sample(10000)
            
        data_store[cfg['name']] = df

    # 用由於 Training file 計算統計量
    train_dfs = [data_store[cfg['name']] for cfg in file_configs if cfg['group'] == 'Train' and cfg['name'] in data_store]
    
    if not train_dfs:
        print("Error: No training data found for PCA fitting!")
        return

    df_train_all = pd.concat(train_dfs, axis=0)
    print(f"PCA Fitted on {len(df_train_all)} samples (Train group).")
    
    mean = df_train_all.mean()
    std = df_train_all.std()

    # 標準化函數
    def standardize(df):
        return (df - mean) / (std + 1e-8)
    
    # 4. 執行 PCA
    print("\n[Step 3] Running PCA (3D)...")
    z_train_all = standardize(df_train_all)
    
    pca = PCA(n_components=3)
    pca.fit(z_train_all)
    
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}")

    # 5. 繪圖 (Plotly)
    try:
        import plotly.graph_objects as go
        print("\n[Step 5] Generating Interactive 3D Plot (Plotly)...")
        
        traces = []
        for cfg in file_configs:
            name = cfg['name']
            if name not in data_store:
                continue
            df = data_store[name]
            z_data = standardize(df)
            pca_data = pca.transform(z_data)
            
            # Random sample if too large
            if len(pca_data) > 5000:
                indices = np.random.choice(len(pca_data), size=5000, replace=False)
                pca_data = pca_data[indices]
                
            trace = go.Scatter3d(
                x=pca_data[:, 0], y=pca_data[:, 1], z=pca_data[:, 2],
                mode='markers',
                name=name,
                marker=dict(size=3, color=cfg['color'], opacity=0.5, symbol=cfg.get('marker', 'circle'))
            )
            traces.append(trace)

        layout = go.Layout(
            title=f'3D PCA Data Distribution (All combined)<br>Total Var: {sum(pca.explained_variance_ratio_):.2%}',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        fig_html = go.Figure(data=traces, layout=layout)
        html_path = 'data_distribution_pca_3d_separate.html'
        fig_html.write_html(html_path)
        print(f"互動式 HTML 已保存至: {html_path}")
        
    except ImportError:
        print("警告: 未安裝 plotly")

if __name__ == "__main__":
    analyze_distribution()
