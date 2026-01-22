import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob
from src.utils import load_data, calculate_zscore_stats, apply_zscore
from src.variable_selection import variable_selection

def analyze_distribution():
    # 1. 設定檔案路徑
    base_dir = "./data/Claus_dynamic/"
    
    # 定義四個獨立的檔案
    # 定義檔案配置
    file_configs = [
        {"name": "Train (R5 Part)", "file": "R5_Train_Part.csv", "group": "Train", "color": "blue", "marker": "circle"},
        {"name": "ID Test (R5 Part)", "file": "R5_ID_Test_Part.csv", "group": "ID", "color": "cyan", "marker": "circle"}, # ID is same distribution
        # OOD Datasets
        {"name": "OOD (R5-1)", "file": "Test_dataform_change_air2_R=5-1.csv", "group": "OOD", "color": "red", "marker": "diamond"},
        {"name": "OOD (R5-2)", "file": "Test_dataform_change_air2_R=5-2.csv", "group": "OOD", "color": "orange", "marker": "diamond"},
        {"name": "OOD (R5-3)", "file": "Test_dataform_change_air2_R=5-3.xlsx", "group": "OOD", "color": "magenta", "marker": "cross"},
        {"name": "OOD (R5-4)", "file": "Test_dataform_change_air2_R=5-4.xlsx", "group": "OOD", "color": "purple", "marker": "cross"},
        {"name": "OOD (R5-5)", "file": "Test_dataform_change_air2_R=5-5.xlsx", "group": "OOD", "color": "yellow", "marker": "cross"},
        {"name": "OOD (R5-6)", "file": "Test_dataform_change_air2_R=5-6.csv", "group": "OOD", "color": "brown", "marker": "diamond"},
    ]
    
    # 2. 變數定義 (切換回 8 變數)
    variables_num = 8
    try:
        de_mv, y_sv, _, en_mv_and_sv = variable_selection(variables_num)
    except:
        # Fallback if 8 is not defined or error, use a hardcoded list for analysis
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
                marker=dict(size=3, color=cfg['color'], opacity=0.5, symbol=cfg['marker'])
            )
            traces.append(trace)

        layout = go.Layout(
            title=f'3D PCA Data Distribution (4 Files Separate)<br>Total Var: {sum(pca.explained_variance_ratio_):.2%}',
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
