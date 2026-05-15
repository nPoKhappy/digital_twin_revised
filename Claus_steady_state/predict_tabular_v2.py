import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import calculate_metrics

SCATTER_COLOR = "#1f77b4"
IDEAL_COLOR = "#d62728"


def style_prediction_axis(ax, ticks=5):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ticks))
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.9, colors="#1f2937")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f2937")
        spine.set_linewidth(1.0)

# =====================================================================
# 1. 完整標籤對接清單 (HYSYS 原始標籤 -> Python 模型中使用的標籤)
# =====================================================================
FULL_MAPPING = {
    # ── 1. 設定點與特殊對應 (Setpoints & Custom Logic) ──
    'B34.SPo.SPo': 'acidgas_Fm',
    'B17.PV.PV': 'air',
    'S20.P.P': 'HEATER1_output_P',
    'B33.SPo.SPo': 'air2_SP',
    'B17.SPo.SPo': 'air_SP',
    'B35.SPo.SPo': 'COG_SP',
    'AIR2.Fv.Fv': 'second_air2',
    'S4.Fv.Fv': 'COG',

    # ── 2. 反應器與壓力控制 (Reactors & PC) ──
    'B18.SPo.SPo': 'burner_input_T_SP',
    'B18.PV.PV': 'burner_input_T_PV',
    'B19.SPo.SPo': 'burner_output_T_SP',
    'B19.PV.PV': 'burner_output_T_PV',
    'BURNER_PC.SPo.SPo': 'burner_output_P_SP',
    'BURNER_PC.PV.PV': 'burner_output_P_PV',
    'FURANCE_PC.SPo.SPo': 'fur_outputP_SP',
    'FURANCE_PC.PV.PV': 'fur_outputP_PV',
    'FURANCE.T.0.(0)': 'fur_inputT',
    'FURANCE.T.1.(1)': 'fur_temp',

    # ── 3. 分離器與換熱設備 (SEPs & Heaters) ──
    'SEP1_PC.SPo.SPo': 'SEP1_P_SP',
    'SEP1_PC.PV.PV': 'SEP1_P_PV',
    'SEP1.T.T': 'SEP1_T',
    'SEP2_PC.SPo.SPo': 'SEP2_P_SP',
    'SEP2_PC.PV.PV': 'SEP2_P_PV',
    'SEP2.T.T': 'SEP2_T',
    'SEP3_PC.SPo.SPo': 'SEP3_P_SP',
    'SEP3_PC.PV.PV': 'SEP3_P_PV',
    'SEP3.T.T': 'SEP3_T',
    'B21.SPo.SPo': 'HEATER1_output_T_SP',
    'B21.PV.PV': 'HEATER1_output_T_PV',
    'B20.SPo.SPo': 'HEATER2_output_T_SP',
    'B20.PV.PV': 'HEATER2_output_T_PV',
    'CAT1_PC.SPo.SPo': 'cat1_output_P_SP',
    'CAT1_PC.PV.PV': 'cat1_output_P_PV',
    'CAT2_PC.SPo.SPo': 'cat2_output_P_SP',
    'CAT2_PC.PV.PV': 'cat2_output_P_PV',

    # ── 4. 流道數據校閱 (Streams) ──
    'S12.F.F': 'fur_F',
    'S12.P.P': 'fur_inputP',
    'S15.T.T': 'fur_outputT',
    'S16.F.F': 'WHB_F',
    'S16.P.P': 'WHB_inputP',
    'S16.T.T': 'WHB_inputT',
    'S13.T.T': 'WHB_outputT',
    'S13.P.P': 'WHB_outputP',
    'S36.F.F': 'HEATER1_F',
    'S36.P.P': 'HEATER1_input_P',
    'S36.T.T': 'HEATER1_input_T',
    'S21.F.F': 'cat1_F',
    'S21.P.P': 'cat1_input_P',
    'S21.T.T': 'cat1_input_temp',
    'S22.T.T': 'cat1_output_temp',
    'S25.F.F': 'HEATER2_F',
    'S25.P.P': 'HEATER2_input_P',
    'S25.T.T': 'HEATER2_input_T',
    'S27.F.F': 'cat2_F',
    'S27.P.P': 'cat2_input_P',
    'S27.T.T': 'cat2_input_temp',
    'S28.T.T': 'cat2_output_temp',
    'S14.F.F': 'SEP1_F',
    'S23.F.F': 'SEP2_F',
    'S29.F.F': 'SEP3_F',

    # ── 5. 組成與進料 (Composition) ──
    'ACIDGAS.T.T': 'acidgas_T',
    'ACIDGAS.P.P': 'acidgas_P',
    'ACIDGAS.Fcn.H2O.("H2O")': 'acidgas_H2O',
    'ACIDGAS.Fcn.H2S.("H2S")': 'acidgas_H2S',
    'ACIDGAS.Fcn.CO2.("CO2")': 'acidgas_CO2',
    'S33.Zn.SO2.("SO2")': 'B35_SO2',
    'S33.Zn.H2S.("H2S")': 'B35_H2S',
    'S8.P.P': 'burner_inputP'
}

INV_MAPPING = {v: k for k, v in FULL_MAPPING.items()}

class SimpleTabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleTabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    de_mv = [
        'air2_SP',
        'HEATER2_output_T_SP',
        'acidgas_Fm',
        'acidgas_P',
        'acidgas_T',
    ]
    target_cols = [
        'B35_H2S',
        'B35_SO2',
    ]

    excel_input_cols = [INV_MAPPING[col] for col in de_mv]
    excel_target_cols = [INV_MAPPING[col] for col in target_cols]

    print("--- 載入獨立預留的 Test 測試集 ---")
    out_dir = os.path.join(base_dir, 'results', 'Tabular_MLP_New')
    test_csv_path = os.path.join(out_dir, 'tabular_test_dataset.csv')
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"找不到測試資料集 {test_csv_path}。請先使用 train_tabular_v2.py 進行訓練，它會自動保留 10% 的資料當作完全獨立的 Test 測試集。")

    df = pd.read_csv(test_csv_path)
    
    df_x = df[de_mv].copy().astype('float32')
    df_y = df[target_cols].copy().astype('float32')

    print(f"測試樣本數總計: {len(df_x)}")

    # 讀取 preprocessing stats
    mean_path = os.path.join(base_dir, 'results', 'Tabular_MLP_New', 'zscore_mean.csv')
    std_path = os.path.join(base_dir, 'results', 'Tabular_MLP_New', 'zscore_std.csv')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError("找不到正規化算出的 zscore_mean.csv 與 zscore_std.csv！請先確定 train_tabular_v2.py 已經跑完。")

    mean_all = pd.read_csv(mean_path, index_col=0).squeeze("columns")
    std_all = pd.read_csv(std_path, index_col=0).squeeze("columns")

    # zscore 正規化
    x_mean = mean_all[de_mv]
    x_std = std_all[de_mv].replace(0, 1)
    df_x_z = (df_x - x_mean) / x_std

    y_mean = mean_all[target_cols].values
    y_std = std_all[target_cols].replace(0, 1).values

    # 模型載入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTabularMLP(input_dim=len(de_mv), output_dim=len(target_cols)).to(device)
    
    model_path = os.path.join(base_dir, 'saved_models', 'Tabular_MLP_5in_2out_QV.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重: {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_tensor = torch.tensor(df_x_z.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_z = model(X_tensor).cpu().numpy()

    # 轉回原始尺度
    y_pred = y_pred_z * y_std + y_mean
    y_true = df_y.values

    # 輸出資料夾
    out_dir = os.path.join(base_dir, 'results', 'Tabular_MLP_New')
    os.makedirs(out_dir, exist_ok=True)
    
    out_csv = os.path.join(out_dir, 'tabular_predictions.csv')
    pd.DataFrame(np.hstack([y_true, y_pred]),
                 columns=target_cols + [f'{c}_pred' for c in target_cols]).to_csv(out_csv, index=False)

    print("\n--- 預測指標 (總測試集) ---")
    metrics = []
    for i, name in enumerate(target_cols):
        y_true_var = y_true[:, i]
        y_pred_var = y_pred[:, i]
        
        res = calculate_metrics(y_true_var, y_pred_var)
        res['Variable'] = name
        metrics.append(res)
        print(f"[{name}] MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}, R2={res['R2']:.4f}, MAPE={res['MAPE']:.2f}%")    
        
        # Parity Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true_var, y_pred_var, color=SCATTER_COLOR, alpha=0.62, s=16, edgecolors='none')
        min_val = min(y_true_var.min(), y_pred_var.min())
        max_val = max(y_true_var.max(), y_pred_var.max())
        margin = (max_val - min_val) * 0.05
        if margin == 0:
            margin = 1.0
        plot_min = min_val - margin
        plot_max = max_val + margin
        ax.plot([plot_min, plot_max], [plot_min, plot_max], color=IDEAL_COLOR, linestyle='--', lw=1.6)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name} Parity Pattern\nRMSE={res["RMSE"]:.4f}, R2={res["R2"]:.4f}', fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        style_prediction_axis(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'parity_plot_{name}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    metrics_df = pd.DataFrame(metrics)[['Variable','MAE','RMSE','R2','MAPE']]
    metrics_df.to_csv(os.path.join(out_dir, 'tabular_metrics.csv'), index=False)
    
    print(f"\n所有預測結果跟圖表都已存入: {out_dir}")

if __name__ == '__main__':
    main()
