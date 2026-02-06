# predict_step_change.py - 使用訓練好的 Transformer 模型預測 step change 數據
# 分別處理 in_training_distribution 和 out_of_training_distribution 的數據

import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_utils
from src.models import get_model
from src.utils import calculate_metrics

# ==============================================================================
# --- 配置 ---
# ==============================================================================

CONFIG_PATH = "configs/transformer_experiment_AT_claus.yaml"
STEP_CHANGE_BASE_DIR = "data/Claus_dynamic/step_change"
OUTPUT_BASE_DIR = "results/step_change_predictions"

# 兩個分類目錄
DISTRIBUTION_DIRS = {
    'in_training': 'in_training_distribution',
    'out_of_training': 'out_of_training_distribution'
}

# ==============================================================================
# --- 預測函數 (與 predict.py 相同) ---
# ==============================================================================

def predict_block_replacement(model, initial_en_input, future_de_inputs, device, H):
    """Block replacement 預測策略
    
    每個窗口：
    1. 用 encoder 輸入 + 當前窗口 MV 預測整個窗口的 y_sv
    2. 將 [MV, y_sv] 合併，作為下一個窗口的 encoder 輸入（整塊替換）
    """
    model.eval()
    
    num_pred_steps = future_de_inputs.shape[1]
    
    if num_pred_steps % H != 0:
        num_pred_steps = (num_pred_steps // H) * H
        future_de_inputs = future_de_inputs[:, :num_pred_steps, :]

    num_windows_to_predict = num_pred_steps // H
    predictions_all_windows = []
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for i in range(num_windows_to_predict):
            start_idx = i * H
            end_idx = (i + 1) * H
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)
            prediction_block = model(current_en_input, de_input_block)
            predictions_all_windows.append(prediction_block)
            # 合併順序：[MV, y_sv] 以匹配 en_mv_and_sv 的順序
            current_en_input = torch.cat([de_input_block, prediction_block], dim=2)

    return torch.cat(predictions_all_windows, dim=1)


def predict_sliding_window(model, initial_en_input, future_de_inputs, device, num_output_features):
    """Sliding window 預測策略
    
    每一步：
    1. 用 encoder 歷史窗口 + 當前 MV 預測下一步的 y_sv
    2. 將 [MV, y_sv] 合併，加入 encoder 窗口（滑動）
    """
    model.eval()
    
    num_pred_steps = future_de_inputs.shape[1]
    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for t in range(num_pred_steps):
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model(current_en_input, single_step_de_input)
            predictions[:, t, :] = single_step_prediction

            # 滑動窗口：移除最舊的一步，加入新的一步
            next_en_input_history = current_en_input[:, 1:, :]
            # 合併順序：[MV, y_sv] 以匹配 en_mv_and_sv 的順序
            new_step_features = torch.cat([single_step_de_input, single_step_prediction], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions

# ==============================================================================
# --- 主程式 ---
# ==============================================================================

def process_single_file(filepath, model, config, mean_all, std_all, device, 
                        de_mv, y_sv, en_mv_and_sv, W, H, inference_strategy):
    """處理單一 step change 檔案"""
    
    filename = os.path.basename(filepath)
    
    # 讀取數據
    df_raw = pd.read_csv(filepath)
    
    # 檢查是否有足夠的數據
    if len(df_raw) <= W:
        print(f"  [SKIP] 數據長度 {len(df_raw)} 不足，需要至少 {W+1} 行")
        return None
    
    # 檢查是否有必要的欄位
    all_needed_cols = list(set(en_mv_and_sv + de_mv + y_sv))
    missing_cols = [c for c in all_needed_cols if c not in df_raw.columns]
    if missing_cols:
        print(f"  [SKIP] 缺少欄位: {missing_cols}")
        return None
    
    # 只取需要的欄位進行正規化，避免不相關欄位的 NaN 問題
    df_subset = df_raw[all_needed_cols].copy()
    
    # 檢查是否有 NaN
    if df_subset.isnull().any().any():
        print(f"  [WARN] 數據包含 NaN，嘗試填補...")
        df_subset = df_subset.fillna(method='ffill').fillna(method='bfill')
    
    # 應用 z-score 正規化 (只對需要的欄位)
    mean_subset = mean_all[all_needed_cols]
    std_subset = std_all[all_needed_cols]
    
    # 避免除以零
    std_safe = std_subset.replace(0, 1)
    df_z = (df_subset - mean_subset) / std_safe
    
    # 準備數據
    initial_history_np = df_z.iloc[0:W][en_mv_and_sv].values
    future_mvs_np = df_z.iloc[W:][de_mv].values
    true_targets_np = df_z.iloc[W:][y_sv].values
    
    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    future_de_inputs = torch.tensor(future_mvs_np, dtype=torch.float32).unsqueeze(0)
    
    # 執行預測
    if inference_strategy == 'block_replacement':
        predictions_z = predict_block_replacement(model, initial_en_input, future_de_inputs, device, H)
    else:
        predictions_z = predict_sliding_window(model, initial_en_input, future_de_inputs, device, len(y_sv))
    
    num_actual_preds = predictions_z.shape[1]
    true_targets_np = true_targets_np[:num_actual_preds, :]
    
    # 反正規化
    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    # 避免除以零的情況
    y_std_safe = np.where(y_std == 0, 1, y_std)
    predictions_cov = predictions_np * y_std_safe + y_mean
    true_targets_cov = true_targets_np * y_std_safe + y_mean
    
    # 檢查是否有 NaN 或 Inf
    if not np.isfinite(predictions_cov).all() or not np.isfinite(true_targets_cov).all():
        print(f"  [WARN] {filename}: 預測結果包含 NaN 或 Inf，跳過")
        return None
    
    # 計算指標
    metrics_results = []
    for i, name in enumerate(y_sv):
        y_true_col = true_targets_cov[:, i]
        y_pred_col = predictions_cov[:, i]
        
        # 跳過包含 NaN 或 Inf 的列
        if not np.isfinite(y_true_col).all() or not np.isfinite(y_pred_col).all():
            metrics = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
        else:
            metrics = calculate_metrics(y_true_col, y_pred_col)
        metrics['Variable'] = name
        metrics_results.append(metrics)
    
    return {
        'filename': filename,
        'predictions': predictions_cov,
        'true_values': true_targets_cov,
        'metrics': metrics_results,
        'y_sv': y_sv,
        'num_steps': num_actual_preds
    }


def main():
    print("=" * 70)
    print("Step Change Prediction using Trained Transformer Model")
    print("=" * 70)
    
    # --- 載入配置 ---
    print(f"\n[1] 載入配置: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    prefix = config['exp_name']
    inference_strategy = config['training'].get('inference_strategy', 'sliding_window')
    print(f"    模型: {prefix}")
    print(f"    推理策略: {inference_strategy}")
    
    cfg_data = config['data']
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']  # 窗口大小
    H = W  # block replacement 的塊大小
    
    # --- 載入訓練數據統計 ---
    print(f"\n[2] 載入訓練數據統計...")
    df_raw_train = pd.read_csv(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_raw_train.dropna(inplace=True)
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw_train)
    
    # --- 獲取變數配置 ---
    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    print(f"    Encoder 輸入變數: {len(en_mv_and_sv)}")
    print(f"    Decoder 輸入變數 (MV): {de_mv}")
    print(f"    預測目標變數: {len(y_sv)}")
    print(f"    窗口大小 W: {W}")
    
    # --- 載入模型 ---
    print(f"\n[3] 載入模型...")
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    
    model = get_model(config)
    model_path = os.path.join('./saved_models/', f'{prefix}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"    模型載入自: {model_path}")
    print(f"    運行設備: {device}")
    
    # --- 處理兩個分類目錄 ---
    print(f"\n[4] 開始處理 step change 數據...")
    
    all_results = {}
    
    for dist_key, dist_dir in DISTRIBUTION_DIRS.items():
        data_dir = os.path.join(STEP_CHANGE_BASE_DIR, dist_dir)
        output_dir = os.path.join(OUTPUT_BASE_DIR, dist_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"處理: {dist_key.upper()} ({dist_dir})")
        print(f"{'='*60}")
        
        # 找所有 CSV 檔案
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"找到 {len(csv_files)} 個檔案")
        
        dist_results = []
        
        for csv_file in tqdm(csv_files, desc=f"預測 {dist_key}"):
            filepath = os.path.join(data_dir, csv_file)
            
            result = process_single_file(
                filepath, model, config, mean_all, std_all, device,
                de_mv, y_sv, en_mv_and_sv, W, H, inference_strategy
            )
            
            if result is None:
                continue
            
            dist_results.append(result)
            
            # 保存個別檔案結果
            file_output_dir = os.path.join(output_dir, csv_file.replace('.csv', ''))
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 保存預測結果 CSV
            df_true = pd.DataFrame(result['true_values'], columns=y_sv)
            df_pred = pd.DataFrame(result['predictions'], columns=[f"{col}_pred" for col in y_sv])
            df_results = pd.concat([df_true, df_pred], axis=1)
            df_results.to_csv(os.path.join(file_output_dir, 'predictions.csv'), index=False)
            
            # 保存指標
            df_metrics = pd.DataFrame(result['metrics'])
            df_metrics = df_metrics[['Variable', 'MAE', 'RMSE', 'R2', 'MAPE']]
            df_metrics.to_csv(os.path.join(file_output_dir, 'metrics.csv'), index=False)
            
            # 繪製所有預測目標變數的綜合圖
            n_vars = len(y_sv)
            n_cols = 2
            n_rows = (n_vars + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
            axes = axes.flatten()
            
            for idx, var in enumerate(y_sv):
                ax = axes[idx]
                var_metrics = result['metrics'][idx]
                
                ax.plot(result['true_values'][:, idx], label='True (Aspen)', color='blue', linewidth=1.2)
                ax.plot(result['predictions'][:, idx], label='Predicted', color='red', 
                        linestyle='--', linewidth=1.2)
                
                title = f'{var}\nR2={var_metrics["R2"]:.4f}, RMSE={var_metrics["RMSE"]:.4f}'
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Time Step')
                ax.set_ylabel(var)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # 隱藏多餘的子圖
            for idx in range(n_vars, len(axes)):
                axes[idx].set_visible(False)
            
            fig.suptitle(f'{csv_file.replace(".csv", "")} - All Variables Prediction', fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(file_output_dir, 'all_variables.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # 單獨繪製關鍵變數的大圖 (B35_H2S, B35_SO2)
            key_vars = ['B35_H2S', 'B35_SO2']
            
            for var in key_vars:
                if var in y_sv:
                    idx = y_sv.index(var)
                    var_metrics = result['metrics'][idx]
                    
                    plt.figure(figsize=(14, 5))
                    plt.plot(result['true_values'][:, idx], label='True (Aspen)', color='blue', linewidth=1.5)
                    plt.plot(result['predictions'][:, idx], label='Predicted (Model)', color='red', 
                             linestyle='--', linewidth=1.5)
                    
                    title = f'{csv_file.replace(".csv", "")} - {var}\n'
                    title += f'MAE={var_metrics["MAE"]:.6f}, RMSE={var_metrics["RMSE"]:.6f}, R2={var_metrics["R2"]:.4f}'
                    plt.title(title, fontsize=11)
                    plt.xlabel('Time Step (minutes)')
                    plt.ylabel(var)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(file_output_dir, f'{var}.png'), dpi=150)
                    plt.close()
        
        all_results[dist_key] = dist_results
        
        # 匯總該分類的指標
        if dist_results:
            print(f"\n--- {dist_key.upper()} 匯總指標 ---")
            
            # 計算每個變數的平均指標
            summary_data = {var: {'MAE': [], 'RMSE': [], 'R2': [], 'MAPE': []} for var in y_sv}
            
            for result in dist_results:
                for m in result['metrics']:
                    var = m['Variable']
                    summary_data[var]['MAE'].append(m['MAE'])
                    summary_data[var]['RMSE'].append(m['RMSE'])
                    summary_data[var]['R2'].append(m['R2'])
                    summary_data[var]['MAPE'].append(m['MAPE'])
            
            # 創建匯總表
            summary_rows = []
            for var in y_sv:
                row = {
                    'Variable': var,
                    'MAE_mean': np.mean(summary_data[var]['MAE']),
                    'MAE_std': np.std(summary_data[var]['MAE']),
                    'RMSE_mean': np.mean(summary_data[var]['RMSE']),
                    'RMSE_std': np.std(summary_data[var]['RMSE']),
                    'R2_mean': np.mean(summary_data[var]['R2']),
                    'R2_std': np.std(summary_data[var]['R2']),
                    'MAPE_mean': np.mean(summary_data[var]['MAPE']),
                    'MAPE_std': np.std(summary_data[var]['MAPE'])
                }
                summary_rows.append(row)
                
                if var in ['B35_H2S', 'B35_SO2']:
                    print(f"  {var}:")
                    print(f"    MAE:  {row['MAE_mean']:.6f} +/- {row['MAE_std']:.6f}")
                    print(f"    RMSE: {row['RMSE_mean']:.6f} +/- {row['RMSE_std']:.6f}")
                    print(f"    R2:   {row['R2_mean']:.4f} +/- {row['R2_std']:.4f}")
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
            print(f"  匯總指標已保存至: {output_dir}/summary_metrics.csv")
    
    # --- 比較 in vs out of training distribution ---
    print(f"\n{'='*70}")
    print("比較: In-Training vs Out-of-Training Distribution")
    print(f"{'='*70}")
    
    for var in ['B35_H2S', 'B35_SO2']:
        if var not in y_sv:
            continue
        idx = y_sv.index(var)
        
        in_r2 = [r['metrics'][idx]['R2'] for r in all_results.get('in_training', [])]
        out_r2 = [r['metrics'][idx]['R2'] for r in all_results.get('out_of_training', [])]
        
        if in_r2 and out_r2:
            print(f"\n{var}:")
            print(f"  In-training R2:     {np.mean(in_r2):.4f} +/- {np.std(in_r2):.4f}")
            print(f"  Out-of-training R2: {np.mean(out_r2):.4f} +/- {np.std(out_r2):.4f}")
    
    print(f"\n{'='*70}")
    print("預測完成！結果保存至: " + OUTPUT_BASE_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
