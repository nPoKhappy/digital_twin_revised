# predict_step_change.py - 使用訓練好的 Transformer 模型預測 step change 數據
# 分別處理 in_training_distribution 和 out_of_training_distribution 的數據

import torch
import numpy as np
import pandas as pd
import os
import re
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.utils import calculate_metrics

# ==============================================================================
# --- 配置 ---
# ==============================================================================

CONFIG_PATH = "configs/transformer_layerwise_57var.yaml"  # default, override with --config
STEP_CHANGE_BASE_DIR = "data/Claus_dynamic/step_change"
OUTPUT_BASE_DIR = "results/step_change_predictions"

# 分類目錄
DISTRIBUTION_DIRS = {
    'in_training': 'in_training_distribution',
    'out_of_training': 'out_of_training_distribution',
    'acidgas_fm_170': 'acidgas_fm=170',
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
                        de_mv, y_sv, en_mv_and_sv, W, H, inference_strategy, warmup_steps=0):
    """處理單一 step change 檔案
    
    Args:
        warmup_steps: encoder 初始窗口往後滑移的步數。
                      sliding_window 下，pre-step 完全 autoregressive，de_mv 不變時 drift 慢。
    """
    
    filename = os.path.basename(filepath)
    
    # 讀取數據
    df_raw = pd.read_csv(filepath)

    # --- 降採樣 (與 train pipeline 一致) ---
    interval = config['window'].get('sampling_interval_min', 1)
    use_median = config['window'].get('use_median_downsampling', True)
    if interval > 1:
        if use_median:
            df_raw = df_raw.rolling(window=interval, min_periods=interval).median(numeric_only=True)
            df_raw = df_raw.iloc[interval-1::interval].reset_index(drop=True)
        else:
            df_raw = df_raw.iloc[::interval].reset_index(drop=True)
        df_raw.dropna(inplace=True)

    # 檢查是否有足夠的數據
    if len(df_raw) < W + H:
        print(f"  [SKIP] 數據長度 {len(df_raw)} 不足，需要至少 {W + H} 行")
        return None
    
    # 檢查是否有必要的欄位
    all_needed_cols = list(set(en_mv_and_sv + de_mv + y_sv))
    missing_cols = [c for c in all_needed_cols if c not in df_raw.columns]
    if missing_cols:
        print(f"  [SKIP] 缺少欄位: {missing_cols}")
        return None
    
    # 只取需要的欄位進行正規化，避免不相關欄位的 NaN 問題
    df_subset = df_raw[all_needed_cols].copy()

    # --- 單位換算 (僅當讀取的是原始檔而非 _converted 檔時才換算)
    # acidgas_Fm 和 air: kmol/hr -> m3/hr
    # 係數 1/0.05637 ≈ 17.740 (莫耳密度 0.05637 kmol/m³)
    if '_converted' not in filepath:
        FLOW_CONVERSION = 1.0 / 0.05637  # ≈ 17.740
        for col in ['acidgas_Fm', 'air']:
            if col in df_subset.columns:
                df_subset[col] = df_subset[col] * FLOW_CONVERSION

    # 檢查是否有 NaN
    if df_subset.isnull().any().any():
        print(f"  [WARN] 數據包含 NaN，嘗試填補...")
        df_subset = df_subset.ffill().bfill()

    # --- Log Transform (與 train pipeline 一致) ---
    log_cols = [c for c in ['B35_H2S', 'B35_SO2'] if c in df_subset.columns]
    if log_cols:
        df_subset = data_utils.apply_log_transform(df_subset, log_cols)

    # 應用 z-score 正規化 (只對需要的欄位)
    mean_subset = mean_all[all_needed_cols]
    std_subset = std_all[all_needed_cols]
    
    # 避免除以零
    std_safe = std_subset.mask(std_subset.abs() < 1e-6, 1.0)
    df_z = (df_subset - mean_subset) / std_safe
    
    # ========== Warm-up：encoder 初始 = SS1 第一行複製 W 次 ==========
    # 在真實數據前，先插入 warmup_steps 步的 SS1 de_mv（複製第一行）
    # 讓模型全程 autoregressive 但 de_mv 一直不變，直到自然收斂到 SS1 穩態
    # 之後接上真實數據（含 step change），只記錄真實段的預測結果

    # z-score 空間的 SS1 值（第一行）
    ss1_en_z  = df_z.iloc[0][en_mv_and_sv].values   # shape (n_en,)
    ss1_de_z  = df_z.iloc[0][de_mv].values           # shape (n_de,)

    # 初始 encoder = SS1 第一行複製 W 次
    initial_history_np = np.tile(ss1_en_z, (W, 1))  # shape (W, n_en)

    # 前段：warmup_steps 步的 SS1 de_mv（複製）
    warmup_de_np = np.tile(ss1_de_z, (warmup_steps, 1))  # shape (warmup_steps, n_de)

    # 後段：真實數據的 de_mv（從 row 0 開始，包含 step change）
    real_de_np      = df_z.iloc[0:][de_mv].values
    real_targets_np = df_z.iloc[0:][y_sv].values

    # 合併：[warmup SS1 de_mv] + [真實 de_mv]
    full_de_np = np.concatenate([warmup_de_np, real_de_np], axis=0)

    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    full_de_inputs   = torch.tensor(full_de_np, dtype=torch.float32).unsqueeze(0)

    if warmup_steps > 0:
        print(f"  [WARM-UP] encoder=SS1×{W}，先跑 {warmup_steps} 步 SS1 de_mv 讓模型收斂")

    # 執行預測（全程 autoregressive，warmup_steps 步後才取結果）
    if inference_strategy == 'block_replacement':
        all_preds_z = predict_block_replacement(model, initial_en_input, full_de_inputs, device, H)
    else:
        all_preds_z = predict_sliding_window(model, initial_en_input, full_de_inputs, device, len(y_sv))

    # 只保留真實數據段的預測（丟棄 warmup 段）
    predictions_z   = all_preds_z[:, warmup_steps:, :]
    true_targets_np = real_targets_np

    num_actual_preds = predictions_z.shape[1]
    true_targets_np = true_targets_np[:num_actual_preds, :]
    
    # 反正規化
    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    # 避免除以零的情況
    y_std_safe = np.where(np.abs(y_std) < 1e-6, 1.0, y_std)
    predictions_cov = predictions_np * y_std_safe + y_mean
    true_targets_cov = true_targets_np * y_std_safe + y_mean

    # --- Inverse Log Transform ---
    log_cols_inv = [c for c in ['B35_H2S', 'B35_SO2'] if c in y_sv]
    if log_cols_inv:
        pred_df_tmp = pd.DataFrame(predictions_cov, columns=y_sv)
        true_df_tmp = pd.DataFrame(true_targets_cov, columns=y_sv)
        pred_df_tmp = data_utils.inverse_log_transform(pred_df_tmp, log_cols_inv)
        true_df_tmp = data_utils.inverse_log_transform(true_df_tmp, log_cols_inv)
        predictions_cov = pred_df_tmp.values
        true_targets_cov = true_df_tmp.values
    
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


def parse_scenario_conditions(filename):
    """從檔名解析操作條件，例如 air2_180_t2_150_air2_change_10.csv"""
    name = filename.replace('.csv', '')
    m = re.match(r'air2_(-?\d+)_t2_(-?\d+)_(\w+)_change_(-?\d+)', name)
    if m:
        return {
            'air2': int(m.group(1)),
            't2':   int(m.group(2)),
            'change_var': m.group(3),
            'change_val': int(m.group(4)),
        }
    return {'air2': '?', 't2': '?', 'change_var': '?', 'change_val': '?'}


def print_conditions_table(csv_files):
    """印出所有場景的操作條件表格"""
    print(f"\n{'No.':<5} {'air2':>6} {'t2':>6} {'change_var':<12} {'change_val':>10}  檔名")
    print('-' * 75)
    for i, f in enumerate(csv_files, 1):
        c = parse_scenario_conditions(f)
        print(f"{i:<5} {c['air2']:>6} {c['t2']:>6} {c['change_var']:<12} {c['change_val']:>10}  {f}")
    print('-' * 75)


def plot_combined_h2s_so2(dist_results, y_sv, dist_key, output_dir, exp_name):
    """繪製所有 step change 場景的 B35_H2S 和 B35_SO2 綜合圖，每 4 個場景一張圖"""
    key_vars = [v for v in ['B35_H2S', 'B35_SO2'] if v in y_sv]
    if not key_vars or not dist_results:
        return

    n_vars = len(key_vars)
    chunk_size = 4  # 每張圖最多 4 個場景
    dist_label = dist_key.replace('_', ' ').title()

    chunks = [dist_results[i:i + chunk_size] for i in range(0, len(dist_results), chunk_size)]

    for fig_idx, chunk in enumerate(chunks):
        n_rows = len(chunk)
        fig, axes = plt.subplots(n_rows, n_vars, figsize=(8 * n_vars, 4 * n_rows),
                                 squeeze=False)

        for row_idx, result in enumerate(chunk):
            cond = parse_scenario_conditions(result['filename'])
            cond_str = (f"air2={cond['air2']}  t2={cond['t2']}  "
                        f"Δ{cond['change_var']}={cond['change_val']:+d}")
            for col_idx, var in enumerate(key_vars):
                ax = axes[row_idx, col_idx]
                var_idx = y_sv.index(var)
                m = result['metrics'][var_idx]

                l1 = ax.plot(result['true_values'][:, var_idx], label='True (Aspen)',
                        color='steelblue', linewidth=1.5)
                ax_twin = ax.twinx()
                l2 = ax_twin.plot(result['predictions'][:, var_idx], label='Predicted (Right)',
                        color='tomato', linestyle='--', linewidth=1.5)

                ax.set_title(
                    f'[{cond_str}]\n{var}   R²={m["R2"]:.4f}  RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}',
                    fontsize=9)
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'{var} (True)')
                ax_twin.set_ylabel(f'{var} (Predicted)', color='tomato')
                ax_twin.tick_params(axis='y', labelcolor='tomato')
                
                lns = l1 + l2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)

        part_label = f'Part {fig_idx + 1}/{len(chunks)}'
        fig.suptitle(
            f'{exp_name}  —  {dist_label}  ({part_label})\nB35_H2S & B35_SO2  Step-Change Predictions',
            fontsize=13, y=1.01)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'H2S_SO2_combined_part{fig_idx + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  綜合圖已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Step Change Prediction")
    parser.add_argument('--config', type=str, default=CONFIG_PATH,
                        help='Path to YAML config file')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='encoder 初始窗口往後滑移的步數（sliding_window下 pre-step 完全autoregressive）')
    args = parser.parse_args()

    print("=" * 70)
    print("Step Change Prediction using Trained Transformer Model")
    print("=" * 70)
    
    # --- 載入配置 ---
    print(f"\n[1] 載入配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    prefix = config['exp_name']
    inference_strategy = config['training'].get('inference_strategy', 'sliding_window')
    print(f"    模型: {prefix}")
    print(f"    推理策略: {inference_strategy}")
    
    cfg_data = config['data']
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']  # encoder 歷史長度
    H = cfg_win['prediction_length']  # block size (單次預測步數)
    
    # --- 載入訓練數據統計 (從 train pipeline 儲存的 zscore stats) ---
    print(f"\n[2] 載入訓練數據統計...")
    prefix = config['exp_name']
    zscore_mean_path = os.path.join('./results', prefix, 'zscore_mean.csv')
    zscore_std_path  = os.path.join('./results', prefix, 'zscore_std.csv')

    if os.path.exists(zscore_mean_path) and os.path.exists(zscore_std_path):
        mean_all = pd.read_csv(zscore_mean_path, index_col=0).squeeze()
        std_all  = pd.read_csv(zscore_std_path,  index_col=0).squeeze()
        print(f"    已載入儲存的 zscore stats: {zscore_mean_path}")
    else:
        print(f"    [WARN] 找不到 {zscore_mean_path}，從訓練資料重新計算...")
        train_dfs = []
        for fname in cfg_data['training_files']:
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                df_t = pd.read_csv(fpath)
                interval = cfg_win.get('sampling_interval_min', 1)
                if interval > 1:
                    df_t = df_t.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                    df_t = df_t.iloc[interval-1::interval].reset_index(drop=True)
                df_t.dropna(inplace=True)
                train_dfs.append(df_t)
        df_all = pd.concat(train_dfs, ignore_index=True)
        log_cols = [c for c in ['B35_H2S', 'B35_SO2'] if c in df_all.columns]
        if log_cols:
            df_all = data_utils.apply_log_transform(df_all, log_cols)
        mean_all, std_all = data_utils.calculate_zscore_stats(df_all)

    # --- 獲取變數配置 ---
    de_mv, y_sv, _, en_mv_and_sv = variable_selection(cfg_data['variables_num'])
    
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
    output_base_dir = os.path.join('./results', prefix, 'step_change_predictions')
    print(f"\n[4] 開始處理 step change 數據...")
    print(f"    結果保存至: {output_base_dir}")

    all_results = {}
    
    for dist_key, dist_dir in DISTRIBUTION_DIRS.items():
        data_dir = os.path.join(STEP_CHANGE_BASE_DIR, dist_dir)
        output_dir = os.path.join(output_base_dir, dist_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"處理: {dist_key.upper()} ({dist_dir})")
        print(f"{'='*60}")
        
        # 找所有 CSV 檔案（只取原始檔，排除 _converted）
        csv_files = sorted([f for f in os.listdir(data_dir)
                            if f.endswith('.csv') and '_converted' not in f])
        print(f"找到 {len(csv_files)} 個檔案")
        print_conditions_table(csv_files)

        dist_results = []

        for csv_file in tqdm(csv_files, desc=f"預測 {dist_key}"):
            # 優先使用已預先換算的 _converted 檔
            base, ext = os.path.splitext(csv_file)
            converted_file = os.path.join(data_dir, f"{base}_converted{ext}")
            orig_file      = os.path.join(data_dir, csv_file)
            filepath = converted_file if os.path.exists(converted_file) else orig_file
            if os.path.exists(converted_file):
                tqdm.write(f"  [converted] {csv_file}")

            result = process_single_file(
                filepath, model, config, mean_all, std_all, device,
                de_mv, y_sv, en_mv_and_sv, W, H, inference_strategy,
                warmup_steps=args.warmup_steps
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
                
                l1 = ax.plot(result['true_values'][:, idx], label='True (Aspen)', color='blue', linewidth=1.2)
                ax_twin = ax.twinx()
                l2 = ax_twin.plot(result['predictions'][:, idx], label='Predicted (Right)', color='red', 
                        linestyle='--', linewidth=1.2)
                
                title = f'{var}\nR2={var_metrics["R2"]:.4f}, RMSE={var_metrics["RMSE"]:.4f}'
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Time Step')
                ax.set_ylabel(f'{var} (True)')
                ax_twin.set_ylabel(f'{var} (Predicted)', color='red')
                ax_twin.tick_params(axis='y', labelcolor='red')
                
                lns = l1 + l2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc='best', fontsize=8)
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
                    
                    fig, ax = plt.subplots(figsize=(14, 5))
                    l1 = ax.plot(result['true_values'][:, idx], label='True (Aspen)', color='blue', linewidth=1.5)
                    ax_twin = ax.twinx()
                    l2 = ax_twin.plot(result['predictions'][:, idx], label='Predicted (Right)', color='red', 
                             linestyle='--', linewidth=1.5)
                    
                    title = f'{csv_file.replace(".csv", "")} - {var}\n'
                    title += f'MAE={var_metrics["MAE"]:.6f}, RMSE={var_metrics["RMSE"]:.6f}, R2={var_metrics["R2"]:.4f}'
                    ax.set_title(title, fontsize=11)
                    ax.set_xlabel('Time Step (minutes)')
                    ax.set_ylabel(f'{var} (True)')
                    ax_twin.set_ylabel(f'{var} (Predicted)', color='red')
                    ax_twin.tick_params(axis='y', labelcolor='red')
                    
                    lns = l1 + l2
                    labs = [l.get_label() for l in lns]
                    ax.legend(lns, labs, loc='best')
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
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

            # 繪製 H2S & SO2 綜合圖
            plot_combined_h2s_so2(dist_results, y_sv, dist_key, output_dir, prefix)

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
    print(f"預測完成！結果保存至: {output_base_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
