# predict.py - Long-term rolling prediction using trained models with sliding window or block replacement strategies (動態數據)
import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 確保你的項目結構中有這些模組
from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.utils import calculate_metrics

# ==============================================================================
# --- 核心預測函數 ---
# ==============================================================================

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
        for t in tqdm(range(num_pred_steps), desc="[策略: 滑動窗口] 預測中"):
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model(current_en_input, single_step_de_input)
            
            # Check if model returns full prediction (like iTransformer) or single step
            if single_step_prediction.shape[1] > 1:
                # Direct step prediction (The model predicts whole future at once)
                # In sliding window context, we just take the first step, or we should switch strategy.
                # Here we assume the user intends to use sliding window, so we take the first step.
                single_step_prediction = single_step_prediction[:, 0, :].unsqueeze(1)
            
            predictions[:, t, :] = single_step_prediction

            # 滑動窗口：移除最舊的一步，加入新的一步
            next_en_input_history = current_en_input[:, 1:, :]
            # 合併順序：[MV, y_sv] 以匹配 en_mv_and_sv 的順序
            new_step_features = torch.cat([single_step_de_input, single_step_prediction], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions


def predict_block_replacement(model, initial_en_input, future_de_inputs, device, config):
    """Block replacement 預測策略
    
    每個窗口：
    1. 用 encoder 輸入 + 當前窗口 MV 預測整個窗口的 y_sv
    2. 將 [MV, y_sv] 合併，作為下一個窗口的 encoder 輸入（整塊替換）
    """
    model.eval()
    
    H = config['window']['train_window_mins'] // config['window']['sampling_interval_min']
    num_pred_steps = future_de_inputs.shape[1]
    
    if num_pred_steps % H != 0:
        print(f"警告: 預測總步數 {num_pred_steps} 不是窗口大小 {H} 的整數倍。")
        num_pred_steps = (num_pred_steps // H) * H
        print(f"預測將只進行到 {num_pred_steps} 步 (最後一個完整窗口)。")
        future_de_inputs = future_de_inputs[:, :num_pred_steps, :]

    num_windows_to_predict = num_pred_steps // H
    predictions_all_windows = []
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for i in tqdm(range(num_windows_to_predict), desc="[策略: 塊替換] 預測中"):
            start_idx = i * H
            end_idx = (i + 1) * H
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)

            prediction_block = model(current_en_input, de_input_block)
            predictions_all_windows.append(prediction_block)

            # 合併順序：[MV, y_sv] 以匹配 en_mv_and_sv 的順序
            current_en_input = torch.cat([de_input_block, prediction_block], dim=2)

    return torch.cat(predictions_all_windows, dim=1)

def predict_horizon_reinit(model, initial_en_input, future_de_inputs, future_targets, full_en_inputs, device, config):
    """
    Horizon Re-initialization Strategy:
    At each step H (prediction horizon), we RESET the encoder input history 
    using the GROUND TRUTH history from 'full_en_inputs'.
    This simulates MPC behavior where at each decision point, we have access to the true past state.
    """
    model.eval()
    predictions_all = []
    
    # Get parameters
    weights = config['training']['loss_weighting']['weights']
    num_windows = len(weights) # Usually 1
    total_pred_len = future_de_inputs.shape[1] # Total steps to predict
    
    # H is the block size for one prediction call
    H = config['window']['prediction_length'] 
    
    # Current history tensor (starts with initial)
    W = initial_en_input.shape[1]
    
    # Calculate Reset Interval (e.g. 18 * 4 = 72 steps)
    reinit_interval_steps = H * num_windows
    
    total_steps = future_de_inputs.shape[1]
    
    # Calculate active windows based on weights
    # e.g. [1, 0, 0, 0] -> Last active index 0 -> Predict 1 block (18 steps)
    # e.g. [1, 1, 0, 0] -> Last active index 1 -> Predict 2 blocks (36 steps)
    last_active_idx = 0
    for idx, w in enumerate(weights):
        if w > 0:
            last_active_idx = idx
    
    num_H_blocks = last_active_idx + 1
    
    print(f"  [Horizon Reinit] Weights={weights} -> Active Blocks={num_H_blocks} ({num_H_blocks * H} steps).")

    current_en_input = None 
    
    with torch.no_grad():
        for i in range(num_H_blocks):
            # Current Global Step Start relative to T_start (W)
            global_step = i * H
            
            # Reset Logic: 
            # i=0 -> Reset (Use Ground Truth)
            # i>0 -> AR Update
            # Since we stop at num_windows, we only reset once at the beginning.
            should_reset = (i == 0)
            
            # 1. Get DE Input for this H-block
            # If we run out of future data, stop
            if global_step >= total_steps:
                break

            end_step = min(global_step + H, total_steps)
            de_input_block = future_de_inputs[:, global_step:end_step, :].to(device)
            
            if should_reset:
                # Reset from Ground Truth History
                # Indexing into full_en_inputs (which starts at T=0)
                # Prediction starts at T=W.
                # History needed for block 0 (predicting T=W..W+H) is T=0..W
                
                # Check bounds just in case
                if global_step + W > full_en_inputs.shape[1]:
                     break
                     
                current_en_input = full_en_inputs[:, global_step : global_step + W, :].to(device)
                
            else:
                # Autoregressive Update
                prev_pred = predictions_all[-1]
                prev_de_input = future_de_inputs[:, global_step-H : global_step, :].to(device)
                
                # Construct new history chunk [MV, SV]
                new_hist_chunk = torch.cat([prev_de_input, prev_pred], dim=2)
                
                # Shift Left and Append
                current_en_input = torch.cat([current_en_input[:, H:, :], new_hist_chunk], dim=1)

            # 2. Model Prediction
            pred = model(current_en_input, de_input_block)
            predictions_all.append(pred)
            
    if not predictions_all:
        return torch.tensor([])
        
    return torch.cat(predictions_all, dim=1)

# ==============================================================================
# --- 主程式 ---
# ==============================================================================

def run_prediction(config, test_cfg, model, device, mean_all, std_all, en_mv_and_sv, de_mv, y_sv, W):
    """執行單一測試集的預測與評估"""
    test_name = test_cfg.get('name', 'Default_Test')
    print(f"\n========== 正在測試: {test_name} ==========")
    print(f"檔案: {test_cfg['filename']}")

    # --- Step 1: 準備測試數據 ---
    cfg_data = config['data']
    try:
        df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], test_cfg['filename']))
        print("成功載入測試數據（帶 DateTime 索引）")
    except (KeyError, ValueError, FileNotFoundError) as e:
        # Try finding in parent dir or absolute path
        fpath = test_cfg['filename']
        if not os.path.exists(fpath):
             fpath = os.path.join(cfg_data['path'], test_cfg['filename'])
        
        try:
            df_raw_test = pd.read_csv(fpath)
            print(f"成功使用 pandas 讀取: {fpath}")
        except Exception as e2:
             print(f"無法讀取檔案: {e2}")
             return

    # Apply point limit
    limit_point = test_cfg.get('point', None)
    if limit_point:
        df_raw_test = df_raw_test.iloc[:limit_point]
        
    # [Step 1: Downsample FIRST] - consistent with training
    interval = cfg_data.get('sampling_interval_min', config['window'].get('sampling_interval_min', 1))
    use_median = config['window'].get('use_median_downsampling', True)

    if interval > 1:
        if use_median:
            print(f"Downsampling test data by MEDIAN resampling: interval={interval}")
            # New Logic: Rolling Median + Slice
            df_median = df_raw_test.rolling(window=interval, min_periods=interval).median()
            df_raw_test = df_median.iloc[interval-1::interval].reset_index(drop=True)
            print(f"  -> Applied Rolling Median Filter (Window={interval})")
        else:
            print(f"Downsampling test data by SIMPLE SLICING: interval={interval}")
            df_raw_test = df_raw_test.iloc[::interval].reset_index(drop=True)
            print(f"  -> Applied Index Slicing (Step={interval})")
        
        print(f"  New test data length: {len(df_raw_test)}")
    
    df_raw_test.dropna(inplace=True)

    # [Step 2: Log Transform] - consistent with training
    # 注意：必須確保這裡變數名稱對應，目前是 B35_H2S, B35_SO2
    target_cols = ['B35_H2S', 'B35_SO2']
    # 檢查這些列是否存在
    valid_log_cols = [c for c in target_cols if c in df_raw_test.columns]
    if valid_log_cols:
        print(f"Applying Log Transform to {valid_log_cols}")
        df_raw_test = data_utils.apply_log_transform(df_raw_test, valid_log_cols)

    # [Step 3: Robust Scaling]
    # Note: run_prediction receives 'mean_all' and 'std_all'.
    # Since training script saved Median to 'zscore_mean.csv' and IQR to 'zscore_std.csv',
    # we can use them directly. Ideally we should use apply_robust_scale explicitly.
    # df_z_test = (df_raw_test - mean_all) / std_all (using the passed args)
    
    # We use apply_zscore for clarity if available
    # mean_all here is MEAN, std_all here is STD
    print("Applying Z-score Scaling (Mean/Std)...")
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)

    # ... (skipping unchanged lines) ...

    # 1. 準備歷史數據 (反標準化 + 反Log)
    # initial_history_np 是 Scaled 的 (W, Enc_Feat)
    df_hist_scaled = pd.DataFrame(initial_history_np, columns=en_mv_and_sv)
    df_hist = data_utils.inverse_zscore(df_hist_scaled, mean_all, std_all) 
    
    if valid_log_cols:
         df_hist = data_utils.inverse_log_transform(df_hist, valid_log_cols)
    
    for i, name in enumerate(y_sv):
        var_metrics = metrics_results[i]
        
        # 獲取該變數的歷史數據 (如果存在於 Encoder Input 中)
        if name in df_hist.columns:
            hist_vals = df_hist[name].values
        else:
            hist_vals = np.array([])
            
        future_true = true_targets_cov[:, i]
        future_pred = predictions_cov[:, i]
        
        # 拼接
        full_true = np.concatenate([hist_vals, future_true])
        # 對於預測線，歷史部分我們通常畫成真實值 (作為 Context)，或者不畫
        # 這裡我們畫成一條線：前段是 History(True)，後段是 Pred
        # 為了區分，我們分兩段畫
        
        plt.figure(figsize=(20, 6))
        
        # Plot History
        x_hist = range(len(hist_vals))
        plt.plot(x_hist, hist_vals, label='History', color='gray', alpha=0.7)
        
        # Plot Future
        x_future = range(len(hist_vals), len(hist_vals) + len(future_true))
        plt.plot(x_future, future_true, label='True (Future)', color='blue')
        plt.plot(x_future, future_pred, label='Pred (Future)', color='red', linestyle='--')
        
        # 連接點視覺優化 (讓 History 和 Pred 連起來)
        if len(hist_vals) > 0:
            plt.plot([x_hist[-1], x_future[0]], [hist_vals[-1], future_pred[0]], color='red', linestyle='--', alpha=0.5)
            plt.plot([x_hist[-1], x_future[0]], [hist_vals[-1], future_true[0]], color='blue', alpha=0.5)

        title = f'{name} ({test_name})\nR2={var_metrics["R2"]:.4f}, MAPE={var_metrics["MAPE"]:.2f}%'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, f'{name}.png'))
        plt.close()
    
    print(f"完成測試: {test_name}. 結果存在: {results_dir}")


def main(config_path):
    # --- Step 0: 載入設定 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    print(f"========== 實驗: {prefix} ==========")
    
    # --- 全域準備: 載入訓練數據的統計量 ---
    # 優先嘗試載入訓練時保存的 zscore_mean.csv (Median) 和 zscore_std.csv (IQR)
    zscore_mean_path = os.path.join('./results/', prefix, 'zscore_mean.csv')
    zscore_std_path = os.path.join('./results/', prefix, 'zscore_std.csv')
    
    cfg_data = config['data']
    
    if os.path.exists(zscore_mean_path) and os.path.exists(zscore_std_path):
        print(f"[Init] 載入已保存的統計量: {zscore_mean_path}")
        # index_col=0 is crucial because saved csv has variable names in first column
        mean_all = pd.read_csv(zscore_mean_path, index_col=0).squeeze()
        std_all = pd.read_csv(zscore_std_path, index_col=0).squeeze()
    else:
        print("[Init] 警告：未找到保存的統計量，正在從訓練數據重新計算 (確保與訓練流程一致)...")
        cfg_data = config['data']
        training_file = cfg_data['training_files'][0] if 'training_files' in cfg_data else cfg_data['filename']
        try:
            df_train = data_utils.load_data(os.path.join(cfg_data['path'], training_file))
        except:
            fpath = os.path.join(cfg_data['path'], training_file)
            if not os.path.exists(fpath): # Try local
                 fpath = training_file
            df_train = pd.read_csv(fpath)
            
        df_train.dropna(inplace=True)
        
        # 關鍵：必須先做 Log Transform 再計算 Stats！
        target_cols = ['B35_H2S', 'B35_SO2']
        valid_log_cols = [c for c in target_cols if c in df_train.columns]
        if valid_log_cols:
             print(f"  Doing Log Transform on {valid_log_cols} before stats calc...")
             df_train = data_utils.apply_log_transform(df_train, valid_log_cols)
             
        # 計算 Robust Stats (Median/IQR)
        # 注意：雖然變數名叫 mean_all/std_all，但內容其實是 Median/IQR
        mean_all, std_all = data_utils.calculate_robust_stats(df_train)
        print("  重新計算完成。")

    # 變數選擇
    de_mv, y_sv, _, en_mv_and_sv = variable_selection(cfg_data['variables_num'])
    
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    
    # --- 載入模型 ---
    print(f"\n[Init] 載入模型...")
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    model = get_model(config)
    model_path = os.path.join('./saved_models/', f'{prefix}.pth')
    
    if not os.path.exists(model_path):
        print(f"錯誤: 模型檔案不存在: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # --- 執行多個測試 ---
    test_suites = config['data'].get('inference_files', [])
    
    # 如果 Yaml 沒定義 inference_files，就用舊的 test_data
    if not test_suites:
        print("未發現 inference_files，使用預設 test_data")
        default_test = config['data']['test_data']
        default_test['name'] = 'Default_Test_Set'
        test_suites = [default_test]
        
    for test_cfg in test_suites:
        run_prediction(config, test_cfg, model, device, mean_all, std_all, en_mv_and_sv, de_mv, y_sv, W)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="執行預測")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
