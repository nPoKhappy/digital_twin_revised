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
    except (KeyError, ValueError) as e:
        print(f"注意：使用預設索引載入: {e}")
        df_raw_test = pd.read_csv(os.path.join(cfg_data['path'], test_cfg['filename']))
    
    # Apply point limit
    limit_point = test_cfg.get('point', None)
    if limit_point:
        df_raw_test = df_raw_test.iloc[:limit_point]
        
    df_raw_test.dropna(inplace=True)
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)

    # [Added] Downsampling to match training logic
    interval = cfg_data.get('sampling_interval_min', config['window'].get('sampling_interval_min', 1))
    if interval > 1:
        print(f"Downsampling test data by interval: {interval}")
        df_z_test = df_z_test.iloc[::interval].reset_index(drop=True)
        print(f"  New test data length: {len(df_z_test)}")

    # --- Step 2: 執行預測 ---
    inference_strategy = config['training'].get('inference_strategy', 'sliding_window')
    print(f"策略: {inference_strategy}")
    
    initial_history_np = df_z_test.iloc[0:W][en_mv_and_sv].values
    
    # [Modify] Inference Strategy: Replace Future PVs with Future SPs in Decoder Input
    # 這是為了模擬真實預測場景：未來只有 SP 已知，PV 未知。
    # 我們將 de_mv 中的 PV 欄位值替換為對應 SP 欄位的值。
    # 這樣模型雖然以為它吃的是 PV，但其實吃的是 SP。
    
    df_future_input = df_z_test.iloc[W:].copy()
    
    # 定義替換規則: {PV_Column: SP_Column}
    # 根據變量配置邏輯 (Claus Process)
    pv_sp_map = {
        'HEATER2_output_T_PV': 'HEATER2_output_T_SP',
        'second_air2': 'air2_SP',
        # 'HEATER1_output_T_PV': 'T1_SP' # 如果有HEATER1的SP也可以加
    }
    
    print("\n[Inference Setup] Preparing Decoder Inputs (Future Knowns)...")
    for pv_col, sp_col in pv_sp_map.items():
        if pv_col in de_mv:  # 只有在 de_mv 中有 PV 欄位時才進行替換
            if sp_col in df_z_test.columns:
                print(f"  -> Replacing Future '{pv_col}' with '{sp_col}' values.")
                df_future_input[pv_col] = df_z_test.iloc[W:][sp_col].values
            else:
                print(f"  Warning: SP column '{sp_col}' not found. Using original '{pv_col}' (Ideal Testing).")
    
    future_mvs_np = df_future_input[de_mv].values
    true_targets_np = df_z_test.iloc[W:][y_sv].values

    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    future_de_inputs = torch.tensor(future_mvs_np, dtype=torch.float32).unsqueeze(0)

    if inference_strategy == 'block_replacement':
        predictions_z = predict_block_replacement(model, initial_en_input, future_de_inputs, device, config)
    elif inference_strategy == 'sliding_window':
        # 注意：這裡假設 num_output 等於 y_sv 的長度
        predictions_z = predict_sliding_window(model, initial_en_input, future_de_inputs, device, len(y_sv))
    else:
        raise ValueError(f"未知的推理策略: '{inference_strategy}'")

    num_actual_preds = predictions_z.shape[1]
    true_targets_np = true_targets_np[:num_actual_preds, :]

    # --- Step 3: 保存與評估 ---
    prefix = config['exp_name']
    results_dir = os.path.join('./results/', prefix, test_name)
    os.makedirs(results_dir, exist_ok=True)

    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    
    predictions_cov = predictions_np * y_std + y_mean
    true_targets_cov = true_targets_np * y_std + y_mean

    # 保存 CSV
    df_true = pd.DataFrame(true_targets_cov, columns=y_sv)
    df_pred = pd.DataFrame(predictions_cov, columns=[f"{col}_pred" for col in y_sv])
    df_results = pd.concat([df_true, df_pred], axis=1)
    df_results.to_csv(os.path.join(results_dir, 'prediction_results.csv'), index=False)

    # 計算指標
    metrics_results = []
    print("\n[評估結果]")
    for i, name in enumerate(y_sv):
        metrics = calculate_metrics(true_targets_cov[:, i], predictions_cov[:, i])
        metrics['Variable'] = name
        metrics_results.append(metrics)
        print(f"  {name}: R²={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    df_metrics = pd.DataFrame(metrics_results)
    df_metrics = df_metrics[['Variable', 'MAE', 'RMSE', 'R2', 'MAPE']]
    df_metrics.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)

    # 繪圖
    print("正在繪圖...")
    for i, name in enumerate(y_sv):
        var_metrics = metrics_results[i]
        plt.figure(figsize=(20, 6))
        plt.plot(true_targets_cov[:, i], label='True', color='blue')
        plt.plot(predictions_cov[:, i], label='Pred', color='red', linestyle='--')
        title = f'{name} ({test_name})\nR2={var_metrics["R2"]:.4f}, MAPE={var_metrics["MAPE"]:.2f}%'
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(results_dir, f'{name}.png'))
        plt.close()
    
    print(f"完成測試: {test_name}. 結果存在: {results_dir}")


def main(config_path):
    # --- Step 0: 載入設定 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    print(f"========== 實驗: {prefix} ==========")
    
    # --- 全域準備: 載入訓練數據的統計量 (mean/std) 與變數選擇 ---
    # 因為 Z-score 必須使用 Training set 的 mean/std 來 transform Test set
    print("\n[Init] 載入訓練數據以獲取統計量...")
    cfg_data = config['data']
    
    # 這裡只為了拿 mean/std，所以簡單讀取
    training_file = cfg_data['training_files'][0] if 'training_files' in cfg_data else cfg_data['filename']
    try:
        df_train = data_utils.load_data(os.path.join(cfg_data['path'], training_file))
    except:
        df_train = pd.read_csv(os.path.join(cfg_data['path'], training_file))
    
    df_train.dropna(inplace=True)
    mean_all, std_all = data_utils.calculate_zscore_stats(df_train)
    
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
