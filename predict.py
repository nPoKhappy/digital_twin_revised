import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 確保你的項目結構中有這些模組
from src import data_utils
from src.models import get_model

# ==============================================================================
# --- 核心預測函數 ---
# ==============================================================================

def predict_sliding_window(model, initial_en_input, future_de_inputs, device, num_output_features):
    model.eval()
    
    num_pred_steps = future_de_inputs.shape[1]
    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for t in tqdm(range(num_pred_steps), desc="[策略: 滑動窗口] 預測中"):
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model(current_en_input, single_step_de_input)
            predictions[:, t, :] = single_step_prediction

            next_en_input_history = current_en_input[:, 1:, :]
            new_step_features = torch.cat([single_step_prediction, single_step_de_input], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions


def predict_block_replacement(model, initial_en_input, future_de_inputs, device, config):
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

            current_en_input = torch.cat([prediction_block, de_input_block], dim=2)

    return torch.cat(predictions_all_windows, dim=1)

# ==============================================================================
# --- 主程式 ---
# ==============================================================================

def main(config_path):
    # --- Step 0: 載入設定 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    inference_strategy = config['training'].get('inference_strategy', 'sliding_window')
    
    print(f"========== 開始預測: {prefix} (模型: {config['model']['name']}) ==========")
    print(f"========== 推理策略: {inference_strategy.upper()} ==========")
    
    # --- Step 1: 準備測試數據 ---
    print("\n[1/4] 準備測試數據...")
    cfg_data = config['data']
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    
    df_raw_train = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw_train)

    test_data_cfg = cfg_data['test_data']
    df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], test_data_cfg['filename']))
    df_raw_test = df_raw_test.iloc[:test_data_cfg['point']]
    df_raw_test.dropna(inplace=True)
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)
    
    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    # --- Step 2: 載入模型 ---
    print(f"\n[2/4] 加載 {config['model']['name']} 模型...")
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    
    model = get_model(config)
    
    model_path = os.path.join('./saved_models/', f'{prefix}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已從 {model_path} 加載至 {device}。")

    # --- Step 3: 執行預測 ---
    print("\n[3/4] 執行長期滾動預測...")
    initial_history_np = df_z_test.iloc[0:W][en_mv_and_sv].values
    future_mvs_np = df_z_test.iloc[W:][de_mv].values
    true_targets_np = df_z_test.iloc[W:][y_sv].values

    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    future_de_inputs = torch.tensor(future_mvs_np, dtype=torch.float32).unsqueeze(0)

    if inference_strategy == 'block_replacement':
        predictions_z = predict_block_replacement(model, initial_en_input, future_de_inputs, device, config)
    elif inference_strategy == 'sliding_window':
        predictions_z = predict_sliding_window(model, initial_en_input, future_de_inputs, device, config['data']['num_output'])
    else:
        raise ValueError(f"未知的推理策略: '{inference_strategy}'。")

    num_actual_preds = predictions_z.shape[1]
    true_targets_np = true_targets_np[:num_actual_preds, :]

    # --- Step 4: 儲存 & 繪圖 ---
    print("\n[4/4] 處理、保存並可視化結果...")
    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    predictions_cov = predictions_np * y_std + y_mean
    true_targets_cov = true_targets_np * y_std + y_mean

    results_dir = os.path.join('./results/', prefix)
    os.makedirs(results_dir, exist_ok=True)
    
    # (A) 保存真實值 + 預測值
    df_true = pd.DataFrame(true_targets_cov, columns=y_sv)
    df_pred = pd.DataFrame(predictions_cov, columns=[f"{col}_pred" for col in y_sv])
    df_results = pd.concat([df_true, df_pred], axis=1)
    results_csv_path = os.path.join(results_dir, 'prediction_results.csv')
    df_results.to_csv(results_csv_path, index=False)
    print(f"數值預測結果已保存至: {results_csv_path}")

    # (B) 計算 & 保存 MAE
    mae_results = {}
    for i, name in enumerate(y_sv):
        mae_results[name] = np.mean(np.abs(true_targets_cov[:, i] - predictions_cov[:, i]))
    df_mae = pd.DataFrame(list(mae_results.items()), columns=["Variable", "MAE"])
    mae_csv_path = os.path.join(results_dir, 'mae_results.csv')
    df_mae.to_csv(mae_csv_path, index=False)
    print(f"MAE 結果已保存至: {mae_csv_path}")

    # (C) 繪圖
    for i, name in enumerate(y_sv):
        plt.figure(figsize=(20, 6))
        plt.plot(true_targets_cov[:, i], label='真實值 (True Value)', color='blue', linewidth=2)
        plt.plot(predictions_cov[:, i], label='預測值 (Predicted Value)', color='red', linestyle='--', linewidth=2)
        plt.title(f'長期滾動預測結果: {name} (策略: {inference_strategy.upper()})', fontsize=16)
        plt.xlabel('時間步 (Time Step)', fontsize=12)
        plt.ylabel('值 (Value)', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        save_path = os.path.join(results_dir, f'prediction_{name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"所有結果圖已保存至: {results_dir}")
    print(f"========== 預測完成: {prefix} ==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="執行基於設定檔的長期滾動預測")
    parser.add_argument('--config', type=str, required=True, help='指向實驗的 YAML 設定檔路徑')
    args = parser.parse_args()
    
    main(args.config)
