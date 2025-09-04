import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# 确保你的项目结构中有这些模块
from src import data_utils
from src.models import get_model

# ==============================================================================
# --- 核心預測函數 ---
# ==============================================================================

def predict_sliding_window(model, initial_en_input, future_de_inputs, device, num_output_features):
    """
    策略一：執行長期的、逐時間步的【滑動窗口】預測。
    
    Args:
        model (torch.nn.Module): 已訓練的模型。
        initial_en_input (torch.Tensor): 初始的編碼器輸入歷史，維度 (1, W, num_en_features)。
        future_de_inputs (torch.Tensor): 未來的解碼器輸入 (MVs)，維度 (1, num_pred_steps, num_de_features)。
        device (str): 'cuda' or 'cpu'。
        num_output_features (int): 輸出的特徵數量。
        
    Returns:
        torch.Tensor: 預測結果，維度 (1, num_pred_steps, num_output_features)。
    """
    model.eval()
    
    num_pred_steps = future_de_inputs.shape[1]

    # 初始化一個空的張量來存放所有預測結果
    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    # 複製初始輸入，避免修改原始數據
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for t in tqdm(range(num_pred_steps), desc="[策略: 滑動窗口] 預測中"):
            # 準備當前單一步的 Decoder 輸入
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            
            # 假設模型 forward 方法能處理 seq2seq 預測，即使只預測一步
            # model(encoder_input, decoder_input) -> decoder_output
            single_step_prediction = model(current_en_input, single_step_de_input)

            # 儲存這一時間步的預測結果
            predictions[:, t, :] = single_step_prediction

            # 【滾動核心】更新 Encoder 的輸入歷史
            # 1. 移除最舊的時間步
            next_en_input_history = current_en_input[:, 1:, :]
            # 2. 將預測結果 (QVs) 和對應的輸入 (MVs) 拼接成新的特徵
            new_step_features = torch.cat([single_step_prediction, single_step_de_input], dim=2)
            # 3. 將新的特徵添加到歷史的末尾
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions


def predict_block_replacement(model, initial_en_input, future_de_inputs, device, config):
    """
    策略二：執行【塊替換】預測，以匹配 Keras 的訓練策略。
    
    Args:
        model (torch.nn.Module): 已訓練的模型。
        initial_en_input (torch.Tensor): 初始的編碼器輸入歷史，維度 (1, W, num_en_features)。
        future_de_inputs (torch.Tensor): 未來的解碼器輸入 (MVs)，維度 (1, num_pred_steps, num_de_features)。
        device (str): 'cuda' or 'cpu'。
        config (dict): 包含窗口大小等信息的 YAML 配置字典。
        
    Returns:
        torch.Tensor: 預測結果，維度 (1, num_pred_steps, num_output_features)。
    """
    model.eval()
    
    # 從配置中獲取窗口/塊的大小 (H)
    H = config['window']['train_window_mins'] // config['window']['sampling_interval_min']
    
    num_pred_steps = future_de_inputs.shape[1]
    
    # 確保預測總步數是窗口大小的整數倍
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
            # 1. 提取當前未來窗口所需的 de_input (MVs)
            start_idx = i * H
            end_idx = (i + 1) * H
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)
            
            # 2. 模型一次性預測一個完整的塊
            prediction_block = model(current_en_input, de_input_block)
            
            # 3. 收集這個窗口的預測結果
            predictions_all_windows.append(prediction_block)
            
            # 4. 【塊替換核心】將預測塊和對應的輸入塊拼接，形成下一個 Encoder 輸入
            current_en_input = torch.cat([prediction_block, de_input_block], dim=2)

    # 將所有預測塊拼接成一個連續的時間序列
    return torch.cat(predictions_all_windows, dim=1)

# ==============================================================================
# --- 主程式碼 ---
# ==============================================================================

def main(config_path):
    # --- 步驟 0: 加載設定檔並設定環境 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    # 從配置中讀取推理策略，如果未指定，則默認為 'sliding_window'
    inference_strategy = config['training'].get('inference_strategy', 'sliding_window')
    
    print(f"========== 開始預測: {prefix} (模型: {config['model']['name']}) ==========")
    print(f"========== 推理策略: {inference_strategy.upper()} ==========")
    
    # --- 步驟 1: 準備測試數據 ---
    print("\n[1/4] 準備測試數據...")
    cfg_data = config['data']
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    
    # 加載訓練數據以計算 Z-score 統計量
    df_raw_train = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw_train)

    # 加載並處理測試數據
    test_data_cfg = cfg_data['test_data']
    df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], test_data_cfg['filename']))
    df_raw_test = df_raw_test.iloc[:test_data_cfg['point']]
    df_raw_test.dropna(inplace=True)
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)
    
    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    
    # 將必要的變數數量寫回 config，方便後續使用
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    # --- 步驟 2: 加載模型 ---
    print(f"\n[2/4] 加載 {config['model']['name']} 模型...")
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    
    model = get_model(config)
    
    model_path = os.path.join('./saved_models/', f'{prefix}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型已從 {model_path} 加載至 {device}。")

    # --- 步驟 3: 執行預測 ---
    print("\n[3/4] 執行長期滾動預測...")
    # 準備模型的初始輸入和未來的已知輸入
    initial_history_np = df_z_test.iloc[0:W][en_mv_and_sv].values
    future_mvs_np = df_z_test.iloc[W:][de_mv].values
    true_targets_np = df_z_test.iloc[W:][y_sv].values

    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    future_de_inputs = torch.tensor(future_mvs_np, dtype=torch.float32).unsqueeze(0)

    # --- 策略切換的核心 ---
    if inference_strategy == 'block_replacement':
        predictions_z = predict_block_replacement(model, initial_en_input, future_de_inputs, device, config)
    elif inference_strategy == 'sliding_window':
        predictions_z = predict_sliding_window(model, initial_en_input, future_de_inputs, device, config['data']['num_output'])
    else:
        raise ValueError(f"未知的推理策略: '{inference_strategy}'。請選擇 'block_replacement' 或 'sliding_window'。")
    print(f"使用的推理策略: {inference_strategy}")
    # 調整真實目標長度，以匹配可能因窗口不對齊而被截斷的預測
    num_actual_preds = predictions_z.shape[1]
    true_targets_np = true_targets_np[:num_actual_preds, :]

    # --- 步驟 4: 處理、保存並可視化結果 ---
    print("\n[4/4] 處理、保存並可視化結果...")
    # 將 Z-score 標準化的預測結果反轉回原始尺度
    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    predictions_cov = predictions_np * y_std + y_mean
    true_targets_cov = true_targets_np * y_std + y_mean

    # 建立結果目錄
    results_dir = os.path.join('./results/', prefix)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存數值結果到 CSV
    df_true = pd.DataFrame(true_targets_cov, columns=y_sv)
    df_pred = pd.DataFrame(predictions_cov, columns=[f"{col}_pred" for col in y_sv])
    df_results = pd.concat([df_true, df_pred], axis=1)
    results_csv_path = os.path.join(results_dir, 'prediction_results.csv')
    df_results.to_csv(results_csv_path, index=False)
    print(f"數值預測結果已保存至: {results_csv_path}")

    # 繪製並保存每個目標變量的結果圖
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