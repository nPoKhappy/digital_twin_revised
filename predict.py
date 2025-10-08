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
from src import data_utils
from src.models import get_model

# ==============================================================================
# --- 評估指標計算函數 ---
# ==============================================================================

def calculate_metrics(y_true, y_pred):
    """
    計算多種評估指標
    
    Args:
        y_true: 真實值 (numpy array)
        y_pred: 預測值 (numpy array)
    
    Returns:
        dict: 包含 MAE, RMSE, R², MAPE 的字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE: Mean Absolute Percentage Error
    # 避免除以零，當真實值為 0 時使用一個小的 epsilon
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

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
    
    # 安全地載入訓練數據，處理可選的 DateTime 索引
    try:
        df_raw_train = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
        print("成功載入訓練數據（帶 DateTime 索引）")
    except (KeyError, ValueError) as e:
        print(f"注意：數據中沒有 DateTime 列，使用預設索引載入: {e}")
        # 如果沒有 DateTime 列，直接讀取 CSV
        df_raw_train = pd.read_csv(os.path.join(cfg_data['path'], cfg_data['filename']))
        print("成功載入訓練數據（使用預設索引）")
    
    # 清理訓練數據中的缺失值，這是計算統計數據前的關鍵步驟
    df_raw_train.dropna(inplace=True)
    
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw_train)

    test_data_cfg = cfg_data['test_data']
    # 安全地載入測試數據
    try:
        df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], test_data_cfg['filename']))
        print("成功載入測試數據（帶 DateTime 索引）")
    except (KeyError, ValueError) as e:
        print(f"注意：數據中沒有 DateTime 列，使用預設索引載入: {e}")
        # 如果沒有 DateTime 列，直接讀取 CSV
        df_raw_test = pd.read_csv(os.path.join(cfg_data['path'], test_data_cfg['filename']))
        print("成功載入測試數據（使用預設索引）")
    
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

    # (B) 計算並保存所有評估指標 (MAE, RMSE, R², MAPE)
    print("\n計算評估指標...")
    metrics_results = []
    
    for i, name in enumerate(y_sv):
        y_true_col = true_targets_cov[:, i]
        y_pred_col = predictions_cov[:, i]
        
        metrics = calculate_metrics(y_true_col, y_pred_col)
        metrics['Variable'] = name
        metrics_results.append(metrics)
        
        print(f"  {name}:")
        print(f"    MAE:  {metrics['MAE']:.6f}")
        print(f"    RMSE: {metrics['RMSE']:.6f}")
        print(f"    R²:   {metrics['R2']:.6f}")
        print(f"    MAPE: {metrics['MAPE']:.2f}%")
    
    # 創建指標彙總表
    df_metrics = pd.DataFrame(metrics_results)
    df_metrics = df_metrics[['Variable', 'MAE', 'RMSE', 'R2', 'MAPE']]  # 調整列順序
    
    # 保存指標到 CSV
    metrics_csv_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\n所有評估指標已保存至: {metrics_csv_path}")
    
    # 計算平均指標
    avg_metrics = {
        'Variable': 'Average',
        'MAE': df_metrics['MAE'].mean(),
        'RMSE': df_metrics['RMSE'].mean(),
        'R2': df_metrics['R2'].mean(),
        'MAPE': df_metrics['MAPE'].mean()
    }
    print(f"\n平均指標:")
    print(f"  平均 MAE:  {avg_metrics['MAE']:.6f}")
    print(f"  平均 RMSE: {avg_metrics['RMSE']:.6f}")
    print(f"  平均 R²:   {avg_metrics['R2']:.6f}")
    print(f"  平均 MAPE: {avg_metrics['MAPE']:.2f}%")

    # (C) 繪圖 - 添加指標信息
    print("\n生成預測對比圖...")
    for i, name in enumerate(y_sv):
        # 獲取該變量的指標
        var_metrics = metrics_results[i]
        
        plt.figure(figsize=(20, 6))
        plt.plot(true_targets_cov[:, i], label='True Value', color='blue', linewidth=2)
        plt.plot(predictions_cov[:, i], label='Predicted Value', color='red', linestyle='--', linewidth=2)
        
        # 在標題中添加指標信息
        title = f'{name} (Strategy: {inference_strategy.upper()})\n'
        title += f'MAE={var_metrics["MAE"]:.4f}, RMSE={var_metrics["RMSE"]:.4f}, R²={var_metrics["R2"]:.4f}, MAPE={var_metrics["MAPE"]:.2f}%'
        plt.title(title, fontsize=14)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        save_path = os.path.join(results_dir, f'prediction_{name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"所有結果圖已保存至: {results_dir}")
    print(f"========== 預測完成: {prefix} ==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="執行基於設定檔的長期滾動預測")
    parser.add_argument('--config', type=str, required=True, help='指向實驗的 YAML 設定檔路徑')
    args = parser.parse_args()
    
    main(args.config)
