# predict.py (已修正 AttributeError 的版本)

import torch
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from src import data_utils
from src.model import Seq2Seq

# ==============================================================================
# --- 預測配置區 ---
# ==============================================================================
CONFIG = {
    'exp_name': 'Rolling_Training_Final',
    'data': {
        'path': './data/',
        'filename': 'Train_input.csv',
        'variables_num': 30,
    },
    'test_data': {
        'filename': 'rolling_data.csv',
        'point': 1500,
    },
    'window': {
        'train_window_mins': 180,
        'sampling_interval_min': 10,
    },
    'model': {
        'embedding_dim': 32,
        'hidden_dim': 64,
        'n_layers': 3,
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output': {
        'models_dir': './saved_models/',
        'results_dir': './results/'
    }
}
# ==============================================================================
# --- 核心預測函數 ---
# ==============================================================================

def predict_long_rolling(model, initial_en_input, future_de_inputs, device):
    """
    執行長期的、逐時間步的滾動預測。
    """
    model.eval()
    
    W = initial_en_input.shape[1]
    num_pred_steps = future_de_inputs.shape[1]
    
    # --- !!! 關鍵修正處 !!! ---
    # 直接從解碼器的最後一層獲取輸出維度，更為穩健
    num_output_features = model.decoder.fc_out.out_features
    # --- 修正結束 ---

    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for t in tqdm(range(num_pred_steps), desc="長期滾動預測"):
            _, hiddens = model.encoder(current_en_input)
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model.decoder(single_step_de_input, hiddens)
            predictions[:, t, :] = single_step_prediction

            next_en_input_history = current_en_input[:, 1:, :]
            new_step_features = torch.cat([single_step_prediction, single_step_de_input], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions

# ==============================================================================
# --- 主程式碼 ---
# (以下部分無須修改)
# ==============================================================================

def main():
    prefix = CONFIG['exp_name']
    print(f"========== 開始預測: {prefix} ==========")
    
    print("\n步驟 1/4: 準備測試數據...")
    cfg_data = CONFIG['data']
    cfg_win = CONFIG['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    
    df_raw_train = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw_train)

    df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], CONFIG['test_data']['filename']))
    df_raw_test = df_raw_test.iloc[:CONFIG['test_data']['point']]
    df_raw_test.dropna(inplace=True)
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)
    
    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    num_en_input, num_de_input, num_output = len(en_mv_and_sv), len(de_mv), len(y_sv)
    
    print("\n步驟 2/4: 加載模型...")
    cfg_model = CONFIG['model']
    model = Seq2Seq(num_en_input, num_de_input, num_output, cfg_model['embedding_dim'], cfg_model['hidden_dim'], cfg_model['n_layers'])
    model_path = os.path.join(CONFIG['output']['models_dir'], f'{prefix}.pth')
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
    model.to(CONFIG['device'])
    print(f"模型已從 {model_path} 加載。")

    print("\n步驟 3/4: 執行長期滾動預測...")
    initial_en_input_np = df_z_test.iloc[0:W][en_mv_and_sv].values
    future_de_inputs_np = df_z_test.iloc[W:][de_mv].values
    true_targets_np = df_z_test.iloc[W:][y_sv].values

    initial_en_input = torch.tensor(initial_en_input_np, dtype=torch.float32).unsqueeze(0)
    future_de_inputs = torch.tensor(future_de_inputs_np, dtype=torch.float32).unsqueeze(0)

    predictions_z = predict_long_rolling(model, initial_en_input, future_de_inputs, CONFIG['device'])

    print("\n步驟 4/4: 處理並可視化結果...")
    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    predictions_cov = predictions_np * y_std + y_mean
    true_targets_cov = true_targets_np * y_std + y_mean

    results_dir = os.path.join(CONFIG['output']['results_dir'], prefix)
    os.makedirs(results_dir, exist_ok=True)
    
    for i, name in enumerate(y_sv):
        plt.figure(figsize=(20, 6))
        plt.plot(true_targets_cov[:, i], label='真實值 (True Value)', color='blue')
        plt.plot(predictions_cov[:, i], label='預測值 (Predicted Value)', color='red', linestyle='--')
        plt.title(f'長期滾動預測結果: {name}')
        plt.xlabel('時間步 (Time Step)')
        plt.ylabel('值 (Value)')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(results_dir, f'long_rolling_prediction_{name}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"結果圖已保存至: {save_path}")

if __name__ == '__main__':
    main()