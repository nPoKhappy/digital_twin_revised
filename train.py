# train.py (完整且正確的版本)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time

# 導入我們自己寫的模組
from src import data_utils, engine
from src.dataset import MultiStepS2SDataset
from src.model import Seq2Seq
from src.utils import generate_results

# ==============================================================================
# --- 實驗配置區 ---
# ==============================================================================
CONFIG = {
    'exp_name': 'Rolling_Training_Direct_Prediction',

    'data': {
        'path': './data/',
        'filename': 'Train_input.csv',
        'variables_num': 30,
        'point': 15000,
        'test_data_split': 0.1,
        'valid_data_split': 0.05,
    },
    
    'window': {
        'train_window_mins': 180,       # 輸入長度 (180/10 = 18 steps)
        'sampling_interval_min': 10,
        'prediction_length': 30,        # 直接指定預測 30 個時間點
    },

    'model': {
        'embedding_dim': 32,
        'hidden_dim': 64,
        'n_layers': 3,
    },

    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 4096,
        'epochs': 1,
        'learning_rate': 0.001,
        'patience': 100,
    },

    'output': {
        'results_dir': './results/',
        'models_dir': './saved_models/',
    }
}
# ==============================================================================
# --- 主程式碼 ---
# ==============================================================================

def main():
    prefix = CONFIG['exp_name']
    print(f"========== 開始實驗: {prefix} ==========")
    start_time = time.time()

    # 1. 數據準備
    print("\n步驟 1/4: 準備數據...")
    cfg_data = CONFIG['data']
    cfg_win = CONFIG['window']
    
    # --- 關鍵修正處 ---
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']  # 使用 H_out，不再使用 H
    CONFIG['en_window_steps'] = W
    CONFIG['de_window_steps'] = H_out     # 將 H_out 存入 config 以便 utils.py 使用
    # --- 修正結束 ---
    
    df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_raw = df_raw.iloc[:cfg_data['point']]
    df_raw.dropna(inplace=True)

    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
    df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)

    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    num_en_input, num_de_input, num_output = len(en_mv_and_sv), len(de_mv), len(y_sv)
    
    data_len = len(df_z)
    split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
    split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
    
    train_df, val_df, test_df = df_z.iloc[:split_point2], df_z.iloc[split_point2:split_point1], df_z.iloc[split_point1:]

    # --- 關鍵修正處 ---
    # 使用新的 Dataset 格式 (W, H_out)，並且不再傳入 horizons
    train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    test_ds = MultiStepS2SDataset(test_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    # --- 修正結束 ---
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['training']['batch_size'], shuffle=False)
    print("數據準備完成。")
    
    # 2. 初始化模型、優化器、損失函數
    print("\n步驟 2/4: 初始化模型...")
    cfg_model = CONFIG['model']
    cfg_training = CONFIG['training']
    model = Seq2Seq(num_en_input, num_de_input, num_output, cfg_model['embedding_dim'], cfg_model['hidden_dim'], cfg_model['n_layers']).to(cfg_training['device'])
    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])
    criterion = nn.L1Loss()
    
    # 3. 選擇訓練策略
    training_step_fn = engine.step_wise_rolling_training_step
    print("訓練模式: Step-wise Rolling Training")

    # 4. 執行訓練
    print("\n步驟 3/4: 開始訓練...")
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_path = os.path.join(CONFIG['output']['models_dir'], f'{prefix}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(cfg_training['epochs']):
        train_loss = engine.train_one_epoch(model, train_loader, optimizer, criterion, cfg_training['device'], training_step_fn)
        val_loss = engine.evaluate(model, val_loader, criterion, cfg_training['device'], training_step_fn)
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print('  -> 驗證損失降低，模型已儲存。')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg_training['patience']:
            print("Early stopping!")
            break
            
    # 5. 最終評估
    print("\n步驟 4/4: 使用最佳模型進行最終評估...")
    best_model = Seq2Seq(num_en_input, num_de_input, num_output, cfg_model['embedding_dim'], cfg_model['hidden_dim'], cfg_model['n_layers']).to(cfg_training['device'])
    best_model.load_state_dict(torch.load(model_save_path))
    print(f"已從 {model_save_path} 加載最佳模型。")

    generate_results(best_model, test_loader, cfg_training['device'], CONFIG, mean_all, std_all, y_sv, de_mv, prefix, 'test')
    generate_results(best_model, val_loader, cfg_training['device'], CONFIG, mean_all, std_all, y_sv, de_mv, prefix, 'validation')
    
    end_time = time.time()
    print(f"\n實驗 {prefix} 完成。總耗時: {end_time - start_time:.2f} 秒。")


if __name__ == '__main__':
    main()