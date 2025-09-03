# train.py (已修改為專注於訓練和保存模型)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time

from src import data_utils, engine
from src.dataset import MultiStepS2SDataset
from src.model import Seq2Seq

# ==============================================================================
# --- 實驗配置區 ---
# ==============================================================================
CONFIG = {
    'exp_name': 'Rolling_Training_Final',

    'data': {
        'path': './data/',
        'filename': 'Train_input.csv',
        'variables_num': 30,
        'point': 15000,
        'test_data_split': 0.1,
        'valid_data_split': 0.05,
    },
    
    'window': {
        'train_window_mins': 180,       # 輸入長度 (18 steps)
        'sampling_interval_min': 10,
        'prediction_length': 18,        # 訓練時的滾動長度 (18 steps)
    },

    'model': {
        'embedding_dim': 32,
        'hidden_dim': 64,
        'n_layers': 3,
    },

    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2048, # 根據顯存可適當調整
        'epochs': 10000,
        'learning_rate': 0.001,
        'patience': 100,
    },

    'output': {
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
    print("\n步驟 1/3: 準備訓練與驗證數據...")
    cfg_data = CONFIG['data']
    cfg_win = CONFIG['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']
    
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
    
    train_df, val_df = df_z.iloc[:split_point2], df_z.iloc[split_point2:split_point1]

    train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['training']['batch_size'], shuffle=False)
    print("數據準備完成。")
    
    # 2. 初始化模型、優化器、損失函數
    print("\n步驟 2/3: 初始化模型...")
    cfg_model = CONFIG['model']
    cfg_training = CONFIG['training']
    model = Seq2Seq(num_en_input, num_de_input, num_output, cfg_model['embedding_dim'], cfg_model['hidden_dim'], cfg_model['n_layers']).to(cfg_training['device'])
    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])
    criterion = nn.L1Loss()
    
    training_step_fn = engine.step_wise_rolling_training_step
    print(f"訓練模式: Step-wise Rolling Training (滾動長度: {H_out} 步)")

    # 3. 執行訓練
    print("\n步驟 3/3: 開始訓練...")
    best_val_loss = float('inf')
    patience_counter = 0
    model_save_path = os.path.join(CONFIG['output']['models_dir'], f'{prefix}.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(cfg_training['epochs']):
        train_loss = engine.train_one_epoch(model, train_loader, optimizer, criterion, cfg_training['device'], training_step_fn)
        val_loss = engine.evaluate(model, val_loader, criterion, cfg_training['device'], training_step_fn)
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'  -> 驗證損失降低，最佳模型已儲存至: {model_save_path}')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg_training['patience']:
            print(f"Early stopping at epoch {epoch+1}!")
            break
            
    end_time = time.time()
    print(f"\n訓練完成。總耗時: {(end_time - start_time)/60:.2f} 分鐘。")
    print(f"最佳模型已保存在 {model_save_path}")
    print("請運行 predict.py 腳本來進行長期滾動預測和評估。")

if __name__ == '__main__':
    main()