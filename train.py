import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import yaml
import argparse

from src import data_utils, engine
from src.dataset import MultiStepS2SDataset
from src.models import get_model # <--- 從模型工廠導入

def main(config_path):
    # 1. 載入設定檔
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    print(f"========== 開始實驗: {prefix} (模型: {config['model']['name']}) ==========")
    start_time = time.time()

    # 2. 數據準備
    print("\n步驟 1/3: 準備訓練與驗證數據...")
    cfg_data = config['data']
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']
    
    df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_raw = df_raw.iloc[:cfg_data['point']]
    df_raw.dropna(inplace=True)

    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
    df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)

    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    
    # 將維度資訊存回 config，以便模型工廠使用
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    # ... (數據分割和 DataLoader 的部分與之前相同) ...
    data_len = len(df_z)
    split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
    split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
    train_df, val_df = df_z.iloc[:split_point2], df_z.iloc[split_point2:split_point1]
    train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    print("數據準備完成。")

    # 3. 初始化模型 (使用模型工廠)
    print(f"\n步驟 2/3: 初始化 {config['model']['name']} 模型...")
    cfg_training = config['training']
    device = 'cuda' if torch.cuda.is_available() and cfg_training['device'] == 'cuda' else 'cpu'
    
    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])
    criterion = nn.L1Loss()
    
    # --- !!! 關鍵修改：根據設定檔自動選擇訓練模式 !!! ---
    if 'loss_weighting' in cfg_training and cfg_training['loss_weighting']['weights']:
        training_step_fn = engine.step_wise_rolling_at_loss_step
        print(f"訓練模式: Step-wise Rolling with AT Loss (權重: {cfg_training['loss_weighting']['weights']})")
    else:
        training_step_fn = engine.step_wise_rolling_training_step
        print("訓練模式: Standard Step-wise Rolling")
    print(f"訓練模式: Step-wise Rolling Training (滾動長度: {H_out} 步)")
    
    print("\n步驟 3/3: 開始訓練...")
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs('./saved_models/', exist_ok=True)
    model_save_path = os.path.join('./saved_models/', f'{prefix}.pth')

    for epoch in range(cfg_training['epochs']):
        train_loss = engine.train_one_epoch(model, train_loader, optimizer, criterion, device, training_step_fn, config)
        val_loss = engine.evaluate(model, val_loader, criterion, device, training_step_fn, config)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    main(args.config)