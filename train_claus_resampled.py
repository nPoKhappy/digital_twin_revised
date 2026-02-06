# ==============================================================================
# 訓練腳本 - Claus 製程專用版本
# 支持多段獨立數據訓練（不同操作條件的模擬）
# ==============================================================================

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os
import time
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# 導入自定義模組
# 導入自定義模組
from src import utils as data_utils  # 為了最小化代碼改動，將 utils 別名為 data_utils
from src import engine
from src import variable_selection  # 新的變量選擇模組
from src.dataset import MultiStepS2SDataset
from src.models import get_model

def main(config_path):
    """
    主訓練函數 - 支持多段獨立數據
    """
    # ===========================================================================
    # 步驟 1: 載入實驗配置
    # ===========================================================================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']

    cfg_training = config['training'] # Initialize early for use in dataset logic
    cfg_win = config['window']
    
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']

    print(f"========== 開始實驗: {prefix} (模型: {config['model']['name']}) ==========")
    start_time = time.time()

    # ===========================================================================
    # 步驟 2: 數據準備與預處理
    # ===========================================================================
    print("\n步驟 1/3: 準備訓練與驗證數據...")
    cfg_data = config['data']
    
    # 變量選擇：分離輸入變量和目標變量
    # 注意：現在從 variable_selection 模組調用
    de_mv, y_sv, _, en_mv_and_sv = variable_selection.variable_selection(cfg_data['variables_num'])
    
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    print(f"  Encoder 輸入: {len(en_mv_and_sv)} 個變數")
    print(f"  Decoder 輸入 (MV): {len(de_mv)} 個變數")
    print(f"  預測目標: {len(y_sv)} 個變數")
    
    # Define Prediction Length (Used in Dataset creation and Strategy Selection)
    # H_out and W are already defined above
    
    # ===========================================================================
    # 步驟 2.1: 載入數據 (支持多段獨立數據)
    # ===========================================================================
    
    interval = config['window']['sampling_interval_min']
    use_median = config['window'].get('use_median_downsampling', True)

    # 1. 載入原始數據 (Load Raw Data) & 2. 立即降採樣 (Downsample immediately)
    all_dfs_processed = []

    if 'training_files' in cfg_data and cfg_data['training_files']:
        # 多段數據模式
        print(f"\n使用多段數據模式: {len(cfg_data['training_files'])} 個獨立數據段")
        
        segment_points = cfg_data.get('segment_points', None)
        default_point = cfg_data.get('point', None)
        
        for idx, fname in enumerate(cfg_data['training_files']):
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                try:
                    df_seg = pd.read_csv(fpath)
                except:
                     df_seg = pd.read_csv(fpath, engine='python') # Fallback

                # 截取長度
                if segment_points and idx < len(segment_points):
                    seg_point = segment_points[idx]
                elif default_point:
                    seg_point = default_point
                else:
                    seg_point = len(df_seg)
                
                df_seg = df_seg.iloc[:seg_point]
                # df_seg.dropna(inplace=True) # 不要太早 drop, 可能會影響 rolling 計算

                # [Step 2: Downsample]
                if interval > 1:
                    if use_median:
                        # 用 10min 窗口的中位數代表這 10min
                        # Fix: numeric_only=True to avoid error on DateTime column
                        df_seg = df_seg.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                        df_seg = df_seg.iloc[interval-1::interval].reset_index(drop=True)
                    else:
                        df_seg = df_seg.iloc[::interval].reset_index(drop=True)
                
                df_seg.dropna(inplace=True) # Downsample 後再 drop NaN
                all_dfs_processed.append(df_seg)
                print(f"  ✓ 載入 {fname}: {len(df_seg)} 行 (已降採樣)")
            else:
                print(f"  ✗ 找不到 {fpath}，跳過")

    else:
        # 單一檔案模式
        print("\n使用單一檔案模式")
        fpath = os.path.join(cfg_data['path'], cfg_data['filename'])
        try:
             df_raw = pd.read_csv(fpath)
        except:
             df_raw = pd.read_csv(fpath, engine='python')

        if 'point' in cfg_data:
            df_raw = df_raw.iloc[:cfg_data['point']]
        
        # [Step 2: Downsample]
        if interval > 1:
            if use_median:
                # Fix: numeric_only=True
                df_raw = df_raw.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                df_raw = df_raw.iloc[interval-1::interval].reset_index(drop=True)
            else:
                df_raw = df_raw.iloc[::interval].reset_index(drop=True)
        
        df_raw.dropna(inplace=True)
        all_dfs_processed = [df_raw]
        print(f"  載入數據: {len(df_raw)} 行 (已降採樣)")

    if not all_dfs_processed:
        raise ValueError("沒有數據！")

    # 3. Log Transform
    target_cols = ['B35_H2S', 'B35_SO2']
    print(f"\n[Data Processing] Applying Log Transform to {target_cols}")
    all_dfs_log = [data_utils.apply_log_transform(df, target_cols) for df in all_dfs_processed]

    # 4. Calculate Stats (Mean/Std) on ALL Combined Data
    df_all_log = pd.concat(all_dfs_log, ignore_index=True)
    
    # 使用 Z-score Scaling
    print(f"\n[Stats] Calculating Z-score Stats (Mean/Std) on {len(df_all_log)} samples...")
    stats_mean, stats_std = data_utils.calculate_zscore_stats(df_all_log)
    
    # 5. Apply Scaling (Z-score)
    print(f"[Data Processing] Applying Z-score Scaling...")
    all_dfs_z = [data_utils.apply_zscore(df, stats_mean, stats_std) for df in all_dfs_log]

    # 保存 Z-score 參數供預測使用
    zscore_dir = f'./results/{prefix}/'
    os.makedirs(zscore_dir, exist_ok=True)
    stats_mean.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
    stats_std.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
    print(f"  Scaling 參數 (Mean/Std) 已保存至: {zscore_dir}")
    
    # ===========================================================================
    # 步驟 3: 數據分割與 DataLoader 創建
    # ===========================================================================
    print("\n創建數據集...")
    
    train_datasets = []
    val_datasets = []
    
    # Determine total dataset prediction length
    # If using AT Rolling (multiple windows), we need total length = H_block * num_windows
    # If using Standard Rolling, we just need H_out
    
    if 'loss_weighting' in cfg_training and len(cfg_training['loss_weighting'].get('weights', [])) > 1:
         num_weights = len(cfg_training['loss_weighting']['weights'])
         
         # Assume Block-wise Weighting (Default)
         # Total Length = Block_Size * Num_Blocks
         dataset_H = H_out * num_weights
         print(f"  偵測到 Block-wise Weighting: {num_weights} 個窗口 x {H_out} 步 = {dataset_H} 步")
    else:
         dataset_H = H_out

    for i, df_z in enumerate(all_dfs_z):
        data_len = len(df_z)
        
        # 計算數據分割點
        split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
        split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
        
        # 按時間順序分割數據
        train_df = df_z.iloc[:split_point2]
        val_df = df_z.iloc[split_point2:split_point1]
        
        # 創建 Dataset 對象
        # Passing dataset_H (Total Length) instead of H_out (Single Block)
        if len(train_df) > W + dataset_H:
            train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            train_datasets.append(train_ds)
            print(f"  段 {i+1} 訓練集: {len(train_ds)} 個樣本")
        
        if len(val_df) > W + dataset_H:
            val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            val_datasets.append(val_ds)
            print(f"  段 {i+1} 驗證集: {len(val_ds)} 個樣本")
    
    # Check for empty datasets
    if not train_datasets:
        raise ValueError(f"訓練集為空！可能原因：數據長度不足以構建窗口 (W={W}) + 預測長度 (Dataset_H={dataset_H}) = {W+dataset_H} 步。\n"
                         f"請檢查數據文件長度或縮短 prediction_length / weights。")
    
    if not val_datasets:
         print(f"警告：驗證集為空！跳過 Validation DataLoader 創建。")
         # Create a dummy dataset or handle logic later
         # But for now, let's just make it equal to train_ds to avoid crash, or better, handle empty val_loader
         # Using a small subset of train_ds as fake validation to prevent crash
         print("  -> 使用部分訓練集作為替補驗證集以防止崩潰...")
         val_datasets = [train_datasets[0]] # Hack to survive

    # 使用 ConcatDataset 合併（每段內部連續，段與段之間獨立）
    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    print(f"\n總計: 訓練樣本 {len(train_ds)}, 驗證樣本 {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    print("數據準備完成。")

    # ===========================================================================
    # 步驟 4: 模型初始化與訓練設置
    # ===========================================================================
    print(f"\n步驟 2/3: 初始化 {config['model']['name']} 模型...")
    cfg_training = config['training']
    
    device = 'cuda' if torch.cuda.is_available() and cfg_training['device'] == 'cuda' else 'cpu'
    print(f"  運行設備: {device}")
    
    model = get_model(config).to(device)
    
    # [Added] Calculate Parameter Count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型總參數量: {total_params:,}")
    print("\n[Model Architecture]")
    print(model) # Print model architecture for reference

    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])
    
    loss_type = cfg_training.get('loss_function', 'l1').lower()
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        print("  Loss Function: MSE (Mean Squared Error)")
    else:
        criterion = nn.L1Loss()
        print("  Loss Function: L1 (MAE)")
    
    # ===========================================================================
    # 步驟 5: 選擇訓練策略
    # ===========================================================================
    strategy_name = cfg_training.get('inference_strategy', 'block_replacement')
    
    if 'loss_weighting' in cfg_training and cfg_training['loss_weighting']['weights']:
        weights = cfg_training['loss_weighting']['weights']
        num_windows = len(weights)
        H_block = H_out
        
        training_step_fn = engine.step_wise_rolling_at_loss_step
        print(f"訓練模式: AT Rolling / Block Replacement (Block={H_block}, Weights={len(weights)})")
        print(f"權重: {weights}")
    else:
        # Fallback default
        training_step_fn = engine.step_wise_rolling_training_step
        print("訓練模式: Standard Step-wise Rolling")
            

    print(f"滾動預測長度: {H_out} 步")
    
    # ===========================================================================
    # 步驟 6: 訓練循環與早停機制
    # ===========================================================================
    print("\n步驟 3/3: 開始訓練...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    os.makedirs('./saved_models/', exist_ok=True)
    model_save_path = os.path.join('./saved_models/', f'{prefix}.pth')

    for epoch in range(cfg_training['epochs']):
        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, criterion, device, training_step_fn, config
        )
        
        val_loss = engine.evaluate(
            model, val_loader, criterion, device, training_step_fn, config
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'  -> 驗證損失降低，最佳模型已儲存至: {model_save_path}')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg_training['patience']:
            print(f"Early stopping at epoch {epoch+1}! (連續 {cfg_training['patience']} 個 epoch 無改善)")
            break
    
    # ===========================================================================
    # 步驟 7: 保存訓練歷史
    # ===========================================================================
    print("\n保存訓練歷史...")
    
    results_dir = f'./results/{prefix}/'
    os.makedirs(results_dir, exist_ok=True)
    
    loss_history_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_csv_path = os.path.join(results_dir, 'training_history.csv')
    loss_history_df.to_csv(loss_csv_path, index=False)
    print(f"  -> 訓練歷史已保存至: {loss_csv_path}")
    
    # 繪製訓練曲線
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_df['epoch'], loss_history_df['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(loss_history_df['epoch'], loss_history_df['val_loss'], label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)')
    plt.title(f'Training History - {prefix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(results_dir, 'training_curve.png')
    plt.savefig(loss_plot_path, dpi=150)
    print(f"  -> 訓練曲線已保存至: {loss_plot_path}")
    plt.close()
            
    end_time = time.time()
    print(f"\n訓練完成。總耗時: {(end_time - start_time)/60:.2f} 分鐘。")
    print(f"最佳驗證損失: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Claus 製程時間序列預測模型訓練腳本")
    parser.add_argument('--config', type=str, required=True, 
                       help='YAML 配置文件的路径')
    args = parser.parse_args()
    main(args.config)
