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
    print(f"========== 開始實驗: {prefix} (模型: {config['model']['name']}) ==========")
    start_time = time.time()

    # ===========================================================================
    # 步驟 2: 數據準備與預處理
    # ===========================================================================
    print("\n步驟 1/3: 準備訓練與驗證數據...")
    cfg_data = config['data']
    cfg_win = config['window']
    
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']
    
    # 變量選擇：分離輸入變量和目標變量
    # 注意：現在從 variable_selection 模組調用
    de_mv, y_sv, _, en_mv_and_sv = variable_selection.variable_selection(cfg_data['variables_num'])
    
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    print(f"  Encoder 輸入: {len(en_mv_and_sv)} 個變數")
    print(f"  Decoder 輸入 (MV): {len(de_mv)} 個變數")
    print(f"  預測目標: {len(y_sv)} 個變數")
    
    # ===========================================================================
    # 步驟 2.1: 載入數據 (支持多段獨立數據)
    # ===========================================================================
    
    if 'training_files' in cfg_data and cfg_data['training_files']:
        # 多段數據模式
        print(f"\n使用多段數據模式: {len(cfg_data['training_files'])} 個獨立數據段")
        all_dfs_raw = []
        
        # 取得每段的數據點數（可以是列表或單一數值）
        segment_points = cfg_data.get('segment_points', None)
        default_point = cfg_data.get('point', None)
        
        for idx, fname in enumerate(cfg_data['training_files']):
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                df_seg = pd.read_csv(fpath)
                
                # 決定這一段要取多少行
                if segment_points and idx < len(segment_points):
                    seg_point = segment_points[idx]
                elif default_point:
                    seg_point = default_point
                else:
                    seg_point = len(df_seg)  # 全部
                
                df_seg = df_seg.iloc[:seg_point]
                df_seg.dropna(inplace=True)
                all_dfs_raw.append(df_seg)
                print(f"  ✓ 載入 {fname}: {len(df_seg)} 行")
            else:
                print(f"  ✗ 找不到 {fpath}，跳過")
        
        if not all_dfs_raw:
            raise ValueError("沒有成功載入任何訓練數據！")
        
        # 計算所有數據的統一 Z-score 參數
        df_all_raw = pd.concat(all_dfs_raw, ignore_index=True)
        mean_all, std_all = data_utils.calculate_zscore_stats(df_all_raw)
        print(f"\n統一 Z-score 參數計算完成 (基於 {len(df_all_raw)} 行數據)")
        
        # 對每段數據分別標準化
        all_dfs_z = [data_utils.apply_zscore(df, mean_all, std_all) for df in all_dfs_raw]
        
    else:
        # 單一檔案模式
        print("\n使用單一檔案模式")
        try:
            df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
        except (KeyError, ValueError):
            df_raw = pd.read_csv(os.path.join(cfg_data['path'], cfg_data['filename']))
        
        if 'point' in cfg_data:
            df_raw = df_raw.iloc[:cfg_data['point']]
        df_raw.dropna(inplace=True)
        print(f"  載入數據: {len(df_raw)} 行")
        
        mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
        all_dfs_z = [data_utils.apply_zscore(df_raw, mean_all, std_all)]
    
    # [Added] Downsampling Logic based on sampling_interval_min
    interval = config['window']['sampling_interval_min']
    if interval > 1:
        print(f"\n[Data Processing] Downsampling data by interval: {interval}")
        print("  (Note: Mean/Std for Z-score were calculated using FULL 1-min data for better accuracy)")
        all_dfs_z = [df.iloc[::interval].reset_index(drop=True) for df in all_dfs_z]
        for idx, df in enumerate(all_dfs_z):
            print(f"  File {idx}: Reduced length = {len(df)}")

    # 保存 Z-score 參數供預測使用
    zscore_dir = f'./results/{prefix}/'
    os.makedirs(zscore_dir, exist_ok=True)
    mean_all.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
    std_all.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
    print(f"  Z-score 參數已保存至: {zscore_dir}")
    
    # ===========================================================================
    # 步驟 3: 數據分割與 DataLoader 創建
    # ===========================================================================
    print("\n創建數據集...")
    
    train_datasets = []
    val_datasets = []
    
    for i, df_z in enumerate(all_dfs_z):
        data_len = len(df_z)
        
        # 計算數據分割點
        split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
        split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
        
        # 按時間順序分割數據
        train_df = df_z.iloc[:split_point2]
        val_df = df_z.iloc[split_point2:split_point1]
        
        # 創建 Dataset 對象
        if len(train_df) > W + H_out:
            train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
            train_datasets.append(train_ds)
            print(f"  段 {i+1} 訓練集: {len(train_ds)} 個樣本")
        
        if len(val_df) > W + H_out:
            val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
            val_datasets.append(val_ds)
            print(f"  段 {i+1} 驗證集: {len(val_ds)} 個樣本")
    
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

    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])
    criterion = nn.L1Loss()
    
    # ===========================================================================
    # 步驟 5: 選擇訓練策略
    # ===========================================================================
    if 'loss_weighting' in cfg_training and cfg_training['loss_weighting']['weights']:
        weights = cfg_training['loss_weighting']['weights']
        num_windows = len(weights)
        H_block = H_out // num_windows
        
        # 自動選擇策略
        if W > H_block:
            training_step_fn = engine.step_wise_sliding_at_loss_step
            print(f"訓練模式: Sliding Window AT Loss (W={W} > H_block={H_block})")
            print(f"權重: {weights}")
        else:
            training_step_fn = engine.step_wise_rolling_at_loss_step
            print(f"訓練模式: Block Replacement AT Loss (W={W} <= H_block={H_block})")
            print(f"權重: {weights}")
    else:
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
