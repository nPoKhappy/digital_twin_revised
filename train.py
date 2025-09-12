# ==============================================================================
# 訓練腳本 - 支持多種模型和訓練策略的時間序列預測模型訓練
# ==============================================================================

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import yaml
import argparse

# 導入自定義模組
from src import data_utils, engine  # 數據工具和訓練引擎
from src.dataset import MultiStepS2SDataset  # 序列到序列數據集類
from src.models import get_model  # 模型工廠，支持多種模型架構

def main(config_path):
    """
    主訓練函數
    Args:
        config_path (str): YAML 配置文件的路径
    """
    # ===========================================================================
    # 步驟 1: 載入實驗配置
    # ===========================================================================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) # 將 YAML 文件內容載入為字典

    prefix = config['exp_name']  # 實驗名稱，用於保存模型和結果
    print(f"========== 開始實驗: {prefix} (模型: {config['model']['name']}) ==========")
    start_time = time.time()  # 記錄訓練開始時間

    # ===========================================================================
    # 步驟 2: 數據準備與預處理
    # ===========================================================================
    print("\n步驟 1/3: 準備訓練與驗證數據...")
    cfg_data = config['data']      # 數據相關配置
    cfg_win = config['window']     # 時間窗口相關配置
    
    # 計算輸入窗口大小 (分鐘轉換為時間步數)
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    H_out = cfg_win['prediction_length']  # 預測窗口長度
    
    # 載入原始數據並進行基本清理
    # 安全地載入數據，處理可選的 DateTime 索引
    try:
        df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
        print("成功載入訓練數據（帶 DateTime 索引）")
    except (KeyError, ValueError) as e:
        print(f"注意：數據中沒有 DateTime 列，使用預設索引載入: {e}")
        # 如果沒有 DateTime 列，直接讀取 CSV
        import pandas as pd
        df_raw = pd.read_csv(os.path.join(cfg_data['path'], cfg_data['filename']))
        print("成功載入訓練數據（使用預設索引）")
    
    df_raw = df_raw.iloc[:cfg_data['point']]  # 截取指定數量的數據點
    df_raw.dropna(inplace=True)               # 移除缺失值

    # 計算 Z-score 標準化參數並應用標準化
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)  # 計算均值和標準差
    df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)      # 標準化數據

    # 變量選擇：分離輸入變量和目標變量
    de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
    # de_mv: decoder 輸入變量 (控制變量)
    # y_sv: 目標變量 (需要預測的變量)  
    # en_mv_and_sv: encoder 輸入變量 (歷史數據，包含控制變量和目標變量)
    
    # 將變量維度信息存回配置，供模型工廠使用
    config['data']['num_en_input'] = len(en_mv_and_sv)  # encoder 輸入特徵數
    config['data']['num_de_input'] = len(de_mv)         # decoder 輸入特徵數  
    config['data']['num_output'] = len(y_sv)            # 輸出特徵數
    
    # ===========================================================================
    # 步驟 3: 數據分割與 DataLoader 創建
    # ===========================================================================
    data_len = len(df_z)  # 標準化後的數據總長度
    
    # 計算數據分割點
    # split_point1: 訓練+驗證 與 測試 的分界點
    split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
    # split_point2: 訓練 與 驗證 的分界點  
    split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
    
    # 按時間順序分割數據 (重要：時間序列不能隨機分割)
    train_df = df_z.iloc[:split_point2]           # 訓練集：最早的數據
    val_df = df_z.iloc[split_point2:split_point1] # 驗證集：中間的數據
    # 測試集將是：df_z.iloc[split_point1:]      # 測試集：最新的數據 (在預測腳本中使用)
    
    # 創建 Dataset 對象：將連續時間序列切割成重疊的訓練樣本
    train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    val_ds = MultiStepS2SDataset(val_df, en_mv_and_sv, de_mv, y_sv, W, H_out)
    
    # 創建 DataLoader：將樣本組織成批次用於訓練
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)   # 訓練時打亂順序
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)     # 驗證時保持順序
    print("數據準備完成。")

    # ===========================================================================
    # 步驟 4: 模型初始化與訓練設置
    # ===========================================================================
    print(f"\n步驟 2/3: 初始化 {config['model']['name']} 模型...")
    cfg_training = config['training']
    
    # 設備選擇：優先使用 GPU，不可用時使用 CPU
    device = 'cuda' if torch.cuda.is_available() and cfg_training['device'] == 'cuda' else 'cpu'
    
    # 創建模型並移到指定設備
    model = get_model(config).to(device)  # 使用模型工廠創建模型
    
    # 初始化優化器和損失函數
    optimizer = optim.Adam(model.parameters(), lr=cfg_training['learning_rate'])  # Adam 優化器
    criterion = nn.L1Loss()  # L1 損失函數 (平均絕對誤差)
    
    # ===========================================================================
    # 步驟 5: 選擇訓練策略
    # ===========================================================================
    # 根據配置文件自動選擇訓練模式：標準訓練 vs 帶權重的 AT Loss 訓練
    if 'loss_weighting' in cfg_training and cfg_training['loss_weighting']['weights']:
        # 使用加權損失的逐步滾動訓練 (Attention-based Time-aware Loss)
        training_step_fn = engine.step_wise_rolling_at_loss_step
        print(f"訓練模式: Step-wise Rolling with AT Loss (權重: {cfg_training['loss_weighting']['weights']})")
    else:
        # 使用標準的逐步滾動訓練
        training_step_fn = engine.step_wise_rolling_training_step
        print("訓練模式: Standard Step-wise Rolling")
    print(f"滾動預測長度: {H_out} 步")
    
    # ===========================================================================
    # 步驟 6: 訓練循環與早停機制
    # ===========================================================================
    print("\n步驟 3/3: 開始訓練...")
    
    # 初始化訓練狀態
    best_val_loss = float('inf')  # 記錄最佳驗證損失
    patience_counter = 0          # 早停計數器
    
    # 創建模型保存目錄
    os.makedirs('./saved_models/', exist_ok=True)
    model_save_path = os.path.join('./saved_models/', f'{prefix}.pth')

    # 開始訓練循環
    for epoch in range(cfg_training['epochs']):
        # 訓練一個 epoch
        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, criterion, device, training_step_fn, config
        )
        
        # 在驗證集上評估模型
        val_loss = engine.evaluate(
            model, val_loader, criterion, device, training_step_fn, config
        )
        
        print(f'Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        # 早停機制：如果驗證損失改善則保存模型，否則增加計數器
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)  # 保存最佳模型
            print(f'  -> 驗證損失降低，最佳模型已儲存至: {model_save_path}')
            patience_counter = 0  # 重置早停計數器
        else:
            patience_counter += 1  # 增加早停計數器
            
        # 如果連續多個 epoch 沒有改善，則早停
        if patience_counter >= cfg_training['patience']:
            print(f"Early stopping at epoch {epoch+1}! (連續 {cfg_training['patience']} 個 epoch 無改善)")
            break
            
    # 計算並顯示總訓練時間        
    end_time = time.time()
    print(f"\n訓練完成。總耗時: {(end_time - start_time)/60:.2f} 分鐘。")

if __name__ == '__main__':
    """
    腳本入口點：解析命令行參數並執行訓練
    
    使用方法：
    python train.py --config configs/experiment_config.yaml
    """
    parser = argparse.ArgumentParser(description="時間序列預測模型訓練腳本")
    parser.add_argument('--config', type=str, required=True, 
                       help='YAML 配置文件的路径，包含模型、數據和訓練參數')
    args = parser.parse_args()
    main(args.config)