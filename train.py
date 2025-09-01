# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import numpy as np

# 從我們的 src 模組中導入所有需要的東西
from src import data_utils
from src.dataset import S2STimeSeriesDataset
from src.model import Seq2Seq

# --- 1. 設定參數 (集中管理) ---
CONFIG = {
    # 數據相關
    'data_path': './data/',
    'input_file': 'Train_input.csv', # <--- 請填寫您的檔案名
    'total_variables': 30,
    'validation_size': 0.2, # 20% 的數據用於驗證

    # 時間窗口相關
    'en_window_mins': 60, # Encoder 輸入序列長度 (分鐘)
    'de_window_mins': 30, # Decoder 輸入/輸出序列長度 (分鐘)
    'sampling_interval_min': 10, # 數據採樣間隔 (分鐘)

    # 模型超參數 (Hyperparameters)
    'embedding_dim': 32, # 輸入特徵的嵌入維度
    'hidden_dim': 64,    # GRU 隱藏層的維度
    'n_layers': 3,       # GRU 堆疊的層數
    
    # 訓練相關
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'num_epochs': 50,    # 訓練的總輪次
    'learning_rate': 0.001,
    'model_save_path': './saved_models/best_s2s_model.pth' # 模型儲存路徑
}

def collate_fn(batch):
    """
    自定義的 collate_fn，用來過濾掉 Dataset 中返回 None 的樣本。
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# --- 2. 數據準備 ---
print("步驟 1/4: 準備數據...")
# 載入數據 (假設所有變數都在一個 CSV 中)
df_raw = data_utils.load_data(CONFIG['data_path'] + CONFIG['input_file'])
# (此處可以添加您在 main.py 中使用的預處理步驟，如 select_date_range, dropna 等)
df_raw.dropna(inplace=True)

# 數據標準化
mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)

# 獲取變數列表
de_mv, y_sv, con_tag, en_mv_and_sv = data_utils.variable_selection(CONFIG['total_variables'])
num_en_input = len(en_mv_and_sv)
num_de_input = len(de_mv)
num_output = len(y_sv)

# 切分訓練集和驗證集
train_df, val_df = train_test_split(df_z, test_size=CONFIG['validation_size'], shuffle=False) # 時間序列數據不應打亂

# 建立 Dataset
train_dataset = S2STimeSeriesDataset(
    df=train_df,
    en_input_tags=en_mv_and_sv, de_input_tags=de_mv, y_tags=y_sv,
    en_window_mins=CONFIG['en_window_mins'], de_window_mins=CONFIG['de_window_mins'],
    sampling_interval_min=CONFIG['sampling_interval_min']
)
val_dataset = S2STimeSeriesDataset(
    df=val_df,
    en_input_tags=en_mv_and_sv, de_input_tags=de_mv, y_tags=y_sv,
    en_window_mins=CONFIG['en_window_mins'], de_window_mins=CONFIG['de_window_mins'],
    sampling_interval_min=CONFIG['sampling_interval_min']
)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

print(f"數據準備完成。訓練樣本數: {len(train_dataset)}, 驗證樣本數: {len(val_dataset)}")


# --- 3. 模型、損失函數、優化器初始化 ---
print(f"步驟 2/4: 初始化模型... 使用設備: {CONFIG['device']}")

model = Seq2Seq(
    num_en_input=num_en_input,
    num_de_input=num_de_input,
    num_output=num_output,
    embedding_dim=CONFIG['embedding_dim'],
    hidden_dim=CONFIG['hidden_dim'],
    n_layers=CONFIG['n_layers']
).to(CONFIG['device'])

criterion = nn.MSELoss() # 均方誤差損失，適用於回歸預測
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])


# --- 4. 訓練與驗證迴圈 ---
print("步驟 3/4: 開始訓練...")
best_val_loss = float('inf')

for epoch in range(CONFIG['num_epochs']):
    # --- 訓練階段 ---
    model.train()
    total_train_loss = 0
    for en_input, de_input, target in train_loader:
        en_input = en_input.to(CONFIG['device'])
        de_input = de_input.to(CONFIG['device'])
        target = target.to(CONFIG['device'])

        # 梯度歸零
        optimizer.zero_grad()
        # 前向傳播
        predictions = model(en_input, de_input)
        # 計算損失
        loss = criterion(predictions, target)
        # 反向傳播
        loss.backward()
        # 更新權重
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- 驗證階段 ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for en_input, de_input, target in val_loader:
            en_input = en_input.to(CONFIG['device'])
            de_input = de_input.to(CONFIG['device'])
            target = target.to(CONFIG['device'])

            predictions = model(en_input, de_input)
            loss = criterion(predictions, target)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1:02d}/{CONFIG["num_epochs"]:02d}] | 訓練損失: {avg_train_loss:.6f} | 驗證損失: {avg_val_loss:.6f}')

    # --- 儲存最佳模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # 確保儲存模型的資料夾存在
        import os
        os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
        torch.save(model.state_dict(), CONFIG['model_save_path'])
        print(f'  -> 驗證損失降低，模型已儲存至: {CONFIG["model_save_path"]}')


print("步驟 4/4: 訓練完成！")