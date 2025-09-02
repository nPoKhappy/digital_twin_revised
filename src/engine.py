# src/engine.py (實現真正的逐步滾動訓練)

import torch
import torch.nn as nn
from tqdm import tqdm

def step_wise_rolling_training_step(model, batch, criterion, device, loss_fraction=None):
    """
    策略：執行逐時間步的滾動訓練，並累積每一步的損失。
    (此版本已更新，不再使用 horizons，並實現真正的滾動邏輯)
    """
    # 1. 從批次中解包數據
    en_input_initial, de_inputs, targets = batch
    
    # 將數據移動到設備
    current_en_input = en_input_initial.clone().to(device)
    all_future_mvs = de_inputs.to(device)
    all_future_targets = targets.to(device)

    # 獲取總的預測步數 (例如 30)
    n_steps = all_future_mvs.shape[1]
    
    total_loss = 0
    
    # 2. 逐時間步進行滾動
    for t in range(n_steps):
        # a. 獲取當前 Encoder 的記憶
        # 注意：在每個時間步，我們都重新對更新後的 current_en_input 進行編碼
        _, current_encoder_hiddens = model.encoder(current_en_input)

        # b. 準備單步 Decoder 輸入 (從已知的未來MV中獲取)
        single_step_de_input = all_future_mvs[:, t, :].unsqueeze(1)

        # c. 解碼器進行單步預測
        single_step_prediction = model.decoder(single_step_de_input, current_encoder_hiddens)
        
        # d. 計算這一步的損失
        single_step_target = all_future_targets[:, t, :].unsqueeze(1)
        loss = criterion(single_step_prediction, single_step_target)
        
        # e. 累積總損失
        total_loss += loss

        # f. 更新 Encoder 輸入以備下一步使用 (這就是滾動的核心)
        if t < n_steps - 1:
            # 拿掉最舊的一步
            next_en_input_history = current_en_input[:, 1:, :]
            
            # 創建新的一步: "dec 1 變成 enc 最後"
            # 使用 detach() 阻止不必要的梯度流，簡化反向傳播路徑
            new_step_features = torch.cat([single_step_prediction.detach(), single_step_de_input], dim=2)
            
            # 拼接成新的 Encoder 輸入
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)

    # 返回所有步數累加的平均損失 (通常返回平均值更穩定)
    return total_loss / n_steps

def train_one_epoch(model, dataloader, optimizer, criterion, device, training_step_fn, loss_fraction=None):
    model.train()
    total_loss = 0
    
    # 在 train.py 中，我們將傳入上面的 step_wise_rolling_training_step
    for batch in tqdm(dataloader, desc="Training Progress"):
        optimizer.zero_grad()
        loss = training_step_fn(model, batch, criterion, device, loss_fraction)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, training_step_fn, loss_fraction=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Progress"):
            loss = training_step_fn(model, batch, criterion, device, loss_fraction)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)