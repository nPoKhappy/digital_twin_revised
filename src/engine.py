# src/engine.py (實現真正的逐步滾動訓練)

import torch
import torch.nn as nn
from tqdm import tqdm
# ==============================================================================
# --- 模式一：標準的逐步滾動訓練 (無 AT Loss) ---
# ==============================================================================
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
        encoder_outputs, context = model.encoder(current_en_input)

        # b. 準備單步 Decoder 輸入 (從已知的未來MV中獲取)
        single_step_de_input = all_future_mvs[:, t, :].unsqueeze(1)

        # c. 解碼器進行單步預測
        single_step_prediction = model.decoder(single_step_de_input, context)
        
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


# ==============================================================================
# --- 模式二：帶有 AT Loss 的逐步滾動訓練 ---
# ==============================================================================
def step_wise_rolling_at_loss_step(model, batch, criterion, device, config):
    """
    策略：在 PyTorch 中实现与 Keras Three_window_pred 等价的“块替换”滚动训练。
    梯度会在整个块预测链条中反向传播。
    """
    en_input_initial, de_inputs, targets = batch
    
    # 将所有数据移动到指定设备
    current_en_input = en_input_initial.to(device)
    all_future_mvs = de_inputs.to(device)      # 维度: (B, total_pred_len, mv_features)
    all_future_targets = targets.to(device) # 维度: (B, total_pred_len, target_features)

    weights = config['training']['loss_weighting']['weights']
    num_windows = len(weights)
    total_pred_len = all_future_mvs.shape[1]
    
    # H 是每个窗口/块的大小
    H = total_pred_len // num_windows
    if total_pred_len % num_windows != 0:
        raise ValueError("prediction_length 必须是权重数量 (num_windows) 的整数倍")

    predictions_all_windows = [] # 用来收集每个窗口的预测结果

    # --- 关键的块预测链条 ---
    
    # 循环预测每个未来的窗口
    for i in range(num_windows):
        # 提取当前未来窗口所需的 de_input (即 MVs)
        start_idx = i * H
        end_idx = (i + 1) * H
        de_input_block = all_future_mvs[:, start_idx:end_idx, :]
        
        # 使用当前的 en_input 和 de_input_block 进行预测
        # current_en_input 在第一次循环时是 en_input_initial，之后是上一步的预测结果
        prediction_block = model(current_en_input, de_input_block)
        
        # 收集这个窗口的预测结果
        predictions_all_windows.append(prediction_block)
        
        # 准备下一个窗口的输入 (如果不是最后一个窗口)
        if i < num_windows - 1:
            # **核心逻辑：将上一步的完整预测块 (QV/SV) 和它对应的
            # de_input 块 (MV) 拼接，形成下一个 en_input。**
            # **这里没有滑动，是完整的替换。**
            # **关键：没有 .detach()**
            current_en_input = torch.cat([prediction_block, de_input_block], dim=2)

    # --- 计算加权总损失 ---

    # 将所有窗口的预测结果合并成一个大的张量
    # predictions_all_windows 是一个 list of tensors, 每个 tensor 维度是 (B, H, target_features)
    all_predictions = torch.cat(predictions_all_windows, dim=1) # 沿序列长度维度拼接
    
    total_loss = 0
    loss_weights = torch.tensor(weights, device=device)
    loss_weights = loss_weights / torch.sum(loss_weights) # 权重归一化

    # 按窗口计算损失
    for i in range(num_windows):
        start_idx = i * H
        end_idx = (i + 1) * H
        
        # 提取对应窗口的预测和目标
        window_predictions = all_predictions[:, start_idx:end_idx, :]
        window_targets = all_future_targets[:, start_idx:end_idx, :]
        
        # 计算这个窗口的平均损失
        l_i = criterion(window_predictions, window_targets)
        
        # 乘以权重并累加
        total_loss += loss_weights[i] * l_i
            
    return total_loss


    """
    策略：在 PyTorch 中实现与 Keras 版本等价的端到端滚动训练。
    梯度会在整个预测链条中反向传播，没有截断。
    """
    en_input_initial, de_inputs, targets = batch
    
    # 将所有数据移动到指定设备
    current_en_input = en_input_initial.to(device)
    all_future_mvs = de_inputs.to(device)      # 维度: (B, total_pred_len, mv_features)
    all_future_targets = targets.to(device) # 维度: (B, total_pred_len, target_features)

    weights = config['training']['loss_weighting']['weights']
    num_windows = len(weights)
    total_pred_len = all_future_mvs.shape[1]
    
    # H 是每个权重窗口包含的时间步数
    H = total_pred_len // num_windows
    if total_pred_len % num_windows != 0:
        raise ValueError("prediction_length 必须是权重数量 (num_windows) 的整数倍")

    predictions = [] # 用来收集每一步的预测结果

    # --- 关键的滚动预测链条 ---
    
    # 第一步预测 (t=0)
    # 提取第一个时间步的解码器输入
    de_input_step_0 = all_future_mvs[:, 0, :].unsqueeze(1) 
    # 进行预测，此时 current_en_input 是真实的初始历史数据
    prediction_step_0 = model(current_en_input, de_input_step_0)
    predictions.append(prediction_step_0)

    # 准备下一次的编码器输入
    # **关键：这里没有 .detach()**
    new_features = torch.cat([prediction_step_0, de_input_step_0], dim=2)
    next_en_input_history = current_en_input[:, 1:, :]
    current_en_input = torch.cat([next_en_input_history, new_features], dim=1)

    # 后续步骤的预测 (t=1 to total_pred_len - 1)
    # 我们用一个循环来表示这个链式过程，但重要的是梯度会一直传递
    for t in range(1, total_pred_len):
        de_input_step_t = all_future_mvs[:, t, :].unsqueeze(1)
        
        # 使用上一步合成的 current_en_input 进行预测
        prediction_step_t = model(current_en_input, de_input_step_t)
        predictions.append(prediction_step_t)
        
        # 如果不是最后一步，则继续准备下一次的输入
        if t < total_pred_len - 1:
            # **关键：同样没有 .detach()**
            new_features = torch.cat([prediction_step_t, de_input_step_t], dim=2)
            next_en_input_history = current_en_input[:, 1:, :]
            current_en_input = torch.cat([next_en_input_history, new_features], dim=1)

    # --- 计算加权总损失 ---

    # 将所有单步预测结果合并成一个大的张量
    all_predictions = torch.cat(predictions, dim=1) # 维度: (B, total_pred_len, target_features)
    
    total_loss = 0
    loss_weights = torch.tensor(weights, device=device)
    loss_weights = loss_weights / torch.sum(loss_weights) # 权重归一化

    # 按窗口计算损失
    for i in range(num_windows):
        start_idx = i * H
        end_idx = (i + 1) * H
        
        # 提取对应窗口的预测和目标
        window_predictions = all_predictions[:, start_idx:end_idx, :]
        window_targets = all_future_targets[:, start_idx:end_idx, :]
        
        # 计算这个窗口的平均损失
        l_i = criterion(window_predictions, window_targets)
        
        # 乘以权重并累加
        total_loss += loss_weights[i] * l_i
            
    return total_loss



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