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
            # 合併順序：[MV, y_sv] 以匹配 en_mv_and_sv 的順序
            # 注意：這裡使用 detach()，因為這是 "Standard Step-wise Rolling" 模式
            # 我們不希望梯度通過"歷史數據更新"這條路徑反向傳播 (Teacher Forcing 變體)
            # 或者如果想要全梯度，就不 detach。通常標準 rolling 訓練會截斷梯度以穩定訓練。
            new_step_features = torch.cat([single_step_de_input, single_step_prediction.detach()], dim=2)
            
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
    支持 Point-wise Weighting (權重長度 = 總步數) 或 Block-wise Weighting (權重長度 = 窗口數)。
    """
    en_input_initial, de_inputs, targets = batch
    
    # 将所有数据移动到指定设备
    current_en_input = en_input_initial.to(device)
    all_future_mvs = de_inputs.to(device)      # 维度: (B, total_pred_len, mv_features)
    all_future_targets = targets.to(device)    # 维度: (B, total_pred_len, target_features)

    weights = config['training']['loss_weighting']['weights']
    total_pred_len = all_future_mvs.shape[1]
    
    # H 是每个窗口/块的大小，應從配置讀取 (單次預測長度)
    # Config 結構: window -> prediction_length
    H = config['window']['prediction_length'] 
    
    num_windows = total_pred_len // H
    if total_pred_len % H != 0:
         # 容錯處理：如果總長度不能被 H 整除
         # 這在某些情況下可能發生，這裡簡單處理為向下取整
         pass

    predictions_all_windows = [] # 用来收集每个窗口的预测结果

    # 循环预测每个未来的窗口
    for i in range(num_windows):
        # 提取当前未来窗口所需的 de_input (即 MVs)
        start_idx = i * H
        end_idx = (i + 1) * H
        de_input_block = all_future_mvs[:, start_idx:end_idx, :]
        
        # 使用当前的 en_input 和 de_input_block 进行预测
        prediction_block = model(current_en_input, de_input_block)
        
        # 收集这个窗口的预测结果
        predictions_all_windows.append(prediction_block)
        
        # 准备下一个窗口的输入 (如果不是最后一个窗口)
        if i < num_windows - 1:
            # Autoregressive Update Logic: Block Replacement
            # 將新的預測拼接到 encoder input 的尾部 (滾動)
            # 注意: 這裡假設 prediction_block 包含所有需要的特徵，或者維度對齊
            # 先前邏輯是 cat([de, pred])，這裡簡化為直接 cat(de, pred)
            
            # 確保維度匹配: Encoder Input 通常是 [MV, SV, QV]
            # prediction_block 是 [SV, QV] (Targets)
            # de_input_block 是 [MV]
            # 需要拼接成 [MV, SV, QV] (與 variable_selection 順序一致)
            # 假設 de_input_block 和 prediction_block 的特徵維度總和等於 encoder_input 的特徵維度
            
            # [CRITICAL CHECK]
            # Keras 邏輯: inputs=[enc_input, dec_input] -> output
            # Next enc_input = concatenate([dec_input, output]) (channel axis)
            new_block = torch.cat([de_input_block, prediction_block], dim=2)
            
            # 然後將這個 new_block 放到時間軸的最後，並移除最舊的 H 步
            current_en_input = torch.cat([current_en_input[:, H:, :], new_block], dim=1)

    # --- 计算加权总损失 ---
    all_predictions = torch.cat(predictions_all_windows, dim=1) 
    
    # 處理權重 (Block-wise Only)
    loss_weights = torch.tensor(weights, device=device)
    
    if len(weights) != num_windows:
         # Fallback / Error
         # 如果權重數量不等於窗口數，這裡報錯
         raise ValueError(f"Config Error: Weights length ({len(weights)}) must match number of windows ({num_windows}). Point-wise weighting is no longer supported.")

    # Block-wise Weighting (每個窗口一個權重)
    total_loss = 0
    loss_weights = loss_weights / torch.sum(loss_weights) # Normalize weights to sum to 1 (optional but good practice)
    
    for i in range(num_windows):
        start_idx = i * H
        end_idx = (i + 1) * H
        
        p_block = all_predictions[:, start_idx:end_idx, :]
        t_block = all_future_targets[:, start_idx:end_idx, :]
        
        l_i = criterion(p_block, t_block)
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