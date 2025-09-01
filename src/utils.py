# 可以放在 main.py 或 src/utils.py

import torch

def predict_autoregressive(model, en_input, de_input_initial, steps):
    """
    使用模型進行自回歸滾動預測。

    Args:
        model (nn.Module): 訓練好的 Seq2Seq 模型。
        en_input (Tensor): 編碼器的輸入 (batch_size, seq_len, num_en_input)。
        de_input_initial (Tensor): 解碼器的初始輸入 (batch_size, seq_len, num_de_input)。
        steps (int): 要向前預測的總步數。

    Returns:
        list: 包含每一步預測結果的 Tensor 列表。
    """
    model.eval()  # 設置為評估模式
    predictions = []
    
    with torch.no_grad(): # 在預測時不需要計算梯度
        # 獲取初始的 encoder 狀態
        encoder_outputs, encoder_hiddens = model.encoder(en_input)

        # 初始的 decoder input
        current_de_input = de_input_initial

        for _ in range(steps):
            # 進行單步預測
            output = model.decoder(current_de_input, encoder_outputs, encoder_hiddens)
            predictions.append(output)

            # --- 準備下一個時間步的輸入 ---
            # 這是最關鍵的部分，需要根據您的具體數據來決定如何構建下一個輸入
            # 這裡的邏輯需要模仿 Keras 的 K.concatenate((predicted_1, de_input_1), axis=2)
            # 假設預測的輸出 (output) 需要和某些已知的未來輸入拼接
            # 為了示例，我們假設下一個 decoder input 就是當前的預測結果
            # 您需要根據您的特徵來修改這部分！
            
            # 假設 de_input_initial 的特徵維度與 output 的特徵維度相同
            # current_de_input = output # 這是一個簡化的例子
            
            # 更接近 Keras 的例子 (假設 de_input 的某些欄位是已知的)
            # known_future_features = current_de_input[:, :, num_predicted_features:]
            # current_de_input = torch.cat([output, known_future_features], dim=2)
            
            # 由於邏輯可能很複雜，我們先在這裡中斷，讓您知道需要在此處定義滾動邏輯
            pass # 您需要在此處實現如何用 output 構造下一個 current_de_input

    return predictions