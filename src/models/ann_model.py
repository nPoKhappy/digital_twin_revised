# src/models/ann_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    ANN Encoder: 將輸入序列壓縮為固定維度的特徵向量
    """
    def __init__(self, num_en_input, embedding_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.num_en_input = num_en_input
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 第一層：輸入嵌入層
        self.input_embed = nn.Linear(num_en_input, embedding_dim)
        
        # 中間的隱藏層
        self.hidden_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(embedding_dim, hidden_dim))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Dropout 層用於正則化
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層：將序列壓縮為固定維度的上下文向量
        self.context_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, num_en_input)
        batch_size, seq_len, _ = src.shape
        
        # 將序列展平處理：逐個時間步處理然後平均/最大池化
        # 方法1: 全局平均池化
        src_flattened = src.reshape(batch_size, -1)  # (batch_size, seq_len * num_en_input)
        
        # 輸入嵌入
        embedded = F.relu(self.input_embed(src.mean(dim=1)))  # 對時間維度求平均
        # embedded shape: (batch_size, embedding_dim)
        
        # 通過隱藏層
        hidden_output = embedded
        for layer in self.hidden_layers:
            hidden_output = F.relu(layer(hidden_output))
            hidden_output = self.dropout(hidden_output)
        
        # 生成上下文向量
        context = F.relu(self.context_layer(hidden_output))
        
        # 為了與現有的 Seq2Seq 架構兼容，我們返回兩個值
        # 第一個是序列輸出（這裡我們重複使用 context），第二個是隱狀態
        return context.unsqueeze(1), context  # (batch_size, 1, hidden_dim), (batch_size, hidden_dim)


class Decoder(nn.Module):
    """
    ANN Decoder: 從上下文向量生成輸出序列
    """
    def __init__(self, num_de_input, num_output, embedding_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        self.num_de_input = num_de_input
        self.num_output = num_output
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 輸入嵌入層
        self.input_embed = nn.Linear(num_de_input, embedding_dim)
        
        # 上下文融合層：將上下文向量與輸入特徵結合
        self.context_fusion = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        
        # 中間的隱藏層
        self.hidden_layers = nn.ModuleList()
        for i in range(n_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_dim, num_output)
        
    def forward(self, de_input, context):
        # de_input shape: (batch_size, seq_len, num_de_input)
        # context shape: (batch_size, hidden_dim)
        
        batch_size, seq_len, _ = de_input.shape
        
        # 輸入嵌入
        embedded = F.relu(self.input_embed(de_input))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # 將上下文向量擴展到每個時間步
        context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
        # context_expanded shape: (batch_size, seq_len, hidden_dim)
        
        # 融合上下文和輸入特徵
        fused_input = torch.cat([embedded, context_expanded], dim=-1)
        # fused_input shape: (batch_size, seq_len, embedding_dim + hidden_dim)
        
        # 上下文融合
        hidden_output = F.relu(self.context_fusion(fused_input))
        # hidden_output shape: (batch_size, seq_len, hidden_dim)
        
        # 通過隱藏層
        for layer in self.hidden_layers:
            hidden_output = F.relu(layer(hidden_output))
            hidden_output = self.dropout(hidden_output)
        
        # 生成輸出
        output = self.output_layer(hidden_output)
        # output shape: (batch_size, seq_len, num_output)
        
        return output


class Seq2Seq(nn.Module):
    """
    ANN Seq2Seq 模型：基於前饋神經網絡的序列到序列模型
    適用於時間序列預測任務
    """
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(
            num_en_input=num_en_input,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            num_de_input=num_de_input,
            num_output=num_output,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
    def forward(self, en_input, de_input):
        # Encoder: 將輸入序列編碼為上下文向量
        encoder_output, context = self.encoder(en_input)
        
        # Decoder: 基於上下文向量生成輸出序列
        predictions = self.decoder(de_input, context)
        
        return predictions
    
    def get_model_info(self):
        """返回模型的基本信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ANN (Artificial Neural Network)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_layers': len(self.encoder.hidden_layers),
            'decoder_layers': len(self.decoder.hidden_layers)
        }


class SimpleANN(nn.Module):
    """
    簡化版的 ANN 模型：直接從輸入序列預測輸出序列
    不使用 Encoder-Decoder 架構
    """
    def __init__(self, input_size, output_size, hidden_dims=[128, 64, 32], dropout=0.1):
        super().__init__()
        
        # 構建網絡層
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(prev_dim, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features) 或 (batch_size, features)
        if len(x.shape) == 3:
            # 如果是序列輸入，展平處理 - 使用 reshape() 而非 view()
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size, -1)  # (batch_size, seq_len * features)
        
        return self.network(x)


class SimpleANNSeq2Seq(nn.Module):
    """
    Simple ANN 的 Seq2Seq 包裝器：將 SimpleANN 適配到現有的 Seq2Seq 訓練框架
    """
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers, dropout=0.1):
        super().__init__()
        
        # 計算輸入和輸出尺寸
        # 假設 encoder 輸入序列長度為 18 (train_window_mins=180, sampling_interval_min=10)
        # decoder 輸入序列長度為 18 (prediction_length=72, 每4個時間步為一組，共18組)
        self.encoder_seq_len = 18
        self.decoder_seq_len = 18
        
        # 總輸入尺寸 = encoder序列長度 * encoder特徵數 + decoder序列長度 * decoder特徵數
        total_input_size = self.encoder_seq_len * num_en_input + self.decoder_seq_len * num_de_input
        
        # 總輸出尺寸 = decoder序列長度 * 輸出特徵數
        total_output_size = self.decoder_seq_len * num_output
        
        # 根據 n_layers 和 hidden_dim 構建隱藏層維度列表
        hidden_dims = []
        for i in range(n_layers):
            if i == 0:
                hidden_dims.append(embedding_dim)
            else:
                hidden_dims.append(hidden_dim)
        
        # 創建 SimpleANN
        self.ann = SimpleANN(
            input_size=total_input_size,
            output_size=total_output_size,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.num_output = num_output
        
    def forward(self, en_input, de_input):
        # en_input shape: (batch_size, encoder_seq_len, num_en_input)
        # de_input shape: (batch_size, decoder_seq_len, num_de_input)
        
        batch_size = en_input.shape[0]
        
        # 展平輸入序列 - 使用 reshape() 而非 view() 以處理非連續張量
        en_flat = en_input.reshape(batch_size, -1)  # (batch_size, encoder_seq_len * num_en_input)
        de_flat = de_input.reshape(batch_size, -1)  # (batch_size, decoder_seq_len * num_de_input)
        
        # 連接 encoder 和 decoder 輸入
        combined_input = torch.cat([en_flat, de_flat], dim=1)
        
        # 通過 ANN 網絡
        output_flat = self.ann(combined_input)
        
        # 重塑輸出為序列格式 - 使用 reshape() 而非 view()
        output = output_flat.reshape(batch_size, self.decoder_seq_len, self.num_output)
        
        return output
    
    def get_model_info(self):
        """返回模型的基本信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Simple ANN (Feed-Forward Neural Network)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'network_layers': len(self.ann.network),
            'architecture': 'Direct Input-Output Mapping'
        }
