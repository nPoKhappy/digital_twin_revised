# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, num_en_input, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Keras 的第一個 Dense 層 -> 轉換為線性層做為輸入嵌入
        self.input_embed = nn.Linear(num_en_input, embedding_dim)
        
        # 使用 nn.ModuleList 來存放堆疊的 GRU 層，取代 globals()
        self.gru_layers = nn.ModuleList(
            [nn.GRU(embedding_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True) for i in range(n_layers)]
        )

    def forward(self, src):
        # src shape: (batch_size, seq_len, num_en_input)
        embedded = F.relu(self.input_embed(src))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        outputs = embedded
        hidden_states = []
        
        for i in range(self.n_layers):
            # 每一層的輸出作為下一層的輸入
            outputs, hidden = self.gru_layers[i](outputs)
            # outputs shape: (batch_size, seq_len, hidden_dim)
            # hidden shape: (1, batch_size, hidden_dim)
            hidden_states.append(hidden)
            
        # 返回最後一層的輸出序列和所有層的最終隱藏狀態
        return outputs, hidden_states


class Decoder(nn.Module):
    def __init__(self, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        
        self.input_embed = nn.Linear(num_de_input, embedding_dim)
        
        self.gru_layers = nn.ModuleList(
            [nn.GRU(embedding_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True) for i in range(n_layers)]
        )
        
        # 實現論文中定義的 "Memory Layer"
        # 它接收所有 H 個時間步的最終隱藏層輸出，將它們串接
        # 輸入維度: H * hidden_dim
        # 輸出維度: H * num_output -> 之後 reshape
        self.fc_out = nn.Linear(hidden_dim, num_output)

    def forward(self, dec_input, initial_hiddens):
        # dec_input shape: (batch_size, seq_len, num_de_input)
        # initial_hiddens: list of (1, batch_size, hidden_dim) from encoder
        
        embedded = F.relu(self.input_embed(dec_input))
        
        # 數據流過所有 GRU 層
        gru_output = embedded
        for i, gru_layer in enumerate(self.gru_layers):
            # 將 Encoder 的最終隱藏狀態作為 Decoder 的初始隱藏狀態
            gru_output, _ = gru_layer(gru_output, initial_hiddens[i])
        
        # gru_output shape: (batch_size, seq_len, hidden_dim)
        
        # 論文的 Memory Layer 邏輯: 每個時間步的輸出都通過同一個 Dense 層
        # 這在 PyTorch 中可以高效地一次性完成
        prediction = self.fc_out(gru_output)
        # prediction shape: (batch_size, seq_len, num_output)
        
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(num_en_input, embedding_dim, hidden_dim, n_layers)
        self.decoder = Decoder(num_de_input, num_output, embedding_dim, hidden_dim, n_layers)

    def forward(self, en_input, de_input):
        # 注意：在 AT 訓練模式下，我們不會直接調用這個 forward
        # 而是會分開調用 encoder 和 decoder
        encoder_outputs, encoder_hiddens = self.encoder(en_input)
        predictions = self.decoder(de_input, encoder_hiddens)
        return predictions
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(num_en_input, embedding_dim, hidden_dim, n_layers)
        self.decoder = Decoder(num_de_input, num_output, embedding_dim, hidden_dim, n_layers)

    def forward(self, en_input, de_input):
        encoder_outputs, encoder_hiddens = self.encoder(en_input)
        predictions = self.decoder(de_input, encoder_hiddens)
        return predictions