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

class Attention(nn.Module):
    """
    Bahdanau Attention 的實現，用來取代自定義的 Memory 層。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: (batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        
        seq_len = encoder_outputs.shape[1]
        
        # 將 decoder_hidden 擴展以匹配 encoder_outputs 的序列長度
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 計算注意力分數
        concat_hidden = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn_W(concat_hidden)) # (batch_size, seq_len, hidden_dim)
        attention_scores = self.attn_v(energy).squeeze(2) # (batch_size, seq_len)
        
        # 返回 softmax 後的權重
        return F.softmax(attention_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.attention = Attention(hidden_dim)
        
        self.input_embed = nn.Linear(num_de_input, embedding_dim)
        
        # 解碼器的 GRU 輸入維度是 embedding_dim (來自 decoder input) + hidden_dim (來自 context vector)
        self.gru_layers = nn.ModuleList(
            [nn.GRU(embedding_dim + hidden_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True) for i in range(n_layers)]
        )
        
        # 最終的輸出層
        self.fc_out = nn.Linear(hidden_dim, num_output)

    def forward(self, dec_input, encoder_outputs, encoder_hiddens):
        # dec_input shape: (batch_size, seq_len, num_de_input)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        # encoder_hiddens: list of (1, batch_size, hidden_dim)
        
        embedded = F.relu(self.input_embed(dec_input))
        
        # 這裡我們一次處理整個序列，模仿 Keras 的行為
        # 注意：在真實的推論中，通常是一個時間步一個時間步地進行
        outputs_list = []
        for t in range(embedded.shape[1]): # 遍歷時間步
            current_input = embedded[:, t, :] # (batch_size, embedding_dim)
            
            # 使用最後一層 encoder 的隱藏狀態來計算注意力
            # hidden[-1] shape: (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)
            last_encoder_hidden = encoder_hiddens[-1].squeeze(0)
            
            attn_weights = self.attention(last_encoder_hidden, encoder_outputs)
            attn_weights = attn_weights.unsqueeze(1) # (batch_size, 1, seq_len)
            
            # 計算上下文向量 (context vector)
            context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
            # context shape: (batch_size, hidden_dim)
            
            # 將當前輸入和上下文向量拼接
            rnn_input = torch.cat((current_input, context), dim=1)
            
            # 將拼接後的向量傳遞給 GRU 層
            gru_output = rnn_input
            decoder_hiddens = []
            for i, gru_layer in enumerate(self.gru_layers):
                # Keras 的 initial_state -> PyTorch GRU 的第二個參數
                # 注意：這裡的 hidden state 傳遞邏輯比 Keras 的更靈活
                # 為了簡化並匹配 Attention，我們只用了最後一層的 hidden
                gru_output, hidden = gru_layer(gru_output.unsqueeze(1), encoder_hiddens[i])
                gru_output = gru_output.squeeze(1)
                decoder_hiddens.append(hidden)

            outputs_list.append(gru_output)

        # 將所有時間步的輸出堆疊起來
        final_outputs = torch.stack(outputs_list, dim=1)
        prediction = self.fc_out(final_outputs)
        
        return prediction


class Seq2Seq(nn.Module):
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(num_en_input, embedding_dim, hidden_dim, n_layers)
        self.decoder = Decoder(num_de_input, num_output, embedding_dim, hidden_dim, n_layers)

    def forward(self, en_input, de_input):
        encoder_outputs, encoder_hiddens = self.encoder(en_input)
        predictions = self.decoder(de_input, encoder_outputs, encoder_hiddens)
        return predictions