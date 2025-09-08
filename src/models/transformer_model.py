# src/model.py (最終推薦版本)

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, num_input, embedding_dim, n_heads, n_layers, intermediate_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_input, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=intermediate_dim,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embedding_dim = embedding_dim

    def forward(self, src):
        src = self.input_proj(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        # PyTorch TransformerEncoder 不返回隱狀態，我們直接返回其輸出作為 memory
        memory = self.transformer_encoder(src)
        return memory, memory # <--- 返回 memory 兩次

class Decoder(nn.Module):
    def __init__(self, num_input, num_output, embedding_dim, n_heads, n_layers, intermediate_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_input, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=intermediate_dim,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_fc = nn.Linear(embedding_dim, num_output)
        self.embedding_dim = embedding_dim

    def forward(self, tgt, memory):
        # memory 就是 encoder 的輸出
        tgt = self.input_proj(tgt) * math.sqrt(self.embedding_dim)
        tgt = self.pos_encoder(tgt)
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.output_fc(output)

class Seq2Seq(nn.Module): # 我們保持類名不變，這樣 train.py 不用改
    def __init__(self, num_en_input, num_de_input, num_output, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        # 注意：我們復用 hidden_dim 作為 n_heads 和 intermediate_dim 的基礎
        n_heads = 1 # Transformer 的頭數
        intermediate_dim = hidden_dim * 4 # 這是 Transformer 的常見配置 # feedforward dimension
        
        self.encoder = Encoder(num_en_input, embedding_dim, n_heads, n_layers, intermediate_dim, dropout=0.1)
        self.decoder = Decoder(num_de_input, num_output, embedding_dim, n_heads, n_layers, intermediate_dim, dropout=0.1)

    def forward(self, src, tgt):
        memory, _ = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output