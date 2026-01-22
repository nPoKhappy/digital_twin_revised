
import torch
import torch.nn as nn
import math

class SinePositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :]

class FusedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.q_decoder = nn.Linear(d_model, d_model)
        self.k_decoder = nn.Linear(d_model, d_model)
        self.v_decoder = nn.Linear(d_model, d_model)

        self.q_encoder = nn.Linear(d_model, d_model)
        self.k_encoder = nn.Linear(d_model, d_model)
        self.v_encoder = nn.Linear(d_model, d_model)

        self.q_dimchange = nn.Linear(d_model * 2, d_model)
        self.k_dimchange = nn.Linear(d_model * 2, d_model)
        self.v_dimchange = nn.Linear(d_model * 2, d_model)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, decoder_input, encoder_input, attn_mask=None, key_padding_mask=None):
        q_dec = self.q_decoder(decoder_input)
        k_dec = self.k_decoder(decoder_input)
        v_dec = self.v_decoder(decoder_input)
        
        q_enc = self.q_encoder(encoder_input)
        k_enc = self.k_encoder(encoder_input)
        v_enc = self.v_encoder(encoder_input)
        
        q = torch.cat([q_dec, q_enc], dim=-1)
        k = torch.cat([k_dec, k_enc], dim=-1)
        v = torch.cat([v_dec, v_enc], dim=-1)
        
        q = self.q_dimchange(q)
        k = self.k_dimchange(k)
        v = self.v_dimchange(v)
        
        attn_out, _ = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        attn_out = self.dropout(attn_out)
        
        return self.norm(decoder_input + attn_out)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_dim, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.cross_attn = FusedCrossAttention(d_model, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, decoder_input, encoder_input, tgt_mask=None):
        if tgt_mask is None:
            seq_len = decoder_input.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(decoder_input.device)
            
        self_out, _ = self.self_attn(decoder_input, decoder_input, decoder_input, attn_mask=tgt_mask)
        x = self.self_norm(decoder_input + self_out)
        x = self.cross_attn(x, encoder_input)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)

class DecoderBlocks(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_dim, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, decoder_input, encoder_input, tgt_mask=None):
        if tgt_mask is None:
            seq_len = decoder_input.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(decoder_input.device)

        self_out, _ = self.self_attn(decoder_input, decoder_input, decoder_input, attn_mask=tgt_mask)
        x = self.self_norm(decoder_input + self_out)
        
        x_concat = torch.cat((encoder_input, x), dim=1)
        cross_out, _ = self.cross_attn(x_concat, x_concat, x_concat)
        x_concat = x_concat + cross_out 
        x_concat = self.self_norm(x_concat)
        
        ffn_out = self.ffn(x_concat)
        ffn_out = self.ffn_norm(x_concat + ffn_out)
        return ffn_out

def get_activation(activation_func):
    if activation_func == 'tanh':
        return nn.Tanh()
    elif activation_func == 'relu':
        return nn.ReLU()
    else:
        return nn.Identity()

class TransformerModel(nn.Module):
    def __init__(self, num_input, num_output, num_embs, intermediate_dim, num_heads, num_layers=1, activation_func='tanh', dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.activation = get_activation(activation_func)
        
        self.enc_dense = nn.Linear(num_input, num_embs)
        self.pos_enc = SinePositionEncoding(num_embs)
        self.dec_dense = nn.Linear(num_input - num_output, num_embs)
        
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=num_embs, nhead=num_heads, dim_feedforward=intermediate_dim, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=num_embs, nhead=num_heads, dim_feedforward=intermediate_dim, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(num_embs)
        self.output_dense = nn.Linear(num_embs, num_output)

    def encoder(self, encoder_input):
        e = self.activation(self.enc_dense(encoder_input))
        e = self.pos_enc(e)
        enc_outputs = []
        curr_e = e
        for layer in self.enc_layers:
            curr_e = layer(curr_e)
            enc_outputs.append(curr_e)
        return curr_e, enc_outputs

    def decoder(self, decoder_input, memory):
        d = self.activation(self.dec_dense(decoder_input))
        d = self.pos_enc(d)
        curr_d = d
        for i, layer in enumerate(self.dec_layers):
            # [MODIFIED] No Causal Mask. Model can see all future MVs.
            curr_d = layer(curr_d, memory[i], tgt_mask=None)
        output = self.layer_norm(curr_d)
        output = self.output_dense(output)
        return output

    def forward(self, encoder_input, decoder_input):
        _, memory = self.encoder(encoder_input)
        return self.decoder(decoder_input, memory)

class TransformerThreeLayersMemoryConnect(nn.Module):
    def __init__(self, num_input, num_output, num_embs, intermediate_dim, num_heads, w=18, activation_func='tanh'):
        super().__init__()
        self.activation = get_activation(activation_func)
        self.w = w
        
        self.enc_dense = nn.Linear(num_input, num_embs)
        self.pos_enc = SinePositionEncoding(num_embs)
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=num_embs, nhead=num_heads, dim_feedforward=intermediate_dim, batch_first=True)
            for _ in range(3)
        ])
        
        self.dec_dense = nn.Linear(num_input - num_output, num_embs)
        self.dec_blocks = nn.ModuleList([
            DecoderBlocks(num_embs, num_heads, intermediate_dim)
            for _ in range(3)
        ])
        
        self.layer_norm = nn.LayerNorm(num_embs)
        self.output_dense = nn.Linear(num_embs, num_output)

    def encoder(self, encoder_input):
        e = self.activation(self.enc_dense(encoder_input))
        e = self.pos_enc(e)
        enc_outputs = []
        curr_e = e
        for layer in self.enc_layers:
            curr_e = layer(curr_e)
            enc_outputs.append(curr_e)
        return curr_e, enc_outputs

    def decoder(self, decoder_input, memory):
        d = self.activation(self.dec_dense(decoder_input))
        d = self.pos_enc(d)
        curr_d = d
        
        # Block 1
        curr_d = self.dec_blocks[0](curr_d, memory[0])
        if curr_d.size(1) > self.w:
            curr_d = curr_d[:, self.w:, :]
            
        # Block 2
        curr_d = self.dec_blocks[1](curr_d, memory[1])
        if curr_d.size(1) > self.w:
            curr_d = curr_d[:, self.w:, :]
            
        # Block 3
        curr_d = self.dec_blocks[2](curr_d, memory[2])
        if curr_d.size(1) > self.w:
            curr_d = curr_d[:, self.w:, :]
            
        output = self.layer_norm(curr_d)
        output = self.output_dense(output)
        return output

    def forward(self, encoder_input, decoder_input):
        _, memory = self.encoder(encoder_input)
        return self.decoder(decoder_input, memory)

class TransformerThreeLayersMemory(nn.Module):
    def __init__(self, num_input, num_output, num_embs, intermediate_dim, num_heads, w=18, activation_func='tanh'):
        super().__init__()
        self.activation = get_activation(activation_func)
        self.w = w
        
        self.enc_dense = nn.Linear(num_input, num_embs)
        self.pos_enc = SinePositionEncoding(num_embs)
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=num_embs, nhead=num_heads, dim_feedforward=intermediate_dim, batch_first=True)
            for _ in range(3)
        ])
        
        self.dec_dense = nn.Linear(num_input - num_output, num_embs)
        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=num_embs, nhead=num_heads, dim_feedforward=intermediate_dim, batch_first=True)
            for _ in range(3)
        ])
        
        self.de_output = nn.Linear(num_embs * 2, num_output) 
        self.layer_norm = nn.LayerNorm(num_embs)
        self.decoder_output_dense = nn.Linear(num_embs, num_output)

    def encoder(self, encoder_input):
        e = self.activation(self.enc_dense(encoder_input))
        e = self.pos_enc(e)
        enc_outputs = []
        curr_e = e
        for layer in self.enc_layers:
            curr_e = layer(curr_e)
            enc_outputs.append(curr_e)
        return curr_e, enc_outputs

    def decoder(self, decoder_input, memory):
        d = self.activation(self.dec_dense(decoder_input))
        d = self.pos_enc(d)
        curr_d = d
        for i, layer in enumerate(self.dec_layers):
            tgt_seq_len = curr_d.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len) * float('-inf'), diagonal=1).to(curr_d.device)
            curr_d = layer(curr_d, memory[i], tgt_mask=tgt_mask)
        final_dec = curr_d
        
        # Assuming we just return the standard output to match user's effective logic
        norm_out = self.layer_norm(final_dec)
        final_out = self.decoder_output_dense(norm_out)
        return final_out

    def forward(self, encoder_input, decoder_input):
        _, memory = self.encoder(encoder_input)
        return self.decoder(decoder_input, memory)

class RollingPredictionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, en_input_1, de_input_1, de_input_2, de_input_3, de_input_4):
        predicted_1 = self.model(en_input_1, de_input_1)
        en_input_2 = torch.cat((predicted_1, de_input_1), dim=2)
        
        predicted_2 = self.model(en_input_2, de_input_2)
        en_input_3 = torch.cat((predicted_2, de_input_2), dim=2)
        
        predicted_3 = self.model(en_input_3, de_input_3)
        en_input_4 = torch.cat((predicted_3, de_input_3), dim=2)
        
        predicted_4 = self.model(en_input_4, de_input_4)
        return predicted_1, predicted_2, predicted_3, predicted_4
