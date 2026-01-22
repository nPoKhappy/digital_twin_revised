import torch
import torch.nn as nn

class GRUSeq2Seq(nn.Module):
    def __init__(self, config):
        """
        PyTorch implementation of the user's Keras GRU Seq2Seq model.
        Architecture:
        Encoder: Input(Feat) -> Dense(Unit, Relu) -> GRU(Unit) -> State
        Decoder: Input(DecFeat) -> Dense(Unit, Relu) -> GRU(Unit, InitState=EncState) -> Dense(TargetDim)
        """
        super(GRUSeq2Seq, self).__init__()
        
        # User config parameters (hardcoded defaults based on request, or from config)
        self.enc_in_dim = config.get('variables_num', 10)  # X_train_encoder.shape[2] (10 variables)
        self.dec_in_dim = 2 # X_train_decoder.shape[2] (2 variables: SPs)
        self.target_dim = 2 # y_train.shape[2] (2 variables: H2S, SO2)
        
        # Keras code: unit = 15
        hidden_dim = config['model'].get('hidden_dim', 15) 
        activation = config['model'].get('activation', 'relu')
        
        # --- Encoder ---
        # hidden_layer = Dense(unit, activation='relu')
        self.enc_dense = nn.Linear(self.enc_in_dim, hidden_dim)
        if activation == 'relu':
            self.enc_act = nn.ReLU()
        else:
            self.enc_act = nn.Tanh() # Default fallback
            
        # encoder = GRU(unit, return_state=True, ...)
        # PyTorch GRU returns (output, h_n). h_n is the final state.
        self.enc_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # --- Decoder ---
        # hidden_layer2 = Dense(unit, activation='relu')
        self.dec_dense = nn.Linear(self.dec_in_dim, hidden_dim)
        self.dec_act = self.enc_act # Share activation type
        
        # decoder_gru = GRU(unit, return_sequences=True, return_state=True, ...)
        self.dec_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # decoder_dense = TimeDistributed(Dense(2))
        # In PyTorch, a Linear layer applied to (Batch, Time, Feat) works like TimeDistributed
        self.output_layer = nn.Linear(hidden_dim, self.target_dim)
        
    def forward(self, x_enc, x_dec):
        """
        x_enc: (Batch, PastSteps, EncFeat) -> e.g. (B, 22, 10)
        x_dec: (Batch, FutureSteps, DecFeat) -> e.g. (B, 12, 2)
        """
        
        # --- Encoder Forward ---
        # 1. Dense + ReLU
        enc_h = self.enc_dense(x_enc) # (B, T_enc, H)
        enc_h = self.enc_act(enc_h)
        
        # 2. GRU
        # we only need the final state (h_n) to initialize decoder
        _, h_n = self.enc_gru(enc_h) # h_n: (NumLayers, Batch, H) -> (1, B, 15)
        
        # --- Decoder Forward ---
        # 1. Dense + ReLU
        dec_h = self.dec_dense(x_dec) # (B, T_dec, H)
        dec_h = self.dec_act(dec_h)
        
        # 2. GRU (Init state = Encoder Final State)
        # return_sequences=True (we want all steps)
        # h_n is used as initial_state
        dec_out, _ = self.dec_gru(dec_h, h_n) # dec_out: (B, T_dec, H)
        
        # 3. Output Dense (TimeDistributed)
        output = self.output_layer(dec_out) # (B, T_dec, 2)
        
        return output
