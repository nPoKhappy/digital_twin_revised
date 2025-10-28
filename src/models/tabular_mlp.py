import torch
import torch.nn as nn
from typing import List

_ACTS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
}

class TabularMLP(nn.Module):
    """
    簡單前饋神經網路 (FNN/MLP) 用於表格迴歸，不依賴時間序列 engine。
    輸入: [B, num_features]
    輸出: [B, num_outputs]
    """
    def __init__(self,
                 num_features: int,
                 num_outputs: int,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        act = _ACTS.get(activation, nn.ReLU)
        dims = [num_features] + hidden_dims
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), act(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], num_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        return self.net(x)
