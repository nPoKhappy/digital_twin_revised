import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

_ACTS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
}

class TabularMLP(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_outputs: int,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 target_mean: torch.Tensor = None,
                 target_std: torch.Tensor = None):
        super().__init__()
        act = _ACTS.get(activation, nn.ReLU)
        dims = [num_features] + hidden_dims
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), act(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], num_outputs)]
        self.net = nn.Sequential(*layers)
        
        # Physical boundary condition initialized via PINN strategy
        if target_mean is not None and target_std is not None:
            self.register_buffer('target_mean', target_mean.clone().detach().view(1, -1))
            self.register_buffer('target_std', target_std.clone().detach().view(1, -1))
            self.has_physical_bounds = True
        else:
            self.has_physical_bounds = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_z = self.net(x)
        
        if self.has_physical_bounds:
            # Prevent zero division
            std_safe = torch.where(torch.abs(self.target_std) < 1e-6, torch.ones_like(self.target_std), self.target_std)
            # Find the Z-score that corresponds to a physical concentration of 1e-6
            z_lower_bound = (1e-6 - self.target_mean) / std_safe
            
            # Application of Softplus to bound the minimum without killing gradients
            return F.softplus(raw_z - z_lower_bound) + z_lower_bound
            
        return raw_z
