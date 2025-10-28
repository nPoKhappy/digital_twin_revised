import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
    """非時間序列資料集：單列即一筆樣本"""
    def __init__(self, df, input_cols, target_cols):
        self.X = df[input_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
