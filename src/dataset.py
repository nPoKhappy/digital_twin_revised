# src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from datetime import timedelta

class S2STimeSeriesDataset(Dataset):
    def __init__(self, df, en_input_tags, de_input_tags, y_tags,
                 en_window_mins, de_window_mins, sampling_interval_min):
        """
        PyTorch Dataset for Sequence-to-Sequence time series data.

        Args:
            df (pd.DataFrame): 包含所有特徵和目標的預處理後 DataFrame。
            en_input_tags (list): Encoder 輸入特徵的欄位名稱列表。
            de_input_tags (list): Decoder 輸入特徵的欄位名稱列表。
            y_tags (list): Target (y) 的欄位名稱列表。
            en_window_mins (int): Encoder 輸入序列的長度 (分鐘)。
            de_window_mins (int): Decoder 輸入/輸出序列的長度 (分鐘)。
            sampling_interval_min (int): 數據點之間的採樣間隔 (分鐘)。
        """
        self.en_input_df = df[en_input_tags]
        self.de_input_df = df[de_input_tags]
        self.y_df = df[y_tags]

        self.en_window_steps = en_window_mins // sampling_interval_min
        self.de_window_steps = de_window_mins // sampling_interval_min
        self.sampling_interval = timedelta(minutes=sampling_interval_min)

        # 建立有效樣本的起始時間點列表
        self.valid_indices = []
        # 從數據的 en_window_steps-th 位置開始，確保有足夠的歷史數據
        for i in range(self.en_window_steps, len(df) - self.de_window_steps):
            self.valid_indices.append(df.index[i])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 根據索引找到當前的時間戳 (這是預測開始的時間點 t)
        t_current = self.valid_indices[idx]

        # 1. Encoder 輸入 (過去的數據)
        # 時間範圍：[t_current - en_window, t_current - 1*interval]
        en_end = t_current - self.sampling_interval
        en_start = t_current - timedelta(minutes=self.en_window_steps * self.sampling_interval.total_seconds() / 60)
        
        en_input_data = self.en_input_df.loc[en_start:en_end].values

        # 2. Decoder 輸入 (未來的已知輸入)
        # 時間範圍：[t_current, t_current + de_window - 1*interval]
        de_end = t_current + timedelta(minutes=(self.de_window_steps - 1) * self.sampling_interval.total_seconds() / 60)
        
        de_input_data = self.de_input_df.loc[t_current:de_end].values

        # 3. Target 目標 (未來的預測目標)
        # 時間範圍與 Decoder 輸入相同
        y_data = self.y_df.loc[t_current:de_end].values
        
        # 進行一致性檢查，確保切片長度正確
        if len(en_input_data) != self.en_window_steps or \
           len(de_input_data) != self.de_window_steps or \
           len(y_data) != self.de_window_steps:
            # 如果長度不對，返回 None，之後在 DataLoader 中過濾掉
            return None

        return (
            torch.from_numpy(en_input_data).float(),
            torch.from_numpy(de_input_data).float(),
            torch.from_numpy(y_data).float()
        )