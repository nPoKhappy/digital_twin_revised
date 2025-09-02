# src/dataset.py (已修改為無 horizon 的版本)

import torch
from torch.utils.data import Dataset

class MultiStepS2SDataset(Dataset):
    def __init__(self, df, en_cols, de_cols, target_cols, input_len_steps, pred_len_steps):
        """
        初始化函數 (已簡化，不再需要 horizons)。
        Args:
            df (pd.DataFrame): 包含時間序列數據的 DataFrame。
            en_cols (list): Encoder 輸入的欄位名稱。
            de_cols (list): Decoder 輸入的欄位名稱。
            target_cols (list): 預測目標的欄位名稱。
            input_len_steps (int): 輸入序列的長度 (W)。
            pred_len_steps (int): 預測序列的長度 (H_out)。
        """
        self.df = df
        self.en_cols = en_cols
        self.de_cols = de_cols
        self.target_cols = target_cols
        self.W = input_len_steps
        self.H_out = pred_len_steps

        # 總樣本長度為輸入 + 輸出
        self.total_len = self.W + self.H_out
        
        # 可生成的樣本總數
        self.num_samples = len(df) - self.total_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        獲取單個樣本。
        """
        # 計算數據切片的起始、中點和結束位置
        start_idx = idx
        mid_idx = start_idx + self.W
        end_idx = mid_idx + self.H_out
        
        # 提取數據切片
        data_slice = self.df.iloc[start_idx:end_idx]

        # 1. Encoder 輸入 (過去 W 步的數據)
        en_input = torch.tensor(data_slice.iloc[:self.W][self.en_cols].values, dtype=torch.float32)

        # 2. Decoder 輸入 (未來 H_out 步的已知變量)
        de_input = torch.tensor(data_slice.iloc[self.W:][self.de_cols].values, dtype=torch.float32)

        # 3. Target (未來 H_out 步的目標變量)
        target = torch.tensor(data_slice.iloc[self.W:][self.target_cols].values, dtype=torch.float32)

        # 返回單個張量，不再是列表
        return en_input, de_input, target