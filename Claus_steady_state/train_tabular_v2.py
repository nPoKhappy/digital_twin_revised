import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. 完整標籤對接清單 (aspen dynamics 原始標籤 -> Python 模型中使用的標籤)
# =====================================================================
FULL_MAPPING = {
    # ── 1. 設定點與特殊對應 (Setpoints & Custom Logic) ──
    'B34.SPo.SPo': 'acidgas_Fm',
    'B17.PV.PV': 'air',
    'S20.P.P': 'HEATER1_output_P',
    'B33.SPo.SPo': 'air2_SP',
    'B17.SPo.SPo': 'air_SP',
    'B35.SPo.SPo': 'COG_SP',
    'AIR2.Fv.Fv': 'second_air2',
    'S4.Fv.Fv': 'COG',

    # ── 2. 反應器與壓力控制 (Reactors & PC) ──
    'B18.SPo.SPo': 'burner_input_T_SP',
    'B18.PV.PV': 'burner_input_T_PV',
    'B19.SPo.SPo': 'burner_output_T_SP',
    'B19.PV.PV': 'burner_output_T_PV',
    'BURNER_PC.SPo.SPo': 'burner_output_P_SP',
    'BURNER_PC.PV.PV': 'burner_output_P_PV',
    'FURANCE_PC.SPo.SPo': 'fur_outputP_SP',
    'FURANCE_PC.PV.PV': 'fur_outputP_PV',
    'FURANCE.T.0.(0)': 'fur_inputT',
    'FURANCE.T.1.(1)': 'fur_temp',

    # ── 3. 分離器與換熱設備 (SEPs & Heaters) ──
    'SEP1_PC.SPo.SPo': 'SEP1_P_SP',
    'SEP1_PC.PV.PV': 'SEP1_P_PV',
    'SEP1.T.T': 'SEP1_T',
    'SEP2_PC.SPo.SPo': 'SEP2_P_SP',
    'SEP2_PC.PV.PV': 'SEP2_P_PV',
    'SEP2.T.T': 'SEP2_T',
    'SEP3_PC.SPo.SPo': 'SEP3_P_SP',
    'SEP3_PC.PV.PV': 'SEP3_P_PV',
    'SEP3.T.T': 'SEP3_T',
    'B21.SPo.SPo': 'HEATER1_output_T_SP',
    'B21.PV.PV': 'HEATER1_output_T_PV',
    'B20.SPo.SPo': 'HEATER2_output_T_SP',
    'B20.PV.PV': 'HEATER2_output_T_PV',
    'CAT1_PC.SPo.SPo': 'cat1_output_P_SP',
    'CAT1_PC.PV.PV': 'cat1_output_P_PV',
    'CAT2_PC.SPo.SPo': 'cat2_output_P_SP',
    'CAT2_PC.PV.PV': 'cat2_output_P_PV',

    # ── 4. 流道數據校閱 (Streams) ──
    'S12.F.F': 'fur_F',
    'S12.P.P': 'fur_inputP',
    'S15.T.T': 'fur_outputT',
    'S16.F.F': 'WHB_F',
    'S16.P.P': 'WHB_inputP',
    'S16.T.T': 'WHB_inputT',
    'S13.T.T': 'WHB_outputT',
    'S13.P.P': 'WHB_outputP',
    'S36.F.F': 'HEATER1_F',
    'S36.P.P': 'HEATER1_input_P',
    'S36.T.T': 'HEATER1_input_T',
    'S21.F.F': 'cat1_F',
    'S21.P.P': 'cat1_input_P',
    'S21.T.T': 'cat1_input_temp',
    'S22.T.T': 'cat1_output_temp',
    'S25.F.F': 'HEATER2_F',
    'S25.P.P': 'HEATER2_input_P',
    'S25.T.T': 'HEATER2_input_T',
    'S27.F.F': 'cat2_F',
    'S27.P.P': 'cat2_input_P',
    'S27.T.T': 'cat2_input_temp',
    'S28.T.T': 'cat2_output_temp',
    'S14.F.F': 'SEP1_F',
    'S23.F.F': 'SEP2_F',
    'S29.F.F': 'SEP3_F',

    # ── 5. 組成與進料 (Composition) ──
    'ACIDGAS.T.T': 'acidgas_T',
    'ACIDGAS.P.P': 'acidgas_P',
    'ACIDGAS.Fcn.H2O.("H2O")': 'acidgas_H2O',
    'ACIDGAS.Fcn.H2S.("H2S")': 'acidgas_H2S',
    'ACIDGAS.Fcn.CO2.("CO2")': 'acidgas_CO2',
    'S33.Zn.SO2.("SO2")': 'B35_SO2',
    'S33.Zn.H2S.("H2S")': 'B35_H2S',
    'S8.P.P': 'burner_inputP'
}

# 建立反向字典 (Python 標籤 -> HYSYS 標籤)，用來從 Excel 抽資料
INV_MAPPING = {v: k for k, v in FULL_MAPPING.items()}

class SimpleTabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleTabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # =====================================================================
    # 2. 定義本次要用的 22 個變數 (20 de_mv + 2 targets)
    # 參考 variable_selection.py 71-var 的 de_mv 與目標
    # =====================================================================
    de_mv = [
        'air2_SP',
        'HEATER2_output_T_SP',
        'acidgas_Fm',
        'acidgas_P',
        'acidgas_T',
    ]
    target_cols = [
        'B35_H2S',
        'B35_SO2',
    ]

    # 取得對應到 Excel 的欄位名稱
    excel_input_cols = [INV_MAPPING[col] for col in de_mv]
    excel_target_cols = [INV_MAPPING[col] for col in target_cols]

    print("--- 載入穩態資料 ---")
    
    # 取得當下這份檔案的路徑相對於腳本或是工作目錄
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    steady_state_dir = os.path.join(base_dir, 'data', 'Claus_steady_state')
    
    # 自動尋找所有符合 LHS_generated 格式的檔案
    import glob
    file_pattern = os.path.join(steady_state_dir, 'lhs_generated_dynamic_ss_data*.xlsx')
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"在 {steady_state_dir} 中找不到任何名稱包含 lhs_generated_dynamic_ss_data 的 Excel 檔案！")

    df_list = []
    for full_data_path in all_files:
        print(f"Reading: {os.path.basename(full_data_path)}...")
        df_temp = pd.read_excel(full_data_path, sheet_name=0, header=2)
        # 第一行通常是單位，我們先將其去除
        df_temp = df_temp.iloc[1:].dropna(how='all').copy()
        
        if 'Status' in df_temp.columns:
            df_temp = df_temp[df_temp['Status'] == 'Run Completed']
            
        df_list.append(df_temp)

    # 將所有讀取出來的 DataFrame 合併
    df = pd.concat(df_list, ignore_index=True)
    print(f"總共從 {len(all_files)} 個檔案中載入 {len(df)} 筆有效資料")
    
    # 抽出需要的 X 與 Y
    required_cols = excel_input_cols + excel_target_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected workbook columns: {missing_cols}")

    df_x = df[excel_input_cols].copy()
    df_y = df[excel_target_cols].copy()

    # 確保資料格式轉換純淨 
    df_combined = pd.concat([df_x, df_y], axis=1).apply(pd.to_numeric, errors='coerce')
    
    # 檢查轉換後哪裡有 NaN，以利 debug
    null_counts = df_combined.isna().sum()
    if null_counts.sum() > 0:
        print("發現包含 NaN 的欄位:")
        print(null_counts[null_counts > 0])
        
    df_combined = df_combined.dropna()

    # 替換回 Python 好讀的名字
    rename_dict = dict(zip(excel_input_cols + excel_target_cols, de_mv + target_cols))
    df_combined = df_combined.rename(columns=rename_dict)

    # Aspen exports B17.PV.PV as kg/hr here, while the dynamic model uses kmol/hr.
    # Keep the tabular ANN target in the same physical unit as the dynamic data.
    AIR_MW = 28.0408  # kg/kmol
    if 'air' in df_combined.columns:
        df_combined['air'] = df_combined['air'] / AIR_MW
        print(f"[Unit] Converted air from kg/hr to kmol/hr using AIR_MW={AIR_MW}")

    # **將資料打亂並切割成 Train(80%), Valid(10%), Test(10%)**
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(df_combined)
    n_train = int(0.8 * n_total)
    n_valid = int(0.1 * n_total)

    df_train = df_combined.iloc[:n_train].copy()
    df_valid = df_combined.iloc[n_train:n_train+n_valid].copy()
    df_test = df_combined.iloc[n_train+n_valid:].copy()

    # 儲存獨立的 Test 測試集給預測程式使用
    out_dir = os.path.join(base_dir, 'results', 'Tabular_MLP_New')
    os.makedirs(out_dir, exist_ok=True)
    df_test.to_csv(os.path.join(out_dir, 'tabular_test_dataset.csv'), index=False)
    print(f"已儲存 {len(df_test)} 筆獨立 Test 資料至 tabular_test_dataset.csv")

    df_x_train = df_train[de_mv].astype('float32')
    df_y_train = df_train[target_cols].astype('float32')
    df_x_valid = df_valid[de_mv].astype('float32')
    df_y_valid = df_valid[target_cols].astype('float32')

    print(f"X features: {de_mv}")
    print(f"Y targets: {target_cols}")

    # =====================================================================
    # 3. 資料預處理 (標準化 Z-Score - 僅使用 Train 的統計量，防止資料洩漏)
    # =====================================================================
    x_mean = df_x_train.mean()
    x_std = df_x_train.std().replace(0, 1) # 避免除以 0
    x_train_scaled = (df_x_train - x_mean) / x_std
    x_valid_scaled = (df_x_valid - x_mean) / x_std

    y_mean = df_y_train.mean()
    y_std = df_y_train.std().replace(0, 1)
    y_train_scaled = (df_y_train - y_mean) / y_std
    y_valid_scaled = (df_y_valid - y_mean) / y_std

    # 儲存 Z-score 統計量供未來預測使用
    pd.concat([x_mean, y_mean]).to_csv(os.path.join(out_dir, 'zscore_mean.csv'))
    pd.concat([x_std, y_std]).to_csv(os.path.join(out_dir, 'zscore_std.csv'))
    
    # =====================================================================
    # 4. 準備 PyTorch Dataset & DataLoader
    # =====================================================================
    train_dataset = TensorDataset(torch.tensor(x_train_scaled.values), torch.tensor(y_train_scaled.values))
    valid_dataset = TensorDataset(torch.tensor(x_valid_scaled.values), torch.tensor(y_valid_scaled.values))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {n_train}, Valid samples: {n_valid}")

    # =====================================================================
    # 5. 初始化模型與訓練
    # =====================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTabularMLP(input_dim=len(de_mv), output_dim=len(target_cols)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    epochs = 1000
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    train_losses = []
    valid_losses = []

    print("\n--- 開始訓練 Tabular MLP ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        # 儲存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            model_save_dir = os.path.join(base_dir, 'saved_models')
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'Tabular_MLP_5in_2out_QV.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("--- 訓練完成 ---")
    print(f"最佳驗證 Loss: {best_loss:.4f}")

    # 繪製 Loss 曲線
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Tabular MLP Training Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'training_curve.png'))
    plt.close()
    print(f"Loss 曲線已儲存至 {out_dir}/training_curve.png")

if __name__ == '__main__':
    main()
