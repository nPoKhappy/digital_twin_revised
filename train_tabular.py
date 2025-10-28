import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from src import data_utils
from src.dataset_tabular import TabularDataset
from src.models.tabular_mlp import TabularMLP

from src.utils import calculate_metrics


def build_input_target_cols(en_mv_and_sv, y_sv):
    # 非時間序列：輸入 = 全部 encoder 欄位扣掉目標欄位
    input_cols = [c for c in en_mv_and_sv if c not in y_sv]
    return input_cols, list(y_sv)


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    exp = config['exp_name']

    # 讀訓練資料與標準化
    df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_raw = df_raw.iloc[:cfg_data['point']].dropna()
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
    df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)

    # 變量選擇或從設定覆寫
    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)

    # 切分（僅訓練/驗證，不再切測試）
    data_len = len(df_z)
    split_idx = int(data_len * (1 - cfg_data['valid_data_split']))
    train_df = df_z.iloc[:split_idx]
    val_df = df_z.iloc[split_idx:]

    # Dataset/DataLoader
    train_ds = TabularDataset(train_df, input_cols, target_cols)
    val_ds = TabularDataset(val_df, input_cols, target_cols)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=True)

    # 建模
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    model = TabularMLP(num_features=len(input_cols),
                       num_outputs=len(target_cols),
                       hidden_dims=config['model'].get('hidden_dims', [128, 64]),
                       dropout=config['model'].get('dropout', 0.1),
                       activation=config['model'].get('activation', 'relu')).to(device)

    # 支援可選的 weight_decay
    weight_decay = float(config['training'].get('weight_decay', 0.0))
    opt = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    criterion = nn.L1Loss()

    # 訓練
    best_val = float('inf')
    patience = config['training']['patience']
    pat = 0
    train_losses, val_losses = [], []

    for epoch in range(config['training']['epochs']):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                running += loss.item()
        val_loss = running / max(1, len(val_loader))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            pat = 0
            os.makedirs('./saved_models', exist_ok=True)
            torch.save(model.state_dict(), f'./saved_models/{exp}_tabular_mlp.pth')
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping")
                break

    # 保存損失
    os.makedirs(f'./results/{exp}', exist_ok=True)
    pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    }).to_csv(f'./results/{exp}/training_history_tabular.csv', index=False)

    return best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
