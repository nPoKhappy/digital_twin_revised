# train_tabular.py - Train a tabular MLP model on steady-state data (蝛拇??豢?)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from src import utils as data_utils
from src.dataset_tabular import TabularDataset
from src.models.tabular_mlp import TabularMLP

from src.utils import calculate_metrics


def build_input_target_cols(en_mv_and_sv, y_sv):
    # ??????頛詨 = ?券 encoder 甈?????格?甈?
    input_cols = [c for c in en_mv_and_sv if c not in y_sv]
    return input_cols, list(y_sv)


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    exp = config['exp_name']

    # Get columns configuration early
    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)
    
    all_needed_cols = input_cols + target_cols

    # 霈閮毀鞈???皞?
    df_raw = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_raw = df_raw.iloc[:cfg_data['point']]
    
    # Add feature engineering
    if 'air_acidgas_ratio' not in df_raw.columns and 'air2_SP_m3' in df_raw.columns and 'acidgas_Fv' in df_raw.columns:
        print("[Info] Adding feature 'air_acidgas_ratio' = 'air2_SP_m3' / 'acidgas_Fv'")
        df_raw['air_acidgas_ratio'] = df_raw['air2_SP_m3'] / df_raw['acidgas_Fv']

    # Selectively drop NA to avoid dropping rows due to missing columns we don't even use
    df_raw_len = len(df_raw)
    cols_to_check = [c for c in all_needed_cols if c in df_raw.columns]
    df_raw = df_raw.dropna(subset=cols_to_check)
    if len(df_raw) == 0:
        raise ValueError(f"num_samples should be a positive integer value, but got num_samples=0. All rows dropped due to NaNs in {cols_to_check}")
    print(f"[Info] Dropped {df_raw_len - len(df_raw)} rows with NaN values in used columns")

    
    # Scale Total_S (?1000) for better numerical stability
    # This helps the model learn small changes more effectively
    if 'Total_S' in df_raw.columns:
        print(f"[Info] Scaling Total_S (?1000)")
        print(f"[Info] Total_S range before: [{df_raw['Total_S'].min():.4f}, {df_raw['Total_S'].max():.4f}]")
        df_raw['Total_S'] = df_raw['Total_S'] * 1000
        print(f"[Info] Total_S range after: [{df_raw['Total_S'].min():.2f}, {df_raw['Total_S'].max():.2f}]")
    
    # Filter data to only numeric columns for z-score calculations and inputs/targets
    df_raw = df_raw.select_dtypes(include=['number'])
    
    # Calculate z-score stats AFTER air2_SP conversion and Total_S scaling
    # Each variable (inputs and targets) will have its own mean and std
    mean_all, std_all = data_utils.calculate_zscore_stats(df_raw)
    df_z = data_utils.apply_zscore(df_raw, mean_all, std_all)
    
    # Store all preprocessing stats for later use in prediction/evaluation
    zscore_dir = f'./results/{exp}/'
    os.makedirs(zscore_dir, exist_ok=True)
    mean_all.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
    std_all.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
    print(f"[Info] Saved preprocessing stats (zscore_mean.csv, zscore_std.csv) to {zscore_dir}")
    print(f"[Info] Stored mean/std for {len(mean_all)} variables (inputs + targets)")

    # 霈??豢???閮剖?閬神
    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)

    # ??嚗?閮毀/撽?嚗???皜祈岫嚗?
    data_len = len(df_z)
    split_idx = int(data_len * (1 - cfg_data['valid_data_split']))
    train_df = df_z.iloc[:split_idx]
    val_df = df_z.iloc[split_idx:]

    # Dataset/DataLoader
    train_ds = TabularDataset(train_df, input_cols, target_cols)
    val_ds = TabularDataset(val_df, input_cols, target_cols)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=True)

    # 撱箸芋
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'

    target_mean_tensor = torch.tensor(mean_all[target_cols].values, dtype=torch.float32)
    target_std_tensor = torch.tensor(std_all[target_cols].values, dtype=torch.float32)

    model = TabularMLP(num_features=len(input_cols),
                       num_outputs=len(target_cols),
                       hidden_dims=config['model'].get('hidden_dims', [128, 64]),
                       dropout=config['model'].get('dropout', 0.1),
                       activation=config['model'].get('activation', 'relu'),
                       target_mean=target_mean_tensor,
                       target_std=target_std_tensor).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"="*50)
    print(f"[Info] Tabular MLP Architecture Details:")
    print(f"       Input Features: {len(input_cols)}")
    print(f"       Output Targets: {len(target_cols)}")
    print(f"       Hidden Dims:    {config['model'].get('hidden_dims', [128, 64])}")
    print(f"       Total Trainable Parameters: {total_params:,}")
    print(f"="*50)

    # 支援可選的 weight_decay
    weight_decay = float(config['training'].get('weight_decay', 0.0))
    opt = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=weight_decay)
    criterion = nn.L1Loss()

    # 閮毀
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

    # ?脣?閮毀?鼓??
    os.makedirs(f'./results/{exp}', exist_ok=True)
    pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    }).to_csv(f'./results/{exp}/training_history_tabular.csv', index=False)
    
    # 蝜芾ˊ Loss ?脩???
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='', linewidth=2)
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='', linewidth=2)
    plt.title(f'Training and Validation Loss Curve - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/{exp}/loss_curve.png', dpi=300)
    plt.close()
    
    print(f"[Info] Loss curve successfully saved to: ./results/{exp}/loss_curve.png")

    return best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)





