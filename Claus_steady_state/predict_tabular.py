# predict_tabular.py - Predict tabular data using a trained MLP model (蝛拇??豢?)
import os
import sys

# Add the project root to sys.path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import utils as data_utils
from src.models.tabular_mlp import TabularMLP
from src.utils import calculate_metrics


def build_input_target_cols(en_mv_and_sv, y_sv):
    input_cols = [c for c in en_mv_and_sv if c not in y_sv]
    return input_cols, list(y_sv)


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    exp = config['exp_name']

    # Load preprocessing stats saved during training
    mean_path = f'./results/{exp}/zscore_mean.csv'
    std_path = f'./results/{exp}/zscore_std.csv'
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean_all = pd.read_csv(mean_path, index_col=0).squeeze("columns")
        std_all = pd.read_csv(std_path, index_col=0).squeeze("columns")
        print(f"[Info] Loaded preprocessing stats from {mean_path} and {std_path}")
        preprocess_stats = True
    else:
        preprocess_stats = False
        print(f"[Warn] Preprocessing stats not found at {mean_path} or {std_path}, using default normalization")

    # ?捱摰撓??頛詨甈?
    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)
    all_cols = input_cols + target_cols

    # 霈閮毀鞈?嚗??瘜冽?雿?嚗?潸?蝞?皞?蝯梯?
    df_train_full = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    
    if 'air_acidgas_ratio' not in df_train_full.columns and 'air2_SP_m3' in df_train_full.columns and 'acidgas_Fv' in df_train_full.columns:
        print("[Info] Adding feature 'air_acidgas_ratio' = 'air2_SP_m3' / 'acidgas_Fv' for train")
        df_train_full['air_acidgas_ratio'] = df_train_full['air2_SP_m3'] / df_train_full['acidgas_Fv']

    df_train = df_train_full.iloc[:cfg_data['point']][all_cols]
    
    df_train = df_train.dropna(subset=all_cols)
    if len(df_train) == 0:
        null_counts = df_train_full[all_cols].isna().sum().to_dict()
        raise ValueError(f"Training data became empty after dropna on {all_cols}. NaN counts: {null_counts}")

    # Use stored stats if available, otherwise calculate from training data
    if preprocess_stats:
        print("[Info] Using stored preprocessing statistics")
    else:
        print("[Info] Calculating preprocessing statistics from training data")
        mean_all, std_all = data_utils.calculate_zscore_stats(df_train)

    # 霈皜祈岫鞈?嚗??瘜冽?雿?
    test_cfg = cfg_data['test_data']
    df_test_full = data_utils.load_data(os.path.join(cfg_data['path'], test_cfg['filename']))
    
    if 'air_acidgas_ratio' not in df_test_full.columns and 'air2_SP_m3' in df_test_full.columns and 'acidgas_Fv' in df_test_full.columns:
        print("[Info] Adding feature 'air_acidgas_ratio' = 'air2_SP_m3' / 'acidgas_Fv' for test")
        df_test_full['air_acidgas_ratio'] = df_test_full['air2_SP_m3'] / df_test_full['acidgas_Fv']

    df_test = df_test_full.iloc[:test_cfg['point']][all_cols]
    
    df_test = df_test.dropna(subset=all_cols)
    if len(df_test) == 0:
        null_counts = df_test_full[all_cols].isna().sum().to_dict()
        raise ValueError(f"Test data became empty after dropna on {all_cols}. NaN counts: {null_counts}")

    # ???釣甈???皞?
    df_z_test = data_utils.apply_zscore(df_test, mean_all, std_all)

    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'

    target_mean_tensor = torch.tensor(mean_all[target_cols].values, dtype=torch.float32)
    target_std_tensor = torch.tensor(std_all[target_cols].values, dtype=torch.float32)

    model = TabularMLP(num_features=len(input_cols),
                       num_outputs=len(target_cols),
                       hidden_dims=config['model'].get('hidden_dims', [128, 64]),
                       dropout=config['model'].get('dropout', 0.1),
                       activation=config['model'].get('activation', 'relu'),
                       target_mean=target_mean_tensor,
                       target_std=target_std_tensor)
    model_path = f'./saved_models/{exp}_tabular_mlp.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X = torch.tensor(df_z_test[input_cols].values.astype(np.float32), device=device)
    with torch.no_grad():
        y_pred_z = model(X).cpu().numpy()

    # ??撠箏漲嚗??格?甈?嚗?
    # All variables use their own mean and std from the stored stats
    y_mean = mean_all[target_cols].values
    y_std = std_all[target_cols].values
    y_pred = y_pred_z * y_std + y_mean
    y_true = df_test[target_cols].values
    

    # 摮?
    os.makedirs(f'./results/{exp}', exist_ok=True)
    out_csv = os.path.join('./results', exp, 'tabular_predictions.csv')
    pd.DataFrame(np.hstack([y_true, y_pred]),
                 columns=target_cols + [f'{c}_pred' for c in target_cols]).to_csv(out_csv, index=False)

    # ??
    metrics = []
    for i, name in enumerate(target_cols):
        y_true_var = y_true[:, i]
        y_pred_var = y_pred[:, i]
        
        res = calculate_metrics(y_true_var, y_pred_var)
        res['Variable'] = name
        metrics.append(res)
        print(f"{name}: MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}, R2={res['R2']:.4f}, MAPE={res['MAPE']:.2f}%")    
        
        # Parity Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_var, y_pred_var, color='blue', alpha=0.5, s=15, edgecolors='none')
        min_val = min(y_true_var.min(), y_pred_var.min())
        max_val = max(y_true_var.max(), y_pred_var.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'{name} Parity\nRMSE={res["RMSE"]:.4f}, R2={res["R2"]:.4f}')
        plt.grid(True, color='lightgray', linestyle='-', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join('./results', exp, f'parity_plot_{name}.png'), dpi=300)
        plt.close()

    pd.DataFrame(metrics)[['Variable','MAE','RMSE','R2','MAPE']].to_csv(
        os.path.join('./results', exp, 'tabular_metrics.csv'), index=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)


