import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np

from src import data_utils
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
    import json
    stats_path = f'./saved_models/{exp}_preprocessing_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            # Load preprocessing stats as a dictionary
            preprocess_stats = json.load(f)
        print(f"[Info] Loaded preprocessing stats from {stats_path}")
    else:
        preprocess_stats = None
        print(f"[Warn] Preprocessing stats not found at {stats_path}, using default normalization")

    # 先決定輸入/輸出欄位
    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = data_utils.variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)
    all_cols = input_cols + target_cols

    # 讀訓練資料（只取關注欄位），用於計算標準化統計
    df_train_full = data_utils.load_data(os.path.join(cfg_data['path'], cfg_data['filename']))
    df_train = df_train_full.iloc[:cfg_data['point']][all_cols]
    
    # Apply air2_SP conversion to training data if needed
    # If training data mean and std weren't stored, we assume conversion is needed. That's how the line68 doing it.(Obtain mean and std)
    if preprocess_stats and preprocess_stats.get('air2_SP_converted') and 'air2_SP' in df_train.columns:
        print(f"[Info] Applying air2_SP conversion to training data: new = 17.228 * old - 0.09")
        df_train['air2_SP'] = 17.228 * df_train['air2_SP'] - 0.09
    
    # Apply Total_S scaling to training data if needed
    if preprocess_stats and preprocess_stats.get('Total_S_scaled') and 'Total_S' in df_train.columns:
        scale_factor = preprocess_stats.get('Total_S_scale_factor', 100.0)
        print(f"[Info] Applying Total_S scaling to training data (×{scale_factor})")
        df_train['Total_S'] = df_train['Total_S'] * scale_factor
    df_train = df_train.dropna(subset=all_cols)
    if len(df_train) == 0:
        null_counts = df_train_full[all_cols].isna().sum().to_dict()
        raise ValueError(f"Training data became empty after dropna on {all_cols}. NaN counts: {null_counts}")

    # Use stored stats if available, otherwise calculate from training data
    if preprocess_stats and 'mean' in preprocess_stats:
        print("[Info] Using stored preprocessing statistics")
        mean_all = pd.Series(preprocess_stats['mean'])
        std_all = pd.Series(preprocess_stats['std'])
    else:
        print("[Info] Calculating preprocessing statistics from training data")
        mean_all, std_all = data_utils.calculate_zscore_stats(df_train)

    # 讀測試資料（只取關注欄位）
    test_cfg = cfg_data['test_data']
    df_test_full = data_utils.load_data(os.path.join(cfg_data['path'], test_cfg['filename']))
    df_test = df_test_full.iloc[:test_cfg['point']][all_cols]
    
    # Apply air2_SP conversion to test data if needed
    if preprocess_stats and preprocess_stats.get('air2_SP_converted') and 'air2_SP' in df_test.columns:
        print(f"[Info] Applying air2_SP conversion to test data: new = 17.228 * old - 0.09")
        print(f"[Info] air2_SP range before: [{df_test['air2_SP'].min():.2f}, {df_test['air2_SP'].max():.2f}]")
        df_test['air2_SP'] = 17.228 * df_test['air2_SP'] - 0.09
        print(f"[Info] air2_SP range after: [{df_test['air2_SP'].min():.2f}, {df_test['air2_SP'].max():.2f}]")
    
    df_test = df_test.dropna(subset=all_cols)
    if len(df_test) == 0:
        null_counts = df_test_full[all_cols].isna().sum().to_dict()
        raise ValueError(f"Test data became empty after dropna on {all_cols}. NaN counts: {null_counts}")

    # 僅對關注欄位做標準化
    df_z_test = data_utils.apply_zscore(df_test, mean_all, std_all)

    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    model = TabularMLP(num_features=len(input_cols),
                       num_outputs=len(target_cols),
                       hidden_dims=config['model'].get('hidden_dims', [128, 64]),
                       dropout=config['model'].get('dropout', 0.1),
                       activation=config['model'].get('activation', 'relu'))

    model_path = f'./saved_models/{exp}_tabular_mlp.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    X = torch.tensor(df_z_test[input_cols].values.astype(np.float32), device=device)
    with torch.no_grad():
        y_pred_z = model(X).cpu().numpy()

    # 還原尺度（僅目標欄位）
    # All variables use their own mean and std from the stored stats
    y_mean = mean_all[target_cols].values
    y_std = std_all[target_cols].values
    y_pred = y_pred_z * y_std + y_mean
    y_true = df_test[target_cols].values
    
    # Convert Total_S back to original scale (÷100) if it was scaled during training
    if preprocess_stats and preprocess_stats.get('Total_S_scaled') and 'Total_S' in target_cols:
        scale_factor = preprocess_stats.get('Total_S_scale_factor', 100.0)
        print(f"[Info] Converting Total_S predictions back to original scale (÷{scale_factor})")
        total_s_idx = target_cols.index('Total_S')
        y_pred[:, total_s_idx] = y_pred[:, total_s_idx] / scale_factor

    # 存檔
    os.makedirs(f'./results/{exp}', exist_ok=True)
    out_csv = os.path.join('./results', exp, 'tabular_predictions.csv')
    pd.DataFrame(np.hstack([y_true, y_pred]),
                 columns=target_cols + [f'{c}_pred' for c in target_cols]).to_csv(out_csv, index=False)

    # 指標
    metrics = []
    for i, name in enumerate(target_cols):
        res = calculate_metrics(y_true[:, i], y_pred[:, i])
        res['Variable'] = name
        metrics.append(res)
        print(f"{name}: MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}, R2={res['R2']:.4f}, MAPE={res['MAPE']:.2f}%")

    pd.DataFrame(metrics)[['Variable','MAE','RMSE','R2','MAPE']].to_csv(
        os.path.join('./results', exp, 'tabular_metrics.csv'), index=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
