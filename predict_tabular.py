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
    df_train = df_train.dropna(subset=all_cols)
    if len(df_train) == 0:
        null_counts = df_train_full[all_cols].isna().sum().to_dict()
        raise ValueError(f"Training data became empty after dropna on {all_cols}. NaN counts: {null_counts}")

    mean_all, std_all = data_utils.calculate_zscore_stats(df_train)

    # 讀測試資料（只取關注欄位）
    test_cfg = cfg_data['test_data']
    df_test_full = data_utils.load_data(os.path.join(cfg_data['path'], test_cfg['filename']))
    df_test = df_test_full.iloc[:test_cfg['point']][all_cols]
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
    y_mean = mean_all[target_cols].values
    y_std = std_all[target_cols].values
    y_pred = y_pred_z * y_std + y_mean
    y_true = df_test[target_cols].values

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
