import os
import sys
import argparse
import yaml
import glob
import torch
import torch.nn as nn
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import contextlib
import io
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model

# =====================================================================
# 1. Variable name mapping
#    HYSYS/raw tag names -> Python-friendly column names
# =====================================================================
FULL_MAPPING = {
    'B34.SPo.SPo': 'acidgas_Fm', 'B17.PV.PV': 'air', 'S20.P.P': 'HEATER1_output_P', 'B33.SPo.SPo': 'air2_SP',
    'B17.SPo.SPo': 'air_SP', 'B35.SPo.SPo': 'COG_SP', 'AIR2.Fv.Fv': 'second_air2', 'S4.Fv.Fv': 'COG',
    'B18.SPo.SPo': 'burner_input_T_SP', 'B18.PV.PV': 'burner_input_T_PV', 'B19.SPo.SPo': 'burner_output_T_SP',
    'B19.PV.PV': 'burner_output_T_PV', 'BURNER_PC.SPo.SPo': 'burner_output_P_SP', 'BURNER_PC.PV.PV': 'burner_output_P_PV',
    'FURANCE_PC.SPo.SPo': 'fur_outputP_SP', 'FURANCE_PC.PV.PV': 'fur_outputP_PV', 'FURANCE.T.0.(0)': 'fur_inputT',
    'FURANCE.T.1.(1)': 'fur_temp', 'SEP1_PC.SPo.SPo': 'SEP1_P_SP', 'SEP1_PC.PV.PV': 'SEP1_P_PV', 'SEP1.T.T': 'SEP1_T',
    'SEP2_PC.SPo.SPo': 'SEP2_P_SP', 'SEP2_PC.PV.PV': 'SEP2_P_PV', 'SEP2.T.T': 'SEP2_T', 'SEP3_PC.SPo.SPo': 'SEP3_P_SP',
    'SEP3_PC.PV.PV': 'SEP3_P_PV', 'SEP3.T.T': 'SEP3_T', 'B21.SPo.SPo': 'HEATER1_output_T_SP',
    'B21.PV.PV': 'HEATER1_output_T_PV', 'B20.SPo.SPo': 'HEATER2_output_T_SP', 'B20.PV.PV': 'HEATER2_output_T_PV',
    'CAT1_PC.SPo.SPo': 'cat1_output_P_SP', 'CAT1_PC.PV.PV': 'cat1_output_P_PV', 'CAT2_PC.SPo.SPo': 'cat2_output_P_SP',
    'CAT2_PC.PV.PV': 'cat2_output_P_PV', 'S12.F.F': 'fur_F', 'S12.P.P': 'fur_inputP', 'S15.T.T': 'fur_outputT',
    'S16.F.F': 'WHB_F', 'S16.P.P': 'WHB_inputP', 'S16.T.T': 'WHB_inputT', 'S13.T.T': 'WHB_outputT',
    'S13.P.P': 'WHB_outputP', 'S36.F.F': 'HEATER1_F', 'S36.P.P': 'HEATER1_input_P', 'S36.T.T': 'HEATER1_input_T',
    'S21.F.F': 'cat1_F', 'S21.P.P': 'cat1_input_P', 'S21.T.T': 'cat1_input_temp', 'S22.T.T': 'cat1_output_temp',
    'S25.F.F': 'HEATER2_F', 'S25.P.P': 'HEATER2_input_P', 'S25.T.T': 'HEATER2_input_T', 'S27.F.F': 'cat2_F',
    'S27.P.P': 'cat2_input_P', 'S27.T.T': 'cat2_input_temp', 'S28.T.T': 'cat2_output_temp', 'S14.F.F': 'SEP1_F',
    'S23.F.F': 'SEP2_F', 'S29.F.F': 'SEP3_F', 'ACIDGAS.T.T': 'acidgas_T', 'ACIDGAS.P.P': 'acidgas_P',
    'ACIDGAS.Fcn.H2O.("H2O")': 'acidgas_H2O', 'ACIDGAS.Fcn.H2S.("H2S")': 'acidgas_H2S',
    'ACIDGAS.Fcn.CO2.("CO2")': 'acidgas_CO2', 'S33.Zn.SO2.("SO2")': 'B35_SO2', 'S33.Zn.H2S.("H2S")': 'B35_H2S',
    'S8.P.P': 'burner_inputP'
}

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

def load_steady_state_source(path, required_cols, log_target_cols=None):
    paths = sorted(glob.glob(path))
    if not paths and os.path.exists(path):
        paths = [path]
    if not paths:
        raise FileNotFoundError(f"Steady-state source file not found: {path}")

    df_list = []
    total_before_drop = 0
    for source_path in paths:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in [".xlsx", ".xlsm", ".xls"]:
            df = pd.read_excel(source_path, sheet_name=0, header=2)
            df = df.iloc[1:].dropna(how='all').copy()
        elif ext == ".csv":
            df = pd.read_csv(source_path)
        else:
            raise ValueError(f"Unsupported steady-state source file type: {ext}")

        if 'Status' in df.columns:
            df = df[df['Status'] == 'Run Completed'].copy()

        rename_map = {raw_name: py_name for raw_name, py_name in FULL_MAPPING.items() if raw_name in df.columns}
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            missing_preview = ', '.join(missing_cols[:20])
            extra = f" ... (+{len(missing_cols) - 20} more)" if len(missing_cols) > 20 else ""
            raise ValueError(f"Missing required steady-state columns in {source_path}: {missing_preview}{extra}")

        df = df[required_cols].apply(pd.to_numeric, errors='coerce')
        total_before_drop += len(df)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No usable steady-state rows after numeric conversion/dropna: {path}")

    for col in log_target_cols or []:
        if col in df.columns:
            df[col] = np.log(np.clip(df[col], 1e-6, None))

    print(
        f"[Info] Loaded steady-state sources for Gain/ANN teacher: {len(paths)} file(s) | "
        f"{len(df)} usable rows ({total_before_drop - len(df)} dropped)"
    )
    for source_path in paths:
        print(f"  - {source_path}")
    return df

def generate_steady_state_batch(df_raw, batch_size, de_mv, all_cols, W, std_all, target_mvs=None, repeat_rows_as_history=False, indices=None):
    if indices is not None:
        idx1 = np.asarray(indices)
    elif repeat_rows_as_history:
        idx1 = np.random.choice(len(df_raw), batch_size)
    else:
        idx1 = np.random.choice(len(df_raw) - W - 1, batch_size)

    historical_dfs = []
    ss1_rows = []

    for i in idx1:
        if repeat_rows_as_history:
            ss_row = df_raw.iloc[[i]].reset_index(drop=True)
            hist_w = pd.concat([ss_row] * W, ignore_index=True)
        else:
            hist_w = df_raw.iloc[i : i+W].reset_index(drop=True)
        historical_dfs.append(hist_w)
        ss1_rows.append(hist_w.iloc[-1:])

    ss1_p_df = pd.concat(ss1_rows, ignore_index=True)
    return historical_dfs, ss1_p_df

def dataframe_to_z_tensor(df, cols, mean, std, device):
    mean_vals = mean[cols].values
    std_vals = std[cols].replace(0, 1).values
    z_values = (df[cols].values - mean_vals) / std_vals
    return torch.tensor(z_values, dtype=torch.float32, device=device)

def step_wise_rolling_loss_and_predictions(model, batch, criterion, device):
    """Mirror src.engine.step_wise_rolling_training_step and keep predictions for D1 loss."""
    en_input_initial, de_inputs, targets = batch
    current_en_input = en_input_initial.clone().to(device)
    all_future_mvs = de_inputs.to(device)
    all_future_targets = targets.to(device)

    n_steps = all_future_mvs.shape[1]
    total_loss = 0
    predictions = []

    for t in range(n_steps):
        _, context = model.encoder(current_en_input)
        single_step_de_input = all_future_mvs[:, t, :].unsqueeze(1)
        single_step_prediction = model.decoder(single_step_de_input, context)
        single_step_target = all_future_targets[:, t, :].unsqueeze(1)

        total_loss += criterion(single_step_prediction, single_step_target)
        predictions.append(single_step_prediction)

        if t < n_steps - 1:
            next_en_input_history = current_en_input[:, 1:, :]
            new_step_features = torch.cat(
                [single_step_de_input, single_step_prediction.detach()],
                dim=2
            )
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)

    return total_loss / n_steps, torch.cat(predictions, dim=1), all_future_targets

def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp = config['exp_name']
    pretrained_path = config['training'].get(
        'pretrained_path',
        './saved_models/transformer_layerwise_71var_decoder_input_sp.pth'
    )
    use_pretrained = config['training'].get('use_pretrained', False)
    finetune_lr = config['training'].get('finetune_learning_rate', 1e-5)
    if use_pretrained:
        exp = config['training'].get('finetune_exp_name', exp + "_Finetuned")
        print(f"[Info] Pre-trained finetuning enabled. Experiment name: {exp}")

    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'

    print("=" * 70)
    print(f"Physics-Informed Gain Training (From Scratch) - Session: {exp}")
    print("=" * 70)

    cfg_data = config['data']
    de_mv, y_sv, non_used, en_mv_and_sv = variable_selection(cfg_data['variables_num'])

    all_dynamic_cols = []
    for col in en_mv_and_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)
    for col in y_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)

    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)

    tab_input_cols = [
        'air2_SP',
        'HEATER2_output_T_SP',
        'acidgas_Fm',
        'acidgas_P',
        'acidgas_T',
    ]
    full_model_target_cols = [
        'B35_H2S',
        'B35_SO2',
    ]
    gain_target_qv = config['training'].get('gain_target_qv', ["B35_H2S", "B35_SO2"])
    gain_target_mv = config['training'].get('gain_target_mv', ['air2_SP', 'HEATER2_output_T_SP'])
    tab_target_cols = [c for c in gain_target_qv if c in y_sv and c in full_model_target_cols]
    gain_target_mv = [c for c in gain_target_mv if c in de_mv and c in tab_input_cols]
    if len(tab_target_cols) == 0:
        raise ValueError("No valid gain_target_qv columns found in both Transformer outputs and ANN outputs.")
    if len(gain_target_mv) == 0:
        raise ValueError("No valid gain_target_mv columns found in both Transformer decoder inputs and ANN inputs.")
    gain_target_mv_indices = [de_mv.index(c) for c in gain_target_mv]

    print("[Info] Loading historical R5-X dataset and calculating Z-score Stats...")
    from src.dataset import MultiStepS2SDataset
    from torch.utils.data import DataLoader, ConcatDataset
    interval = config['window']['sampling_interval_min']
    use_median = config['window'].get('use_median_downsampling', True)

    all_dfs_log = []
    if 'training_files' in cfg_data and cfg_data['training_files']:
        for fname in cfg_data['training_files']:
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                df_seg = pd.read_csv(fpath)
                if interval > 1:
                    if use_median:
                        df_seg = df_seg.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                        df_seg = df_seg.iloc[interval-1::interval].reset_index(drop=True)
                    else:
                        df_seg = df_seg.iloc[::interval].reset_index(drop=True)
                df_seg.dropna(inplace=True)
                target_cols_log = ['B35_H2S', 'B35_SO2']
                df_seg = data_utils.apply_log_transform(df_seg, target_cols_log)
                all_dfs_log.append(df_seg)
            else:
                print(f"  [Warning] Training file not found, skipped: {fpath}")

    if len(all_dfs_log) == 0:
        raise ValueError("No training data found to calculate Z-score stats.")

    df_all_log = pd.concat(all_dfs_log, ignore_index=True)
    mean_all, std_all = data_utils.calculate_zscore_stats(df_all_log)

    # Save the z-score statistics used by this PGIN training run.
    zscore_dir = f'./results/{exp}/'
    os.makedirs(zscore_dir, exist_ok=True)
    mean_all.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
    std_all.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
    print(f"  [Save] Z-score stats saved to {zscore_dir}")

    tab_mean_path = './results/Tabular_MLP_New/zscore_mean.csv'
    tab_std_path = './results/Tabular_MLP_New/zscore_std.csv'
    if not os.path.exists(tab_mean_path) or not os.path.exists(tab_std_path):
        raise FileNotFoundError(
            "Tabular MLP z-score stats are required for ANN steady-state targets: "
            f"{tab_mean_path}, {tab_std_path}"
        )

    tab_mean = pd.read_csv(tab_mean_path, index_col=0).squeeze("columns")
    tab_std = pd.read_csv(tab_std_path, index_col=0).squeeze("columns").replace(0, 1)
    missing_tab_stats = [c for c in tab_input_cols + full_model_target_cols if c not in tab_mean.index or c not in tab_std.index]
    if missing_tab_stats:
        missing_preview = ', '.join(missing_tab_stats[:20])
        extra = f" ... (+{len(missing_tab_stats) - 20} more)" if len(missing_tab_stats) > 20 else ""
        raise ValueError(f"Missing Tabular MLP z-score stats for: {missing_preview}{extra}")

    for col in tab_input_cols + full_model_target_cols:
        if col not in mean_all.index and col in tab_mean.index:
            mean_all[col] = tab_mean[col]
            std_all[col] = tab_std[col]

    all_dfs_z = []
    for df_log in all_dfs_log:
        df_z = data_utils.apply_zscore(df_log, mean_all, std_all)
        all_dfs_z.append(df_z)

    train_datasets = []
    valid_datasets = []
    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    H_out = config['window']['prediction_length']
    regression_steps = int(config['training'].get('stepwise_regression_steps', 36))
    if regression_steps < 1:
        raise ValueError("stepwise_regression_steps must be >= 1.")
    dataset_H = regression_steps
    print(f"[Info] Step-wise regression training length: {dataset_H} steps")

    for df_z in all_dfs_z:
        split_point1 = int(len(df_z) * (1 - cfg_data['test_data_split']))
        split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))

        train_df = df_z.iloc[:split_point2]
        if len(train_df) > W + dataset_H:
            train_ds = MultiStepS2SDataset(train_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            train_datasets.append(train_ds)

        valid_df = df_z.iloc[split_point2:split_point1]
        if len(valid_df) > W + dataset_H:
            valid_ds = MultiStepS2SDataset(valid_df, en_mv_and_sv, de_mv, y_sv, W, dataset_H)
            valid_datasets.append(valid_ds)

    train_ds = ConcatDataset(train_datasets)
    batch_size = config['training'].get('batch_size', 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    print(f'[Info] Loaded Dynamic Train Loader: {len(train_loader)} batches')

    if len(valid_datasets) > 0:
        valid_ds = ConcatDataset(valid_datasets)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        print(f'[Info] Loaded Dynamic Valid Loader: {len(valid_loader)} batches')
    else:
        valid_loader = None
        print('[Warning] No Validation data available!')

    print('[Info] Preparing LHS steady-state data for Gain loss...')
    keep_cols = []
    for c in all_dynamic_cols + tab_input_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    steady_state_source_path = config['training'].get(
        'steady_state_source_path',
        './data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx'
    )
    df_raw = load_steady_state_source(
        steady_state_source_path,
        keep_cols,
        log_target_cols=['B35_H2S', 'B35_SO2']
    )

    print("[Info] Initializing Transformer Model...")
    dynamic_model = get_model(config).to(device)
    if use_pretrained:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Configured pretrained_path does not exist: {pretrained_path}")
        print(f"  [Info] Loading pre-trained MSE weights: {pretrained_path}")
        print(f"  [Info] Loading weights for Finetuning...")
        dynamic_model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("  [Info] Pre-trained loading disabled. Training entirely from scratch.")
    dynamic_model.train()

    print("[Info] Loading Pre-trained TabularMLP Model...")
    tab_target_mean_tensor = torch.tensor(tab_mean[full_model_target_cols].values, dtype=torch.float32, device=device)
    tab_target_std_tensor = torch.tensor(tab_std[full_model_target_cols].values, dtype=torch.float32, device=device)
    dynamic_full_target_mean_tensor = torch.tensor(mean_all[full_model_target_cols].values, dtype=torch.float32, device=device)
    dynamic_full_target_std_tensor = torch.tensor(std_all[full_model_target_cols].values, dtype=torch.float32, device=device)
    mlp_model = SimpleTabularMLP(input_dim=len(tab_input_cols), output_dim=len(full_model_target_cols))
    mlp_path = f'./saved_models/Tabular_MLP_5in_2out_QV.pth'
    if not os.path.exists(mlp_path):
        mlp_path = f'./saved_models/{exp}_tabular_mlp.pth'
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.to(device)
    mlp_model.eval()
    mlp_model.requires_grad_(False)

    epochs = config['training'].get('epochs', 20)
    steps_per_epoch = min(config['training'].get('steps_per_epoch', 50), len(train_loader))
    base_lr = config['training'].get('learning_rate', 1e-4)
    if use_pretrained:
        base_lr = finetune_lr
        print(f"  [Info] Using fine-tuning Learning Rate: {base_lr}")

    optimizer = optim.Adam(dynamic_model.parameters(), lr=base_lr)

    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    gain_warmup_steps = int(config['training'].get('stepwise_gain_warmup_steps', 60))
    H_gain = int(config['training'].get('stepwise_gain_steps', gain_warmup_steps + H_out))
    if H_gain < 1:
        raise ValueError("stepwise_gain_steps must be >= 1.")
    if gain_warmup_steps < 0 or gain_warmup_steps >= H_gain:
        raise ValueError("stepwise_gain_warmup_steps must be >= 0 and less than stepwise_gain_steps.")
    dynamic_gain_step_change_step = int(
        config['training'].get('dynamic_gain_step_change_step', gain_warmup_steps + 1)
    )
    if dynamic_gain_step_change_step < 1 or dynamic_gain_step_change_step > H_gain:
        raise ValueError("dynamic_gain_step_change_step must be 1-based and within stepwise_gain_steps.")
    dynamic_gain_step_change_idx = dynamic_gain_step_change_step - 1
    pgin_runtime_plot = config['training'].get('pgin_runtime_plot', True)

    best_loss = float('inf')
    early_stop_patience = config['training'].get('patience', 10)
    epochs_no_improve = 0

    y_mean_tensor = torch.tensor(mean_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(std_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_safe = torch.where(torch.abs(y_std_tensor) < 1e-6, torch.ones_like(y_std_tensor), y_std_tensor)

    tab_target_std_safe = torch.where(torch.abs(tab_target_std_tensor) < 1e-6, torch.ones_like(tab_target_std_tensor), tab_target_std_tensor)
    dynamic_full_target_std_safe = torch.where(
        torch.abs(dynamic_full_target_std_tensor) < 1e-6,
        torch.ones_like(dynamic_full_target_std_tensor),
        dynamic_full_target_std_tensor
    )
    ss_target_cols_cfg = config['training'].get('ss_target_cols')
    if ss_target_cols_cfg:
        ss_target_cols = [c for c in ss_target_cols_cfg if c in full_model_target_cols and c in y_sv]
        missing_ss_targets = [c for c in ss_target_cols_cfg if c not in ss_target_cols]
        if missing_ss_targets:
            raise ValueError(f"Invalid ss_target_cols entries: {missing_ss_targets}")
    else:
        ss_target_cols = [c for c in full_model_target_cols if c in y_sv]
    if len(ss_target_cols) == 0:
        raise ValueError("No valid steady-state target columns configured.")
    ann_monitor_idx = [full_model_target_cols.index(c) for c in tab_target_cols]

    gain_loss_weight = config['training'].get('gain_loss_weight', config['training'].get('pgin_loss_weight', 0.3))
    smooth_loss_weight = config['training'].get('smooth_loss_weight', 0.0)
    enable_gain_loss = config['training'].get('enable_gain_loss', True)
    monitor_gain_kci = config['training'].get('monitor_gain_kci', True)
    effective_gain_loss_weight = gain_loss_weight if enable_gain_loss else 0.0
    compute_gain_metrics = enable_gain_loss or monitor_gain_kci or config['training'].get('pgin_runtime_plot', True)
    gain_valid_delta_threshold = config['training'].get('gain_valid_delta_threshold', 1e-5)
    dynamic_gain_method = config['training'].get('dynamic_gain_method', 'autograd').lower()
    dynamic_gain_tail_start_step = config['training'].get(
        'stepwise_gain_tail_start_step',
        max(config['training'].get('dynamic_gain_tail_start_step', dynamic_gain_step_change_step), dynamic_gain_step_change_step)
    )
    finite_diff_delta_std = config['training'].get('finite_diff_delta_std', 0.5)
    ss_batch_size = config['training'].get('ss_batch_size', 4)
    ss_coverage_mode = config['training'].get('ss_coverage_mode', 'random_with_replacement').lower()
    if dynamic_gain_method not in ['autograd', 'finite_difference']:
        raise ValueError("dynamic_gain_method must be either 'autograd' or 'finite_difference'.")
    if ss_coverage_mode not in ['random_with_replacement', 'shuffle_without_replacement']:
        raise ValueError("ss_coverage_mode must be either 'random_with_replacement' or 'shuffle_without_replacement'.")
    if ss_batch_size < 1:
        raise ValueError("ss_batch_size must be >= 1.")
    if ss_coverage_mode == 'shuffle_without_replacement' and ss_batch_size > len(df_raw):
        raise ValueError("ss_batch_size cannot exceed steady-state row count when using shuffle_without_replacement.")
    if dynamic_gain_tail_start_step < 1:
        raise ValueError("dynamic_gain_tail_start_step must be 1-based and >= 1.")
    dynamic_gain_tail_start_idx = min(max(dynamic_gain_tail_start_step, dynamic_gain_step_change_step) - 1, H_gain - 1)
    dynamic_gain_tail_end_idx = H_gain
    ss_perm = np.random.permutation(len(df_raw))
    ss_cursor = 0

    def next_steady_state_indices(batch_size):
        nonlocal ss_perm, ss_cursor
        if ss_coverage_mode != 'shuffle_without_replacement':
            return None
        if ss_cursor + batch_size > len(ss_perm):
            ss_perm = np.random.permutation(len(df_raw))
            ss_cursor = 0
        batch_indices = ss_perm[ss_cursor:ss_cursor + batch_size]
        ss_cursor += batch_size
        return batch_indices

    print(
        f"  [Info] Loss weights | MSE: 1.0, Smooth: {smooth_loss_weight}, "
        f"Gain: {effective_gain_loss_weight}; "
        f"Plot targets: {len(ss_target_cols)} vars; Gain loss enabled: {enable_gain_loss}; "
        f"KCI monitor: {compute_gain_metrics}; "
        f"Gain targets: {tab_target_cols} x {gain_target_mv}; "
        f"Gain valid delta threshold: {gain_valid_delta_threshold}; "
        f"Dynamic gain method: {dynamic_gain_method}; "
        f"Dynamic gain rollout: {H_gain} step-wise steps; "
        f"Step change starts at step {dynamic_gain_step_change_step}; "
        f"Dynamic gain output: steps {dynamic_gain_tail_start_idx + 1}-{dynamic_gain_tail_end_idx}; "
        f"FD delta std: {finite_diff_delta_std}; "
        f"SS batch: {ss_batch_size}; SS coverage: {ss_coverage_mode}"
    )

    history_losses = []
    history_val_losses = []
    history_kci = []
    history_gain_pair_rows = []

    from src.engine import step_wise_rolling_training_step
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        dynamic_model.train()
        epoch_gain_loss = 0.0
        epoch_correct_dir = 0
        epoch_total_eval = 0
        epoch_mse_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_total_loss = 0.0
        pair_correct = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_total = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_ann_sign_sum = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        pair_dyn_pos_sum = torch.zeros(len(tab_target_cols), len(gain_target_mv), device=device)
        ann_monitor_min = torch.full((len(tab_target_cols),), float('inf'), device=device)
        ann_monitor_max = torch.full((len(tab_target_cols),), float('-inf'), device=device)
        ann_monitor_clamped = 0
        ann_monitor_count = 0

        step_limit = min(steps_per_epoch, len(train_loader))
        pbar = tqdm(enumerate(train_loader), total=step_limit, desc=f"Epoch {epoch+1}/{epochs}")

        for step, mse_batch in pbar:
            if step >= step_limit:
                break

            optimizer.zero_grad()

            if smooth_loss_weight != 0:
                mse_loss_val, pred_rollout_z, target_rollout_z = step_wise_rolling_loss_and_predictions(
                    dynamic_model, mse_batch, criterion, device
                )
                if pred_rollout_z.shape[1] > 1:
                    pred_diff_z = pred_rollout_z[:, 1:, :] - pred_rollout_z[:, :-1, :]
                    target_diff_z = target_rollout_z[:, 1:, :] - target_rollout_z[:, :-1, :]
                    smooth_loss_val = criterion(pred_diff_z, target_diff_z)
                else:
                    smooth_loss_val = torch.tensor(0.0, device=device)
            else:
                mse_loss_val = step_wise_rolling_training_step(dynamic_model, mse_batch, criterion, device)
                smooth_loss_val = torch.tensor(0.0, device=device)

            ss_indices = next_steady_state_indices(ss_batch_size)
            historical_dfs, ss1_df = generate_steady_state_batch(
                df_raw, ss_batch_size, de_mv, keep_cols, W, std_all, gain_target_mv,
                repeat_rows_as_history=True, indices=ss_indices
            )
            B_actual = len(historical_dfs)

            with contextlib.redirect_stdout(io.StringIO()):
                x_en_z_list = []
                for b_df in historical_dfs:
                    b_z = data_utils.apply_zscore(b_df, mean_all, std_all).fillna(0.0)
                    x_en_z_list.append(torch.tensor(b_z[en_mv_and_sv].values, dtype=torch.float32, device=device))
                x_en_z_history = torch.stack(x_en_z_list)

                ss1_z_df = data_utils.apply_zscore(ss1_df, mean_all, std_all).fillna(0.0)

            ss1_de_p = torch.tensor(ss1_df[de_mv].values, dtype=torch.float32, device=device)

            mlp_x_z_ss1 = dataframe_to_z_tensor(ss1_df, tab_input_cols, tab_mean, tab_std, device)
            mlp_x_z_ss1.requires_grad_(True)

            y_mlp_z_ss1 = mlp_model(mlp_x_z_ss1)
            y_mlp_p_ss1 = y_mlp_z_ss1 * tab_target_std_safe + tab_target_mean_tensor
            ann_monitor_vals = y_mlp_p_ss1.detach()[:, ann_monitor_idx]
            ann_monitor_min = torch.minimum(ann_monitor_min, ann_monitor_vals.min(dim=0).values)
            ann_monitor_max = torch.maximum(ann_monitor_max, ann_monitor_vals.max(dim=0).values)
            ann_monitor_clamped += (ann_monitor_vals <= 1e-6).sum().item()
            ann_monitor_count += ann_monitor_vals.numel()

            ss_target_z = torch.tensor(ss1_z_df[ss_target_cols].values, dtype=torch.float32, device=device)

            loss_gain = torch.tensor(0.0, device=device)
            gain_prev_training_mode = dynamic_model.training
            if compute_gain_metrics:
                dynamic_model.eval()
                K_ss_matrix = torch.zeros(B_actual, len(tab_target_cols), len(gain_target_mv), device=device)
                for t_idx, tgt_col in enumerate(tab_target_cols):
                    if tgt_col in full_model_target_cols:
                        full_out_idx = full_model_target_cols.index(tgt_col)
                        grad_outputs = torch.zeros_like(y_mlp_p_ss1)
                        grad_outputs[:, full_out_idx] = 1.0

                        grads_x, = torch.autograd.grad(
                            outputs=y_mlp_p_ss1,
                            inputs=mlp_x_z_ss1,
                            grad_outputs=grad_outputs,
                            create_graph=False,
                            retain_graph=True
                        )

                        for col_idx_in_gain, col in enumerate(gain_target_mv):
                            col_idx_in_tab = tab_input_cols.index(col)
                            s_val = tab_std.get(col, 1.0)
                            s = s_val if abs(s_val) > 1e-6 else 1.0
                            K_ss_matrix[:, t_idx, col_idx_in_gain] = grads_x[:, col_idx_in_tab] / s

                K_ss_direction = torch.sign(K_ss_matrix)

                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                log_cols_inv = [c for c in ['B35_H2S', 'B35_SO2'] if c in y_sv]
                log_target_idx = [y_sv.index(c) for c in log_cols_inv]
                gain_start_history = x_en_z_history.clone()

                def de_physical_to_z(de_p_tensor):
                    de_z_cols = []
                    for col_idx_in_de, col in enumerate(de_mv):
                        m = mean_all.get(col, 0.0)
                        s_val = std_all.get(col, 1.0)
                        s = s_val if abs(s_val) > 1e-6 else 1.0
                        de_z_cols.append((de_p_tensor[:, col_idx_in_de] - m) / s)
                    return torch.stack(de_z_cols, dim=1)

                def rollout_gain_stack(base_de_p_tensor, step_de_p_tensor=None, collect_debug=False):
                    base_de_z_graph = de_physical_to_z(base_de_p_tensor)
                    step_de_z_graph = (
                        base_de_z_graph if step_de_p_tensor is None else de_physical_to_z(step_de_p_tensor)
                    )
                    gain_history = gain_start_history.clone()
                    gain_log_steps_local = []
                    debug_steps_local = []
                    for step_idx in range(H_gain):
                        current_de_z = (
                            step_de_z_graph
                            if step_idx >= dynamic_gain_step_change_idx
                            else base_de_z_graph
                        )
                        single_step_de_input = current_de_z.unsqueeze(1)
                        _, context = dynamic_model.encoder(gain_history)
                        single_step_prediction = dynamic_model.decoder(single_step_de_input, context)
                        pred_log_or_p_gain = single_step_prediction.squeeze(1) * y_std_safe + y_mean_tensor
                        gain_log_steps_local.append(pred_log_or_p_gain[:, tab_target_idx])

                        if collect_debug:
                            pred_p_gain = pred_log_or_p_gain.clone()
                            if len(log_target_idx) > 0:
                                pred_p_gain[:, log_target_idx] = torch.exp(pred_p_gain[:, log_target_idx])
                            pred_p_gain[:, tab_target_idx] = torch.clamp(pred_p_gain[:, tab_target_idx], min=1e-6)
                            debug_steps_local.append(pred_p_gain[:, tab_target_idx].detach())

                        if step_idx < H_gain - 1:
                            new_step_features = torch.cat(
                                [single_step_de_input, single_step_prediction.detach()],
                                dim=2
                            )
                            gain_history = torch.cat([gain_history[:, 1:, :], new_step_features], dim=1)

                    gain_log_stack_local = torch.stack(gain_log_steps_local, dim=0)
                    return gain_log_stack_local, base_de_z_graph, debug_steps_local

                baseline_de_p = ss1_de_p.detach().clone()
                ss_gain_de_p = baseline_de_p.clone()
                if dynamic_gain_method == 'autograd':
                    ss_gain_de_p.requires_grad_(True)

                gain_log_stack, _, _ = rollout_gain_stack(
                    baseline_de_p,
                    step_de_p_tensor=ss_gain_de_p if dynamic_gain_method == 'autograd' else None,
                    collect_debug=False
                )

                baseline_target_steps = gain_log_stack[
                    dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                ].mean(dim=0)

                if dynamic_gain_method == 'autograd':
                    K_dyn_by_target = []
                    for target_idx in range(len(tab_target_cols)):
                        target_steps = baseline_target_steps[:, target_idx]
                        grads_de_p, = torch.autograd.grad(
                            outputs=target_steps.sum(),
                            inputs=ss_gain_de_p,
                            create_graph=True,
                            retain_graph=True
                        )
                        K_dyn_by_target.append(grads_de_p[:, gain_target_mv_indices])
                    K_dyn_matrix = torch.stack(K_dyn_by_target, dim=1)
                    K_dyn_matrix_stack = K_dyn_matrix.unsqueeze(0)
                else:
                    K_dyn_plus_by_mv = []
                    K_dyn_minus_by_mv = []
                    for mv_name in gain_target_mv:
                        mv_de_idx = de_mv.index(mv_name)
                        std_val = std_all[mv_name] if abs(std_all[mv_name]) > 1e-6 else 1.0
                        delta = finite_diff_delta_std * std_val

                        plus_de_p = baseline_de_p.clone()
                        minus_de_p = baseline_de_p.clone()
                        plus_de_p[:, mv_de_idx] += delta
                        minus_de_p[:, mv_de_idx] -= delta

                        plus_log_stack, _, _ = rollout_gain_stack(
                            baseline_de_p, step_de_p_tensor=plus_de_p, collect_debug=False
                        )
                        minus_log_stack, _, _ = rollout_gain_stack(
                            baseline_de_p, step_de_p_tensor=minus_de_p, collect_debug=False
                        )
                        plus_target_steps = plus_log_stack[
                            dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                        ].mean(dim=0)
                        minus_target_steps = minus_log_stack[
                            dynamic_gain_tail_start_idx:dynamic_gain_tail_end_idx, :, :
                        ].mean(dim=0)

                        K_dyn_plus_by_mv.append((plus_target_steps - baseline_target_steps) / delta)
                        K_dyn_minus_by_mv.append((baseline_target_steps - minus_target_steps) / delta)

                    K_dyn_plus_matrix = torch.stack(K_dyn_plus_by_mv, dim=2)
                    K_dyn_minus_matrix = torch.stack(K_dyn_minus_by_mv, dim=2)
                    K_dyn_matrix_stack = torch.stack([K_dyn_plus_matrix, K_dyn_minus_matrix], dim=0)

                if pgin_runtime_plot and step == 0:
                    runtime_dir = './results/PGIN_Visualizations'
                    os.makedirs(runtime_dir, exist_ok=True)

                    ss_target_sample = ss_target_z[0].detach().cpu().numpy()
                    ss_target_mean_np = mean_all[ss_target_cols].values
                    ss_target_std_np = std_all[ss_target_cols].replace(0, 1).values
                    ss_pred_phys = gain_log_stack[:, 0, :].detach().cpu().numpy()
                    ss_target_phys = ss_target_sample * ss_target_std_np + ss_target_mean_np
                    for log_idx, col_name in enumerate(tab_target_cols):
                        if col_name in ['B35_H2S', 'B35_SO2']:
                            ss_pred_phys[:, log_idx] = np.exp(ss_pred_phys[:, log_idx])
                    for log_idx, col_name in enumerate(ss_target_cols):
                        if col_name in ['B35_H2S', 'B35_SO2']:
                            ss_target_phys[log_idx] = np.exp(ss_target_phys[log_idx])

                    time_axis = np.arange(1, H_gain + 1)

                    response_curves = {}
                    response_ann_targets = {}
                    with torch.no_grad():
                        for mv_name in gain_target_mv:
                            mv_de_idx = de_mv.index(mv_name)
                            mv_tab_idx = tab_input_cols.index(mv_name)
                            delta = 0.5 * (std_all[mv_name] if abs(std_all[mv_name]) > 1e-6 else 1.0)
                            response_curves[mv_name] = {}
                            response_ann_targets[mv_name] = {}

                            for direction, signed_delta in [("plus", delta), ("minus", -delta)]:
                                response_de_p = ss1_de_p.detach().clone()
                                response_de_p[:, mv_de_idx] += signed_delta
                                response_log_stack, _, _ = rollout_gain_stack(
                                    baseline_de_p,
                                    step_de_p_tensor=response_de_p,
                                    collect_debug=False
                                )
                                response_phys_stack = response_log_stack[:, 0, :].detach().cpu().numpy()
                                for log_idx, col_name in enumerate(tab_target_cols):
                                    if col_name in ['B35_H2S', 'B35_SO2']:
                                        response_phys_stack[:, log_idx] = np.exp(response_phys_stack[:, log_idx])
                                response_curves[mv_name][direction] = response_phys_stack

                                response_mlp_x_z = mlp_x_z_ss1.detach().clone()
                                tab_s_val = tab_std.get(mv_name, 1.0)
                                tab_s = tab_s_val if abs(tab_s_val) > 1e-6 else 1.0
                                response_mlp_x_z[:, mv_tab_idx] += signed_delta / tab_s
                                response_mlp_p = mlp_model(response_mlp_x_z) * tab_target_std_safe + tab_target_mean_tensor
                                response_ann_targets[mv_name][direction] = (
                                    response_mlp_p[0, ann_monitor_idx].detach().cpu().numpy()
                                )

                    fig, axes = plt.subplots(
                        1 + len(gain_target_mv),
                        len(tab_target_cols),
                        figsize=(6 * len(tab_target_cols), 4 * (1 + len(gain_target_mv)))
                    )
                    if len(tab_target_cols) == 1:
                        axes = axes.reshape(1 + len(gain_target_mv), 1)

                    for tgt_idx, tgt_name in enumerate(tab_target_cols):
                        ss_target_col_idx = ss_target_cols.index(tgt_name)
                        axes[0, tgt_idx].plot(time_axis, ss_pred_phys[:, tgt_idx], color='purple', linewidth=2, label='Transformer rollout')
                        axes[0, tgt_idx].axhline(ss_target_phys[ss_target_col_idx], color='black', linestyle='--', linewidth=1.5, label='Excel S.S. target')
                        axes[0, tgt_idx].set_title(f'{tgt_name} rollout | epoch {epoch + 1}, step {step}')
                        axes[0, tgt_idx].set_xlabel('Decoder step')
                        axes[0, tgt_idx].set_ylabel('Physical value')
                        axes[0, tgt_idx].grid(True, linestyle='--', alpha=0.5)
                        axes[0, tgt_idx].legend(loc='best')

                        for mv_plot_idx, mv_name in enumerate(gain_target_mv, start=1):
                            ax = axes[mv_plot_idx, tgt_idx]
                            ax.plot(time_axis, ss_pred_phys[:, tgt_idx], color='black', linewidth=1.8, label='baseline')
                            ax.plot(time_axis, response_curves[mv_name]["plus"][:, tgt_idx], color='crimson', linewidth=2, label=f'{mv_name} +0.5 std')
                            ax.plot(time_axis, response_curves[mv_name]["minus"][:, tgt_idx], color='steelblue', linewidth=2, label=f'{mv_name} -0.5 std')
                            ax.axhline(ss_target_phys[ss_target_col_idx], color='black', linestyle='--', linewidth=1, alpha=0.8, label='Excel baseline')
                            ax.axhline(response_ann_targets[mv_name]["plus"][tgt_idx], color='crimson', linestyle='--', linewidth=1, alpha=0.8, label='ANN +step')
                            ax.axhline(response_ann_targets[mv_name]["minus"][tgt_idx], color='steelblue', linestyle='--', linewidth=1, alpha=0.8, label='ANN -step')
                            ax.set_title(f'{tgt_name} response to {mv_name}')
                            ax.set_xlabel('Decoder step')
                            ax.set_ylabel('Physical value')
                            ax.grid(True, linestyle='--', alpha=0.5)
                            ax.legend(loc='best', fontsize=8)

                    plt.tight_layout()
                    latest_runtime_path = os.path.join(runtime_dir, 'pgin_runtime_latest.png')
                    plt.savefig(latest_runtime_path, dpi=180)
                    plt.close()
                    print(f"\n  [Plot] Updated PGIN runtime rollout/gain plot: {latest_runtime_path}")

                K_ss_direction_exp = K_ss_direction.unsqueeze(0)

                loss_matrix_stack = torch.nn.functional.relu(-K_dyn_matrix_stack * K_ss_direction_exp)

                valid_mlp_mask = torch.abs(K_ss_matrix) >= gain_valid_delta_threshold
                valid_mlp_mask_expanded = valid_mlp_mask.unsqueeze(0).expand_as(loss_matrix_stack)

                final_mask = valid_mlp_mask_expanded
                correct_mask = (K_dyn_matrix_stack * K_ss_direction_exp > 0) & final_mask

                valid_items = final_mask.sum().item()
                epoch_correct_dir += correct_mask.sum().item()
                epoch_total_eval += valid_items

                pair_correct += correct_mask.sum(dim=(0, 1))
                pair_total += final_mask.sum(dim=(0, 1))
                pair_ann_sign_sum += (K_ss_direction_exp.expand_as(final_mask).float() * final_mask.float()).sum(dim=(0, 1))
                pair_dyn_pos_sum += ((K_dyn_matrix_stack > 0).float() * final_mask.float()).sum(dim=(0, 1))

                if valid_items > 0:
                    loss_gain = torch.mean(loss_matrix_stack[final_mask])
                else:
                    loss_gain = torch.tensor(0.0, device=device, requires_grad=True)
                dynamic_model.train(gain_prev_training_mode)

            total_loss = (
                mse_loss_val
                + smooth_loss_weight * smooth_loss_val
                + effective_gain_loss_weight * loss_gain
            )
            epoch_mse_loss += mse_loss_val.item()
            epoch_smooth_loss += smooth_loss_val.item()
            epoch_total_loss += total_loss.item()

            total_norm_val = 0.0
            if total_loss.item() > 0:
                total_loss.backward()
                # clip_grad_norm_ returns the gradient norm before clipping.
                total_norm = torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), 1.0)
                total_norm_val = total_norm.item()
                optimizer.step()

            epoch_gain_loss += loss_gain.item()
            pbar.set_postfix({
                'MSE': f"{mse_loss_val.item():.4f}",
                'D1_w': f"{(smooth_loss_weight * smooth_loss_val).item():.5f}",
                'Gain_w': f"{(effective_gain_loss_weight * loss_gain).item():.5f}",
                'KCI_n': f"{epoch_correct_dir}/{epoch_total_eval}"
            })

        avg_loss = epoch_total_loss / step_limit
        avg_mse_loss = epoch_mse_loss / step_limit
        avg_smooth_loss = epoch_smooth_loss / step_limit
        avg_gain_loss = epoch_gain_loss / step_limit
        epoch_kci = epoch_correct_dir / epoch_total_eval if epoch_total_eval > 0 else 1.0
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Total L: {avg_loss:.6f} | "
            f"MSE: {avg_mse_loss:.6f} | D1: {avg_smooth_loss:.6f} | "
            f"Gain: {avg_gain_loss:.8f} | KCI: {epoch_correct_dir}/{epoch_total_eval} = {epoch_kci*100:.2f}%"
        )
        if compute_gain_metrics:
            pair_correct_cpu = pair_correct.detach().cpu()
            pair_total_cpu = pair_total.detach().cpu()
            pair_ann_sign_cpu = pair_ann_sign_sum.detach().cpu()
            pair_dyn_pos_cpu = pair_dyn_pos_sum.detach().cpu()
            print("  [Gain Pair KCI]")
            for q_idx, q_name in enumerate(tab_target_cols):
                parts = []
                for mv_idx, mv_name in enumerate(gain_target_mv):
                    total_pair = pair_total_cpu[q_idx, mv_idx].item()
                    correct_pair = pair_correct_cpu[q_idx, mv_idx].item()
                    if total_pair > 0:
                        kci_pair = 100.0 * correct_pair / total_pair
                        ann_sign_avg = pair_ann_sign_cpu[q_idx, mv_idx].item() / total_pair
                        dyn_pos_pct = 100.0 * pair_dyn_pos_cpu[q_idx, mv_idx].item() / total_pair
                        history_gain_pair_rows.append({
                            'epoch': epoch + 1,
                            'target': q_name,
                            'mv': mv_name,
                            'correct': correct_pair,
                            'total': total_pair,
                            'kci_percent': kci_pair,
                            'ann_sign_avg': ann_sign_avg,
                            'dynamic_positive_percent': dyn_pos_pct,
                        })
                        parts.append(
                            f"{mv_name}: {correct_pair:.0f}/{total_pair:.0f}={kci_pair:.1f}% "
                            f"ann_sign_avg={ann_sign_avg:.2f} dyn_pos={dyn_pos_pct:.1f}%"
                        )
                    else:
                        history_gain_pair_rows.append({
                            'epoch': epoch + 1,
                            'target': q_name,
                            'mv': mv_name,
                            'correct': 0.0,
                            'total': 0.0,
                            'kci_percent': np.nan,
                            'ann_sign_avg': np.nan,
                            'dynamic_positive_percent': np.nan,
                        })
                        parts.append(f"{mv_name}: no valid")
                print(f"    {q_name} | " + " ; ".join(parts))
        ann_clamp_pct = 100.0 * ann_monitor_clamped / ann_monitor_count if ann_monitor_count > 0 else 0.0
        ann_min_str = ', '.join(
            f"{name}={value:.6g}" for name, value in zip(tab_target_cols, ann_monitor_min.detach().cpu().tolist())
        )
        ann_max_str = ', '.join(
            f"{name}={value:.6g}" for name, value in zip(tab_target_cols, ann_monitor_max.detach().cpu().tolist())
        )
        print(
            f"  [ANN Gain Teacher] target min: {ann_min_str} | max: {ann_max_str} | "
            f"<=1e-6: {ann_clamp_pct:.2f}%"
        )
        history_losses.append(avg_loss)
        history_kci.append(epoch_kci)

        # Validation Phase
        dynamic_model.eval()
        val_mse_loss = 0.0
        if valid_loader is not None:
            with torch.no_grad():
                for mse_batch in valid_loader:
                    v_loss = step_wise_rolling_training_step(dynamic_model, mse_batch, criterion, device)
                    val_mse_loss += v_loss.item()
            val_mse_loss /= len(valid_loader)
            print(f"  [Valid] MSE Loss: {val_mse_loss:.6f}")
        else:
            val_mse_loss = avg_loss

        history_val_losses.append(val_mse_loss)

        out_model_path = f'./saved_models/{exp}.pth'
        if val_mse_loss < best_loss:
            best_loss = val_mse_loss
            epochs_no_improve = 0
            print(f"  [Save] New Best Model saved to {out_model_path} (Val Loss: {best_loss:.6f})")
            torch.save(dynamic_model.state_dict(), out_model_path)
        else:
            epochs_no_improve += 1
            print(f"  [Info] No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stop_patience:
            print(f"\n[Early Stopping] Triggered after {epoch + 1} epochs without improvement in validation loss.")
            break

    out_dir = f'./results/{exp}'
    os.makedirs(out_dir, exist_ok=True)

    actual_epochs = len(history_losses)
    epochs_axis = list(range(1, actual_epochs + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, history_losses, marker='o', color='blue', label='Train Total', linewidth=2)
    if len(history_val_losses) > 0 and valid_loader is not None:
        plt.plot(epochs_axis, history_val_losses, marker='o', color='red', label='Valid MSE', linewidth=2)
    plt.title(f'Training & Validation Loss - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_loss_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    kci_percentages = [kci * 100 for kci in history_kci]
    kci_df = pd.DataFrame({
        'epoch': epochs_axis,
        'kci_percent': kci_percentages,
    })
    kci_csv_path = os.path.join(out_dir, 'kci_history.csv')
    kci_df.to_csv(kci_csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, kci_percentages, marker='s', color='orange', linewidth=2)
    plt.title(f'PGIN KCI Consistency - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Consistent Steps (%)')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    kci_plot_path = os.path.join(out_dir, 'kci_consistency_curve.png')
    plt.savefig(kci_plot_path, dpi=300)
    plt.close()

    if history_gain_pair_rows:
        gain_pair_df = pd.DataFrame(history_gain_pair_rows)
        gain_pair_csv_path = os.path.join(out_dir, 'gain_pair_kci_history.csv')
        gain_pair_df.to_csv(gain_pair_csv_path, index=False)

        last_epoch = int(gain_pair_df['epoch'].max())
        last_pair_df = gain_pair_df[gain_pair_df['epoch'] == last_epoch].copy()
        last_pair_df['pair'] = last_pair_df['target'] + '\n' + last_pair_df['mv']

        plt.figure(figsize=(11, 5))
        colors = ['seagreen' if v >= 95 else 'goldenrod' if v >= 80 else 'crimson'
                  for v in last_pair_df['kci_percent'].fillna(0)]
        bars = plt.bar(last_pair_df['pair'], last_pair_df['kci_percent'].fillna(0), color=colors)
        plt.axhline(95, color='black', linestyle='--', linewidth=1, alpha=0.7, label='95% reference')
        plt.ylim(0, 105)
        plt.ylabel('KCI (%)')
        plt.title(f'Final Gain Direction Consistency - Epoch {last_epoch}')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        for bar, total in zip(bars, last_pair_df['total']):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                min(height + 2, 103),
                f'{height:.1f}%\nn={int(total)}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        plt.tight_layout()
        gain_pair_plot_path = os.path.join(out_dir, 'final_gain_pair_kci_bar.png')
        plt.savefig(gain_pair_plot_path, dpi=300)
        plt.close()

    print(f"\n[Done] Model training complete. Weights securely saved to {out_model_path}")
    print(f"[Info] Training metrics exported to {plot_path}")
    print(f"[Info] KCI curve exported to {kci_plot_path}")
    print(f"[Info] KCI CSV exported to {kci_csv_path}")
    if history_gain_pair_rows:
        print(f"[Info] Gain pair KCI bar chart exported to {gain_pair_plot_path}")
        print(f"[Info] Gain pair KCI history exported to {gain_pair_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)


