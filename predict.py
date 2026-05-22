# predict.py - Long-term rolling prediction using trained models with sliding window or block replacement strategies (????)
import torch
import numpy as np
import pandas as pd
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ????????
from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.utils import calculate_metrics

SPLIT_COLORS = {
    'train': '#1f77b4',
    'valid': '#d62728',
    'test': '#2ca02c',
    'unknown': '#6b7280',
}

TRUE_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"
HISTORY_COLOR = "#6b7280"


def style_prediction_axis(ax, y_ticks=5):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.9, colors="#1f2937")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f2937")
        spine.set_linewidth(1.0)

# ==============================================================================
# --- ???? ---
# ==============================================================================

def predict_sliding_window(model, initial_en_input, future_de_inputs, device, num_output_features):
    """Sliding window ?
    
    ??
    1. ??encoder  + ?? MV ??? y_sv
    2. ?[MV, y_sv] ????encoder ???
    """
    model.eval()
    
    num_pred_steps = future_de_inputs.shape[1]
    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for t in tqdm(range(num_pred_steps), desc="[Strategy: sliding window] Predicting"):
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model(current_en_input, single_step_de_input)
            
            # Check if model returns full prediction (like iTransformer) or single step
            if single_step_prediction.shape[1] > 1:
                # Direct step prediction (The model predicts whole future at once)
                # In sliding window context, we just take the first step, or we should switch strategy.
                # Here we assume the user intends to use sliding window, so we take the first step.
                single_step_prediction = single_step_prediction[:, 0, :].unsqueeze(1)
            
            predictions[:, t, :] = single_step_prediction

            # ??????????
            next_en_input_history = current_en_input[:, 1:, :]
            # ???MV, y_sv] ??en_mv_and_sv ???
            new_step_features = torch.cat([single_step_de_input, single_step_prediction], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)
    
    return predictions


def predict_block_replacement(model, initial_en_input, future_de_inputs, device, config):
    """Block replacement ?
    
    ????
    1. ??encoder  + ?? MV ?????? y_sv
    2. ?[MV, y_sv] ????????? encoder ??
    """
    model.eval()
    
    H = config['window']['train_window_mins'] // config['window']['sampling_interval_min']
    num_pred_steps = future_de_inputs.shape[1]
    
    if num_pred_steps % H != 0:
        print(f"Warning: prediction length {num_pred_steps} is not divisible by block size {H}.")
        num_pred_steps = (num_pred_steps // H) * H
        print(f"Truncating prediction length to {num_pred_steps} steps.")
        future_de_inputs = future_de_inputs[:, :num_pred_steps, :]

    num_windows_to_predict = num_pred_steps // H
    predictions_all_windows = []
    current_en_input = initial_en_input.clone().to(device)
    
    with torch.no_grad():
        for i in tqdm(range(num_windows_to_predict), desc="[Strategy: block replacement] Predicting"):
            start_idx = i * H
            end_idx = (i + 1) * H
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)

            prediction_block = model(current_en_input, de_input_block)
            predictions_all_windows.append(prediction_block)

            # ???MV, y_sv] ??en_mv_and_sv ???
            current_en_input = torch.cat([de_input_block, prediction_block], dim=2)

    return torch.cat(predictions_all_windows, dim=1)

def predict_receding_block_replacement(model, initial_en_input, future_de_inputs, device, config):
    """Predict H steps ahead, commit only the first few steps, then re-plan."""
    model.eval()

    H = config['window']['prediction_length']
    commit_steps = config['training'].get('block_commit_steps', 3)
    commit_steps = max(1, min(commit_steps, H))
    num_pred_steps = future_de_inputs.shape[1]

    predictions_committed = []
    current_en_input = initial_en_input.clone().to(device)

    with torch.no_grad():
        for start_idx in tqdm(
            range(0, num_pred_steps, commit_steps),
            desc=f"[Strategy: receding block | commit={commit_steps}] Predicting"
        ):
            end_idx = min(start_idx + H, num_pred_steps)
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)

            if de_input_block.size(1) < H:
                pad_len = H - de_input_block.size(1)
                pad_block = de_input_block[:, -1:, :].expand(-1, pad_len, -1)
                de_input_for_model = torch.cat([de_input_block, pad_block], dim=1)
            else:
                de_input_for_model = de_input_block

            prediction_block = model(current_en_input, de_input_for_model)
            keep_len = min(commit_steps, num_pred_steps - start_idx)
            committed_pred = prediction_block[:, :keep_len, :]
            committed_de = future_de_inputs[:, start_idx:start_idx + keep_len, :].to(device)

            predictions_committed.append(committed_pred)

            new_block = torch.cat([committed_de, committed_pred], dim=2)
            current_en_input = torch.cat([current_en_input[:, new_block.size(1):, :], new_block], dim=1)

    return torch.cat(predictions_committed, dim=1)

def predict_horizon_reinit(model, initial_en_input, future_de_inputs, future_targets, full_en_inputs, device, config):
    """
    Horizon Re-initialization Strategy:
    At each step H (prediction horizon), we RESET the encoder input history 
    using the GROUND TRUTH history from 'full_en_inputs'.
    This simulates MPC behavior where at each decision point, we have access to the true past state.
    """
    model.eval()
    predictions_all = []
    
    # Get parameters
    weights = config['training']['loss_weighting']['weights']
    num_windows = len(weights) # Usually 1
    total_pred_len = future_de_inputs.shape[1] # Total steps to predict
    
    # H is the block size for one prediction call
    H = config['window']['prediction_length'] 
    
    # Current history tensor (starts with initial)
    W = initial_en_input.shape[1]
    
    # Calculate Reset Interval (e.g. 18 * 4 = 72 steps)
    reinit_interval_steps = H * num_windows
    
    total_steps = future_de_inputs.shape[1]
    
    # Calculate active windows based on weights
    # e.g. [1, 0, 0, 0] -> Last active index 0 -> Predict 1 block (18 steps)
    # e.g. [1, 1, 0, 0] -> Last active index 1 -> Predict 2 blocks (36 steps)
    last_active_idx = 0
    for idx, w in enumerate(weights):
        if w > 0:
            last_active_idx = idx
    
    num_H_blocks = last_active_idx + 1
    
    print(f"  [Horizon Reinit] Weights={weights} -> Active Blocks={num_H_blocks} ({num_H_blocks * H} steps).")

    current_en_input = None 
    
    with torch.no_grad():
        for i in range(num_H_blocks):
            # Current Global Step Start relative to T_start (W)
            global_step = i * H
            
            # Reset Logic: 
            # i=0 -> Reset (Use Ground Truth)
            # i>0 -> AR Update
            # Since we stop at num_windows, we only reset once at the beginning.
            should_reset = (i == 0)
            
            # 1. Get DE Input for this H-block
            # If we run out of future data, stop
            if global_step >= total_steps:
                break

            end_step = min(global_step + H, total_steps)
            de_input_block = future_de_inputs[:, global_step:end_step, :].to(device)
            
            if should_reset:
                # Reset from Ground Truth History
                # Indexing into full_en_inputs (which starts at T=0)
                # Prediction starts at T=W.
                # History needed for block 0 (predicting T=W..W+H) is T=0..W
                
                # Check bounds just in case
                if global_step + W > full_en_inputs.shape[1]:
                     break
                     
                current_en_input = full_en_inputs[:, global_step : global_step + W, :].to(device)
                
            else:
                # Autoregressive Update
                prev_pred = predictions_all[-1]
                prev_de_input = future_de_inputs[:, global_step-H : global_step, :].to(device)
                
                # Construct new history chunk [MV, SV]
                new_hist_chunk = torch.cat([prev_de_input, prev_pred], dim=2)
                
                # Shift Left and Append
                current_en_input = torch.cat([current_en_input[:, H:, :], new_hist_chunk], dim=1)

            # 2. Model Prediction
            pred = model(current_en_input, de_input_block)
            predictions_all.append(pred)
            
    if not predictions_all:
        return torch.tensor([])
        
    return torch.cat(predictions_all, dim=1)

def split_case_name(test_name):
    """Return (case_name, split_name) from names like R5-1_train_split."""
    for suffix, split_name in [
        ('_train_split', 'train'),
        ('_valid_split', 'valid'),
        ('_test_split', 'test'),
    ]:
        if test_name.endswith(suffix):
            return test_name[:-len(suffix)], split_name
    return test_name, 'unknown'


def plot_grouped_horizon_parity(horizon_runs, config, output_root):
    """Plot all cases together: one parity overview for t+1 and one for t+18."""
    if not horizon_runs:
        return

    grouped = {}
    case_order = []
    variable_order = []
    step_order = []
    for run in horizon_runs:
        test_name = run['test_name']
        case_name, split_name = split_case_name(test_name)
        if case_name not in case_order:
            case_order.append(case_name)
        for item in run.get('parity_data', []):
            key = (case_name, item['step_num'], item['variable'])
            if item['variable'] not in variable_order:
                variable_order.append(item['variable'])
            if item['step_num'] not in step_order:
                step_order.append(item['step_num'])
            grouped.setdefault(key, []).append({
                'split': split_name,
                'true': item['true'],
                'pred': item['pred'],
            })

    if not grouped:
        return

    out_dir = os.path.join(output_root, config['exp_name'], 'grouped_horizon_parity')
    os.makedirs(out_dir, exist_ok=True)
    metric_rows = []

    for step_num in sorted(step_order):
        n_rows = len(case_order)
        n_cols = len(variable_order)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.0 * n_cols, 4.8 * n_rows),
            squeeze=False,
        )

        for row_idx, case_name in enumerate(case_order):
            for col_idx, variable in enumerate(variable_order):
                ax = axes[row_idx, col_idx]
                split_items = grouped.get((case_name, step_num, variable), [])
                all_true = []
                all_pred = []

                for split_name in ['train', 'valid', 'test', 'unknown']:
                    items = [x for x in split_items if x['split'] == split_name]
                    if not items:
                        continue

                    y_true = np.concatenate([x['true'] for x in items])
                    y_pred = np.concatenate([x['pred'] for x in items])
                    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_pred) < 1e100)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    if len(y_true) == 0:
                        continue

                    all_true.append(y_true)
                    all_pred.append(y_pred)
                    metrics = calculate_metrics(y_true, y_pred)
                    metric_rows.append({
                        'plot_type': 'parity',
                        'case': case_name,
                        'horizon': step_num,
                        'variable': variable,
                        'split': split_name,
                        'n_samples': len(y_true),
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'R2': metrics['R2'],
                        'MAPE': metrics['MAPE'],
                    })
                    ax.scatter(
                        y_true,
                        y_pred,
                        s=14,
                        alpha=0.62,
                        color=SPLIT_COLORS.get(split_name, 'tab:gray'),
                        edgecolors='none',
                        label=split_name,
                    )

                if all_true:
                    y_true_all = np.concatenate(all_true)
                    y_pred_all = np.concatenate(all_pred)
                    combined_metrics = calculate_metrics(y_true_all, y_pred_all)
                    metric_rows.append({
                        'plot_type': 'parity',
                        'case': case_name,
                        'horizon': step_num,
                        'variable': variable,
                        'split': 'all',
                        'n_samples': len(y_true_all),
                        'MAE': combined_metrics['MAE'],
                        'RMSE': combined_metrics['RMSE'],
                        'R2': combined_metrics['R2'],
                        'MAPE': combined_metrics['MAPE'],
                    })
                    min_val = min(y_true_all.min(), y_pred_all.min())
                    max_val = max(y_true_all.max(), y_pred_all.max())
                    margin = (max_val - min_val) * 0.05
                    if margin == 0:
                        margin = 1.0

                    ax.plot(
                        [min_val - margin, max_val + margin],
                        [min_val - margin, max_val + margin],
                        color='#111827',
                        linestyle='--',
                        linewidth=1.2,
                        label='Ideal',
                    )
                    ax.set_xlim(min_val - margin, max_val + margin)
                    ax.set_ylim(min_val - margin, max_val + margin)
                    ax.legend(fontsize=7, frameon=True, edgecolor='#1f2937')
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

                ax.set_xlabel('True Value')
                ax.set_ylabel('Predicted Value')
                style_prediction_axis(ax)
                ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        save_name = f'all_cases_grouped_parity_t{step_num}.png'
        plt.savefig(os.path.join(out_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()

        for case_name in case_order:
            fig_case, axes_case = plt.subplots(
                1,
                n_cols,
                figsize=(7.0 * n_cols, 4.8),
                squeeze=False,
            )

            for col_idx, variable in enumerate(variable_order):
                ax = axes_case[0, col_idx]
                split_items = grouped.get((case_name, step_num, variable), [])
                all_true = []
                all_pred = []

                for split_name in ['train', 'valid', 'test', 'unknown']:
                    items = [x for x in split_items if x['split'] == split_name]
                    if not items:
                        continue

                    y_true = np.concatenate([x['true'] for x in items])
                    y_pred = np.concatenate([x['pred'] for x in items])
                    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_pred) < 1e100)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    if len(y_true) == 0:
                        continue

                    all_true.append(y_true)
                    all_pred.append(y_pred)
                    ax.scatter(
                        y_true,
                        y_pred,
                        s=16,
                        alpha=0.65,
                        color=SPLIT_COLORS.get(split_name, 'tab:gray'),
                        edgecolors='none',
                        label=split_name,
                    )

                if all_true:
                    y_true_all = np.concatenate(all_true)
                    y_pred_all = np.concatenate(all_pred)
                    min_val = min(y_true_all.min(), y_pred_all.min())
                    max_val = max(y_true_all.max(), y_pred_all.max())
                    margin = (max_val - min_val) * 0.05
                    if margin == 0:
                        margin = 1.0

                    ax.plot(
                        [min_val - margin, max_val + margin],
                        [min_val - margin, max_val + margin],
                        color='#111827',
                        linestyle='--',
                        linewidth=1.2,
                        label='Ideal',
                    )
                    ax.set_xlim(min_val - margin, max_val + margin)
                    ax.set_ylim(min_val - margin, max_val + margin)
                    ax.legend(fontsize=8, frameon=True, edgecolor='#1f2937')
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')

                ax.set_xlabel('True Value')
                ax.set_ylabel('Predicted Value')
                style_prediction_axis(ax)
                ax.set_aspect('equal', adjustable='box')

            plt.tight_layout()
            safe_case_name = case_name.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
            case_save_name = f'{safe_case_name}_grouped_parity_t{step_num}.png'
            plt.savefig(os.path.join(out_dir, case_save_name), dpi=150, bbox_inches='tight')
            plt.close()

    metrics_path = os.path.join(out_dir, 'grouped_parity_metrics.csv')
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f'Grouped t+1/t+18 parity plots saved to {out_dir}')
    print(f'Grouped parity metrics saved to {metrics_path}')


def plot_grouped_time_series(run_results, config, output_root):
    """Plot all cases together in one rolling time-series overview."""
    if not run_results:
        return

    grouped = {}
    case_order = []
    variable_order = []
    for run in run_results:
        test_name = run['test_name']
        case_name, split_name = split_case_name(test_name)
        if case_name not in case_order:
            case_order.append(case_name)
        for item in run.get('time_series_data', []):
            variable = item['variable']
            if variable not in variable_order:
                variable_order.append(variable)
            grouped.setdefault((case_name, variable), []).append({
                'split': split_name,
                'true': item['true'],
                'pred': item['pred'],
                'r2': item['r2'],
                'rmse': item['rmse'],
            })

    if not grouped:
        return

    out_dir = os.path.join(output_root, config['exp_name'], 'grouped_time_series')
    os.makedirs(out_dir, exist_ok=True)
    metric_rows = []

    n_rows = len(case_order) * len(variable_order)
    n_cols = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.8, 3.2 * n_rows),
        squeeze=False,
    )

    for row_idx, case_name in enumerate(case_order):
        for col_idx, variable in enumerate(variable_order):
            ax = axes[row_idx * len(variable_order) + col_idx, 0]
            split_items = grouped.get((case_name, variable), [])

            x_offset = 0
            split_boundaries = []
            true_segments = []
            pred_segments = []
            for split_name in ['train', 'valid', 'test', 'unknown']:
                items = [x for x in split_items if x['split'] == split_name]
                if not items:
                    continue

                # Normally there is one item per split/case/variable.
                y_true = np.concatenate([x['true'] for x in items])
                y_pred = np.concatenate([x['pred'] for x in items])
                mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_pred) < 1e100)
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                if len(y_true) == 0:
                    continue

                metrics = calculate_metrics(y_true, y_pred)
                x = np.arange(x_offset, x_offset + len(y_true))
                color = SPLIT_COLORS.get(split_name, 'tab:gray')
                true_segments.append((x, y_true))
                pred_segments.append((x, y_pred, split_name, color, metrics))
                metric_rows.append({
                    'plot_type': 'rolling_time_series',
                    'case': case_name,
                    'variable': variable,
                    'split': split_name,
                    'n_samples': len(y_true),
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'MAPE': metrics['MAPE'],
                })
                split_boundaries.append((x_offset, x_offset + len(y_true) - 1, split_name))
                x_offset += len(y_true)

            if true_segments and pred_segments:
                y_true_all = np.concatenate([y for _, y in true_segments])
                y_pred_all = np.concatenate([y for _, y, _, _, _ in pred_segments])
                combined_metrics = calculate_metrics(y_true_all, y_pred_all)
                metric_rows.append({
                    'plot_type': 'rolling_time_series',
                    'case': case_name,
                    'variable': variable,
                    'split': 'all',
                    'n_samples': len(y_true_all),
                    'MAE': combined_metrics['MAE'],
                    'RMSE': combined_metrics['RMSE'],
                    'R2': combined_metrics['R2'],
                    'MAPE': combined_metrics['MAPE'],
                })

            for start, end, split_name in split_boundaries:
                center = (start + end) / 2
                color = SPLIT_COLORS.get(split_name, 'tab:gray')
                ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.055, linewidth=0)
                ax.text(
                    center,
                    0.96,
                    split_name,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=10,
                    fontweight='bold',
                    color=color,
                )
                if start > 0:
                    ax.axvline(start - 0.5, color='0.35', linestyle=':', linewidth=1.1, alpha=0.9)

            for x, y_pred, split_name, color, metrics in pred_segments:
                ax.plot(
                    x,
                    y_pred,
                    color=color,
                    linewidth=0.8,
                    linestyle='--',
                    alpha=0.95,
                    label=f'{split_name} pred',
                    zorder=2,
                )

            for idx, (x, y_true) in enumerate(true_segments):
                ax.plot(
                    x,
                    y_true,
                    color='black',
                    linewidth=1.05,
                    alpha=0.82,
                    label='true' if idx == 0 else None,
                    zorder=3,
                )

            ax.set_xlabel('Rolling sample index')
            ax.set_ylabel(variable)
            style_prediction_axis(ax)
            ax.legend(
                fontsize=7,
                ncol=2,
                loc='lower left',
                frameon=True,
            )

    plt.tight_layout()
    save_name = 'all_cases_grouped_time_series.png'
    plt.savefig(os.path.join(out_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close()

    for case_name in case_order:
        fig_case, axes_case = plt.subplots(
            len(variable_order),
            1,
            figsize=(8.8, 3.2 * len(variable_order)),
            squeeze=False,
        )

        for col_idx, variable in enumerate(variable_order):
            ax = axes_case[col_idx, 0]
            split_items = grouped.get((case_name, variable), [])

            x_offset = 0
            split_boundaries = []
            true_segments = []
            pred_segments = []
            for split_name in ['train', 'valid', 'test', 'unknown']:
                items = [x for x in split_items if x['split'] == split_name]
                if not items:
                    continue

                y_true = np.concatenate([x['true'] for x in items])
                y_pred = np.concatenate([x['pred'] for x in items])
                mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_pred) < 1e100)
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                if len(y_true) == 0:
                    continue

                x = np.arange(x_offset, x_offset + len(y_true))
                color = SPLIT_COLORS.get(split_name, 'tab:gray')
                true_segments.append((x, y_true))
                pred_segments.append((x, y_pred, split_name, color))
                split_boundaries.append((x_offset, x_offset + len(y_true) - 1, split_name))
                x_offset += len(y_true)

            for start, end, split_name in split_boundaries:
                center = (start + end) / 2
                color = SPLIT_COLORS.get(split_name, 'tab:gray')
                ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.055, linewidth=0)
                ax.text(
                    center,
                    0.96,
                    split_name,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=10,
                    fontweight='bold',
                    color=color,
                )
                if start > 0:
                    ax.axvline(start - 0.5, color='0.35', linestyle=':', linewidth=1.1, alpha=0.9)

            for x, y_pred, split_name, color in pred_segments:
                ax.plot(
                    x,
                    y_pred,
                    color=color,
                    linewidth=0.85,
                    linestyle='--',
                    alpha=0.95,
                    label=f'{split_name} pred',
                    zorder=2,
                )

            for idx, (x, y_true) in enumerate(true_segments):
                ax.plot(
                    x,
                    y_true,
                    color='black',
                    linewidth=1.1,
                    alpha=0.84,
                    label='true' if idx == 0 else None,
                    zorder=3,
                )

            ax.set_xlabel('Rolling sample index')
            ax.set_ylabel(variable)
            style_prediction_axis(ax)
            ax.legend(
                fontsize=8,
                ncol=2,
                loc='lower left',
                frameon=True,
                edgecolor='#1f2937',
            )

        plt.tight_layout()
        safe_case_name = case_name.replace(os.sep, '_').replace('/', '_').replace('\\', '_')
        case_save_name = f'{safe_case_name}_grouped_time_series.png'
        plt.savefig(os.path.join(out_dir, case_save_name), dpi=150, bbox_inches='tight')
        plt.close()

    metrics_path = os.path.join(out_dir, 'grouped_time_series_metrics.csv')
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f'Grouped rolling time-series plots saved to {out_dir}')
    print(f'Grouped rolling time-series metrics saved to {metrics_path}')


# ==============================================================================
# --- ??---
# ==============================================================================

def run_prediction(config, test_cfg, model, device, mean_all, std_all, en_mv_and_sv, de_mv, y_sv, W):
    """Run prediction and per-split time-series plots for one inference file."""
    test_name = test_cfg.get('name', 'Default_Test')
    print(f"\n========== Running inference: {test_name} ==========")
    print(f"File: {test_cfg['filename']}")

    # --- Step 1: ??? ---
    cfg_data = config['data']
    try:
        df_raw_test = data_utils.load_data(os.path.join(cfg_data['path'], test_cfg['filename']))
        print("Loaded test data with data_utils.load_data.")
    except (KeyError, ValueError, FileNotFoundError) as e:
        # Try finding in parent dir or absolute path
        fpath = test_cfg['filename']
        if not os.path.exists(fpath):
             fpath = os.path.join(cfg_data['path'], test_cfg['filename'])
        
        try:
            df_raw_test = pd.read_csv(fpath)
            print(f"Loaded test data with pandas: {fpath}")
        except Exception as e2:
             print(f"Failed to load test data: {e2}")
             return

    # Apply point limit
    limit_point = test_cfg.get('point', None)
    if limit_point:
        df_raw_test = df_raw_test.iloc[:limit_point]
        
    # [Step 1: Downsample FIRST] - consistent with training
    interval = cfg_data.get('sampling_interval_min', config['window'].get('sampling_interval_min', 1))
    use_median = config['window'].get('use_median_downsampling', True)

    if interval > 1:
        if use_median:
            print(f"Downsampling test data by MEDIAN resampling: interval={interval}")
            # New Logic: Rolling Median + Slice
            # Fix: numeric_only=True to prevent DataError on non-numeric columns
            df_median = df_raw_test.rolling(window=interval, min_periods=interval).median(numeric_only=True)
            df_raw_test = df_median.iloc[interval-1::interval].reset_index(drop=True)
            print(f"  -> Applied Rolling Median Filter (Window={interval})")
        else:
            print(f"Downsampling test data by SIMPLE SLICING: interval={interval}")
            df_raw_test = df_raw_test.iloc[::interval].reset_index(drop=True)
            print(f"  -> Applied Index Slicing (Step={interval})")
        
        print(f"  New test data length: {len(df_raw_test)}")
    
    df_raw_test.dropna(inplace=True)

    # [Step 2: Log Transform] - consistent with training
    # ?????? B35_H2S, B35_SO2
    target_cols = ['B35_H2S', 'B35_SO2']
    # ???????
    valid_log_cols = [c for c in target_cols if c in df_raw_test.columns]
    if valid_log_cols:
        print(f"Applying Log Transform to {valid_log_cols}")
        df_raw_test = data_utils.apply_log_transform(df_raw_test, valid_log_cols)

    # [Step 3: Robust Scaling]
    # Note: run_prediction receives 'mean_all' and 'std_all'.
    # Since training script saved Median to 'zscore_mean.csv' and IQR to 'zscore_std.csv',
    # we can use them directly. Ideally we should use apply_robust_scale explicitly.
    # df_z_test = (df_raw_test - mean_all) / std_all (using the passed args)
    
    # We use apply_zscore for clarity if available
    # mean_all here is MEAN, std_all here is STD
    print("Applying Z-score Scaling (Mean/Std)...")
    df_z_test = data_utils.apply_zscore(df_raw_test, mean_all, std_all)

    # [Step 4: Prepare Tensors]
    # Align columns
    test_en_input = df_z_test[en_mv_and_sv].values
    test_de_input = df_z_test[de_mv].values
    test_target = df_z_test[y_sv].values
    
    # Needs at least W steps
    if len(df_z_test) <= W:
        print(f"Data length ({len(df_z_test)}) is not greater than encoder window W ({W}).")
        return

    # Initial History (First W steps)
    initial_history_np = test_en_input[:W]
    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0) # (1, W, F_en)
    
    # Future Inputs (W to End)
    future_de_inputs = torch.tensor(test_de_input[W:], dtype=torch.float32).unsqueeze(0) # (1, H_total, F_de)
    
    # True Targets (W to End) for evaluation
    true_targets_np = test_target[W:]
    
    # Full Encoder Inputs (for Horizon Reinit if needed)
    full_en_inputs = torch.tensor(test_en_input, dtype=torch.float32).unsqueeze(0)

    # --- ??? ---
    strategy = test_cfg.get('inference_strategy', 'sliding_window')
    print(f"?: {strategy}")
    
    if strategy == 'sliding_window':
        # ??
        predictions_tensor = predict_sliding_window(
            model, initial_en_input, future_de_inputs, device, config['data']['num_output']
        )
    elif strategy == 'block_replacement':
        # ??
        predictions_tensor = predict_block_replacement(
            model, initial_en_input, future_de_inputs, device, config
        )
    elif strategy == 'receding_block_replacement':
        predictions_tensor = predict_receding_block_replacement(
            model, initial_en_input, future_de_inputs, device, config
        )
    elif strategy == 'horizon_reinit':
        # Horizon Re-initialization
        predictions_tensor = predict_horizon_reinit(
            model, initial_en_input, future_de_inputs, None, full_en_inputs, device, config
        )
    else:
        print(f"????? {strategy}")
        return

    # --- ???? (Tensor -> Numpy) ---
    predictions_cov = predictions_tensor.cpu().numpy().squeeze(0)
    
    # Align True Targets
    pred_len = predictions_cov.shape[0]
    true_targets_cov = true_targets_np[:pred_len]
    original_index = df_raw_test.index[W : W+pred_len] 

    # --- ??? ---
    metrics_results = []
    
    # 1. Reverse Z-Score for Preds and Targets
    pred_df_z = pd.DataFrame(predictions_cov, columns=y_sv)
    true_df_z = pd.DataFrame(true_targets_cov, columns=y_sv)
    
    # Filter mean/std to only include target variables y_sv
    mean_y = mean_all[y_sv]
    std_y = std_all[y_sv]
    
    pred_df_inv = data_utils.inverse_zscore(pred_df_z, mean_y, std_y)
    true_df_inv = data_utils.inverse_zscore(true_df_z, mean_y, std_y)
    
    # 2. Reverse Log Transform (if applied)
    target_log_cols = [c for c in valid_log_cols if c in y_sv]
    if target_log_cols:
         pred_df_inv = data_utils.inverse_log_transform(pred_df_inv, target_log_cols)
         true_df_inv = data_utils.inverse_log_transform(true_df_inv, target_log_cols)
         
    # Save Metrics
    results_dir = os.path.join(config.get('output', {}).get('results_dir', './results'), config['exp_name'], test_name)
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_list = []
    for i, col in enumerate(y_sv):
        y_true = true_df_inv[col].values
        y_pred = pred_df_inv[col].values
        
        #  np.isfinite ??? NaN ??Inf???Overflow
        # numpy float64 max is ~1.8e308, square is inf. 
        # sklearn MSE might square 1e154 -> overflow. 1e100 is a safe upper bound.
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_pred) < 1e100)
        
        ignored_count = len(y_true) - np.sum(mask)
        if ignored_count > 0:
             print(f"  [WARN] {col}: ignored {ignored_count} NaN/Inf/extreme values.")

        if np.sum(mask) == 0:
             metrics = {"MAE": 0, "RMSE": 0, "R2": 0, "MAPE": 0}
        else:
             metrics = calculate_metrics(y_true[mask], y_pred[mask])
        metrics['Variable'] = col
        metrics_list.append(metrics)
        metrics_results.append(metrics)
        
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"???? {metrics_path}")
    print(metrics_df)
    
    # Save Predictions CSV
    pred_df_inv.index = original_index
    pred_path = os.path.join(results_dir, 'prediction_results.csv')
    pred_df_inv.to_csv(pred_path)
    
    predictions_cov = pred_df_inv.values 
    true_targets_cov = true_df_inv.values 

    # 1. ????? (???)
    # initial_history_np ??Scaled ??(W, Enc_Feat)
    df_hist_scaled = pd.DataFrame(initial_history_np, columns=en_mv_and_sv)
    df_hist = data_utils.inverse_zscore(df_hist_scaled, mean_all, std_all) 
    
    if valid_log_cols:
         df_hist = data_utils.inverse_log_transform(df_hist, valid_log_cols)
    
    # [Modified] Only plot targets H2S and SO2
    target_plot_cols = ['B35_H2S', 'B35_SO2']
    time_series_data = []
    
    for i, name in enumerate(y_sv):
        var_metrics = metrics_results[i]
        
        # --- Plotting Constraint ---
        if name not in target_plot_cols:
            continue
        
        # ??????? (???Encoder Input ?
        if name in df_hist.columns:
            hist_vals = df_hist[name].values
        else:
            hist_vals = np.array([])
            
        future_true = true_targets_cov[:, i]
        future_pred = predictions_cov[:, i]
        time_series_data.append({
            'variable': name,
            'true': future_true.copy(),
            'pred': future_pred.copy(),
            'r2': var_metrics['R2'],
            'rmse': var_metrics['RMSE'],
        })
        
        # ?
        full_true = np.concatenate([hist_vals, future_true])
        # ??????????( Context)?????
        # ?????? History(True)? Pred
        # ??????????
        
        plt.figure(figsize=(20, 6))
        
        # Plot History
        x_hist = range(len(hist_vals))
        plt.plot(x_hist, hist_vals, label='History', color=HISTORY_COLOR, alpha=0.75, linewidth=0.9)
        
        # Plot Future
        x_future = range(len(hist_vals), len(hist_vals) + len(future_true))
        plt.plot(x_future, future_true, label='True (Future)', color=TRUE_COLOR, linewidth=1.0)
        plt.plot(x_future, future_pred, label='Pred (Future)', color=PRED_COLOR, linestyle='--', linewidth=1.0)
        
        # ?????(?History ??Pred ???
        # Connect points for visual continuity
        if len(hist_vals) > 0:
            plt.plot([x_hist[-1], x_future[0]], [hist_vals[-1], future_pred[0]], color=PRED_COLOR, linestyle='--', alpha=0.55, linewidth=0.8)
            plt.plot([x_hist[-1], x_future[0]], [hist_vals[-1], future_true[0]], color=TRUE_COLOR, alpha=0.55, linewidth=0.8)

        # Set Y-Axis Limits based on Valid Data (History + True Future)
        # prevents the plot from being unreadable due to massive outlier predictions (e.g. 1e292)
        valid_plot_data = np.concatenate([hist_vals, future_true])
        valid_plot_data = valid_plot_data[np.isfinite(valid_plot_data)]
        
        if len(valid_plot_data) > 0:
            y_data_min = np.min(valid_plot_data)
            y_data_max = np.max(valid_plot_data)
            y_margin = (y_data_max - y_data_min) * 0.2
            if y_margin == 0: y_margin = 1.0
            plt.ylim(y_data_min - y_margin, y_data_max + y_margin)

        plt.legend()
        style_prediction_axis(plt.gca())
        plt.savefig(os.path.join(results_dir, f'{name}.png'))
        plt.close()

    
    print(f"?: {test_name}. ?: {results_dir}")

    # ==========================================
    # Random 5 Case Studies (72-step Horizon) -> Now Horizon Analysis (1-18)
    # ==========================================
    horizon_result = analyze_horizon_performance(model, df_z_test, config, results_dir, 
                           mean_all, std_all, valid_log_cols,
                           en_mv_and_sv, de_mv, y_sv, W, device)
    return {
        'test_name': test_name,
        'parity_data': horizon_result.get('parity_data', []) if horizon_result else [],
        'time_series_data': time_series_data,
    }

def analyze_horizon_performance(model, df_z, config, results_dir, mean_all, std_all, log_cols,
                           en_cols, de_cols, y_cols, W, device):
    """Run rolling horizon analysis and keep t+1/t+18 parity data for grouped plots."""
    print(f"\n[Horizon Analysis] Evaluating t+1 to t+18 rolling predictions...")
    
    # 1. ? DataLoader (Sliding Window)
    # ?????????????Dataset
    # ?? H  max(prediction_length, 18) ???????
    # ????H=12? 12??
    H_model = config['window']['prediction_length']
    analyze_steps = 18
    
    if H_model < analyze_steps:
        print(f"  Warning: model prediction length ({H_model}) is shorter than {analyze_steps}.")
        print(f"  Analyzing t+1 to t+{H_model} only.")
        analyze_steps = H_model
    
    # ? Dataset
    #  MultiStepS2SDataset ???
    from src.dataset import MultiStepS2SDataset
    
    dataset = MultiStepS2SDataset(
        df_z, 
        en_cols, de_cols, y_cols, 
        W, H_model
    )
    
    # ?? OOM???Batch Size
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    
    all_preds_list = []
    all_targets_list = []
    
    model.eval()
    with torch.no_grad():
        for curr_en, curr_de, curr_target in tqdm(loader, desc="Rolling Prediction"):
            curr_en = curr_en.to(device)
            curr_de = curr_de.to(device)
            
            # Predict
            # output: [Batch, H, F_out]
            out = model(curr_en, curr_de)
            
            all_preds_list.append(out.cpu().numpy())
            # target is [Batch, H, F_out] (Dataset returns slice)
            all_targets_list.append(curr_target.numpy())
            
    # Concatenate
    if len(all_preds_list) == 0:
        print("????? (????)")
        return {'parity_data': []}

    all_preds = np.concatenate(all_preds_list, axis=0)     # (N_samples, H, F_out)
    all_targets = np.concatenate(all_targets_list, axis=0) # (N_samples, H, F_out)
    
    # 2. Inverse Transform Helper
    N, H, F = all_preds.shape
    
    # ? Inverse ??Helper
    def inverse_full(arr_3d):
        # arr_3d: (N, H, F)
        # Reshape to 2D for inverse
        arr_flat = arr_3d.reshape(-1, F)
        df_flat = pd.DataFrame(arr_flat, columns=y_cols)
        
        # Determine mean/std for y_cols
        mu = mean_all[y_cols]
        sigma = std_all[y_cols]
        
        # Inverse Z-score
        df_inv = data_utils.inverse_zscore(df_flat, mu, sigma)
        
        # Inverse Log
        valid_log = [c for c in log_cols if c in y_cols]
        if valid_log:
             df_inv = data_utils.inverse_log_transform(df_inv, valid_log)
             
        return df_inv.values.reshape(N, H, F)

    print("  ????..")
    all_preds_inv = inverse_full(all_preds)
    all_targets_inv = inverse_full(all_targets)
    
    # 3. Collect metrics and selected horizon parity data.
    target_plot_cols = ['B35_H2S', 'B35_SO2'] # Only analyze these
    selected_parity_steps = {1, analyze_steps}
    selected_parity_data = []
    
    print(f"  Collecting horizon metrics and t+1/t+{analyze_steps} parity data...")

    horizon_metrics_rows = []
    
    for t_idx in range(analyze_steps):
        step_num = t_idx + 1
        step_name = f"t+{step_num}"
        
        # Extract data for this step
        # Shape: (N, F)
        preds_t = all_preds_inv[:, t_idx, :]
        targets_t = all_targets_inv[:, t_idx, :]
        
        for v_idx, var_name in enumerate(y_cols):
            if var_name not in target_plot_cols:
                continue
                
            y_p = preds_t[:, v_idx]
            y_t = targets_t[:, v_idx]
            
            # Filter NaN/Inf
            # ???? Plot ??
            mask = np.isfinite(y_p) & np.isfinite(y_t) & (np.abs(y_p) < 1e100)
            y_p = y_p[mask]
            y_t = y_t[mask]
            
            if len(y_p) == 0:
                continue
                
            # Metrics
            metric_values = calculate_metrics(y_t, y_p)
            rmse = metric_values['RMSE']
            r2 = metric_values['R2']

            horizon_metrics_rows.append({
                'Horizon': step_num,
                'Step': step_name,
                'Variable': var_name,
                'MAE': metric_values['MAE'],
                'RMSE': metric_values['RMSE'],
                'R2': metric_values['R2'],
                'MAPE': metric_values['MAPE'],
            })

            if step_num in selected_parity_steps:
                selected_parity_data.append({
                    'step_num': step_num,
                    'step_name': step_name,
                    'variable': var_name,
                    'true': y_t.copy(),
                    'pred': y_p.copy(),
                })

    # Save horizon metrics CSV
    metrics_df = pd.DataFrame(horizon_metrics_rows)
    metrics_path = os.path.join(results_dir, f'horizon_metrics_t1_to_t{analyze_steps}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ???: {metrics_path}")
    print(f"Horizon metrics saved to {metrics_path}")
    return {'parity_data': selected_parity_data}


def main(config_path):
    # --- Step 0: ? ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefix = config['exp_name']
    print(f"========== ?: {prefix} ==========")
    
    # --- ???: ???? ---
    # ?????? zscore_mean.csv (Median) ??zscore_std.csv (IQR)
    zscore_mean_path = os.path.join('./results/', prefix, 'zscore_mean.csv')
    zscore_std_path = os.path.join('./results/', prefix, 'zscore_std.csv')
    
    cfg_data = config['data']
    
    if os.path.exists(zscore_mean_path) and os.path.exists(zscore_std_path):
        print(f"[Init] ????? {zscore_mean_path}")
        # index_col=0 is crucial because saved csv has variable names in first column
        mean_all = pd.read_csv(zscore_mean_path, index_col=0).squeeze()
        std_all = pd.read_csv(zscore_std_path, index_col=0).squeeze()
    else:
        print("[Init] ??????????? (???????...")
        cfg_data = config['data']
        training_file = cfg_data['training_files'][0] if 'training_files' in cfg_data else cfg_data['filename']
        try:
            df_train = data_utils.load_data(os.path.join(cfg_data['path'], training_file))
        except:
            fpath = os.path.join(cfg_data['path'], training_file)
            if not os.path.exists(fpath): # Try local
                 fpath = training_file
            df_train = pd.read_csv(fpath)
            
        df_train.dropna(inplace=True)
        
        # ??????Log Transform ???Stats?
        target_cols = ['B35_H2S', 'B35_SO2']
        valid_log_cols = [c for c in target_cols if c in df_train.columns]
        if valid_log_cols:
             print(f"  Doing Log Transform on {valid_log_cols} before stats calc...")
             df_train = data_utils.apply_log_transform(df_train, valid_log_cols)
             
        # ? Robust Stats (Median/IQR)
        # ????????mean_all/std_all?????Median/IQR
        mean_all, std_all = data_utils.calculate_robust_stats(df_train)
        print("  Recalculated robust statistics.")

    # ??
    de_mv, y_sv, _, en_mv_and_sv = variable_selection(cfg_data['variables_num'])
    
    cfg_win = config['window']
    W = cfg_win['train_window_mins'] // cfg_win['sampling_interval_min']
    
    # --- ? ---
    print(f"\n[Init] Loading model...")
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'
    model = get_model(config)
    model_path = os.path.join('./saved_models/', f'{prefix}.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # --- ???---
    test_suites = config['data'].get('inference_files', [])
    
    # ? Yaml ??inference_files????test_data
    if not test_suites:
        print("???inference_files???test_data")
        default_test = config['data']['test_data']
        default_test['name'] = 'Default_Test_Set'
        test_suites = [default_test]
        
    horizon_runs = []
    for test_cfg in test_suites:
        run_result = run_prediction(config, test_cfg, model, device, mean_all, std_all, en_mv_and_sv, de_mv, y_sv, W)
        if run_result:
            horizon_runs.append(run_result)

    output_root = config.get('output', {}).get('results_dir', './results')
    plot_grouped_horizon_parity(horizon_runs, config, output_root)
    plot_grouped_time_series(horizon_runs, config, output_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)

