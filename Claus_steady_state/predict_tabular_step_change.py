# predict_tabular_step_change.py
import os
import sys
import glob
import re

# Add the project root to sys.path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models.tabular_mlp import TabularMLP
from src.models import get_model
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict_step_change import process_single_file
from src.utils import calculate_metrics

def build_input_target_cols(en_mv_and_sv, y_sv):
    input_cols = [c for c in en_mv_and_sv if c not in y_sv]
    return input_cols, list(y_sv)

def parse_scenario_conditions(filename):
    """從檔名解析操作條件，例如 air2_400_t2_190_air2_change_10_converted.csv"""
    name = os.path.basename(filename).replace('.csv', '').replace('_converted', '')
    m = re.match(r'air2_(-?\d+)_t2_(-?\d+)_(\w+)_change_(-?\d+)', name)
    if m:
        return {
            'air2': int(m.group(1)),
            't2':   int(m.group(2)),
            'change_var': m.group(3),
            'change_val': int(m.group(4)),
        }
    return {'air2': '?', 't2': '?', 'change_var': '?', 'change_val': '?'}

def run_step_change_prediction(model, config, input_cols, target_cols, mean_all, std_all, device, csv_path):
    df_step = pd.read_csv(csv_path)

    # Convert components to fractions to match training data format (where they are fractions of total)
    acid_cols = ['acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S']
    has_acid_cols = all(col in df_step.columns for col in acid_cols)
    if has_acid_cols:
        total_acid = df_step['acidgas_CO2'] + df_step['acidgas_H2O'] + df_step['acidgas_H2S']
        # 只有當它們大於 2.0 時(代表是實際流量而非 fraction) 才進行轉換
        if total_acid.mean() > 2.0:
            for col in acid_cols:
                df_step[col] = df_step[col] / total_acid

    # 處理特徵名稱差異
    if 'air2_SP_m3' in input_cols and 'air2_SP_m3' not in df_step.columns and 'air2_SP' in df_step.columns:
        df_step['air2_SP_m3'] = df_step['air2_SP']
    
    # Check if necessary columns exist
    missing_cols = [col for col in input_cols + target_cols if col not in df_step.columns]
    if missing_cols:
        print(f"[Warn] Missing columns {missing_cols} in {csv_path}, skipping.")
        return None

    # Z-score scaling using training data statistics
    std_safe = std_all.copy()
    std_safe[std_safe < 1e-6] = 1.0
    
    # 進行 Z-score 標準化
    df_z_step = df_step.copy()
    for col in input_cols + target_cols:
        df_z_step[col] = (df_step[col] - mean_all[col]) / std_safe[col]

    X = torch.tensor(df_z_step[input_cols].values.astype(np.float32), device=device)
    
    with torch.no_grad():
        y_pred_z = model(X).cpu().numpy()

    # Inverse Z-score
    y_mean = mean_all[target_cols].values
    y_std = std_all[target_cols].values
    y_std_safe = np.where(np.abs(y_std) < 1e-6, 1.0, y_std)
    
    predictions_cov = y_pred_z * y_std_safe + y_mean
    true_targets_cov = df_step[target_cols].values
    
    res_dict = {
        'filename': os.path.basename(csv_path),
        'predictions': predictions_cov,
        'true_values': true_targets_cov,
        'y_sv': target_cols,
        'num_steps': len(df_step)
    }
    return res_dict

def main(config_path: str, tf_config_path: str, tf_weights_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    exp = config['exp_name']

    mean_path = f'./results/{exp}/zscore_mean.csv'
    std_path = f'./results/{exp}/zscore_std.csv'
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean_all = pd.read_csv(mean_path, index_col=0).squeeze("columns")
        std_all = pd.read_csv(std_path, index_col=0).squeeze("columns")
    else:
        raise FileNotFoundError(f"Preprocessing stats not found at {mean_path} or {std_path}.")

    tab_cfg = config.get('tabular', {})
    if tab_cfg and 'input_cols' in tab_cfg and 'target_cols' in tab_cfg:
        input_cols = list(tab_cfg['input_cols'])
        target_cols = list(tab_cfg['target_cols'])
    else:
        de_mv, y_sv, _, en_mv_and_sv = variable_selection(cfg_data['variables_num'])
        input_cols, target_cols = build_input_target_cols(en_mv_and_sv, y_sv)

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

    # --- Load Transformer Model ---
    tf_mean = None
    tf_std = None
    tf_model = None
    tf_config = None
    if tf_config_path and tf_weights_path:
        with open(tf_config_path, 'r', encoding='utf-8') as f:
            tf_config = yaml.safe_load(f)
        tf_exp = tf_config['exp_name']
        tf_mean_path = f'./results/{tf_exp}/zscore_mean.csv'
        tf_std_path = f'./results/{tf_exp}/zscore_std.csv'
        tf_mean = pd.read_csv(tf_mean_path, index_col=0).squeeze("columns")
        tf_std = pd.read_csv(tf_std_path, index_col=0).squeeze("columns")
        
        tf_de_mv, tf_y_sv, _, tf_en_mv_and_sv = variable_selection(tf_config['data']['variables_num'])
        tf_config['data']['num_en_input'] = len(tf_en_mv_and_sv)
        tf_config['data']['num_de_input'] = len(tf_de_mv)
        tf_config['data']['num_output'] = len(tf_y_sv)
        
        tf_model = get_model(tf_config)
        tf_model.load_state_dict(torch.load(tf_weights_path, map_location=device))
        tf_model.to(device)
        tf_model.eval()
        tf_W = tf_config['window']['train_window_mins'] // tf_config['window']['sampling_interval_min']
        tf_H = tf_config['window']['prediction_length']
        tf_strategy = 'sliding_window'

    step_change_dir = os.path.join("data", "Claus_dynamic", "step_change")
    csv_files = sorted(glob.glob(os.path.join(step_change_dir, "**", "*_converted.csv"), recursive=True))

    if not csv_files:
        print(f"No _converted.csv files found in {step_change_dir}")
        return

    from collections import defaultdict
    grouped_files = defaultdict(list)
    for csv_path in csv_files:
        subfolder = os.path.basename(os.path.dirname(csv_path))
        grouped_files[subfolder].append(csv_path)

    for folder_name, files in grouped_files.items():
        results = []
        for csv_path in files:
            print(f"\n--- [{folder_name}] {os.path.basename(csv_path)} ---")
            res = run_step_change_prediction(model, config, input_cols, target_cols, mean_all, std_all, device, csv_path)
            
            tf_res = None
            if tf_model is not None:
                tf_res = process_single_file(csv_path, tf_model, tf_config, tf_mean, tf_std, device, tf_de_mv, tf_y_sv, tf_en_mv_and_sv, tf_W, tf_H, tf_strategy, warmup_steps=100)
            
            if res:
                idx = target_cols.index('B35_H2S')
                true_h2s = res['true_values'][:, idx]
                pred_h2s = res['predictions'][:, idx]
                res['tf_predictions'] = tf_res['predictions'] if tf_res else None
                print(f"H2S True range: [{true_h2s.min():.5f}, {true_h2s.max():.5f}] => change: {true_h2s.max()-true_h2s.min():.5f}")
                print(f"H2S Tab Pred: [{pred_h2s.min():.5f}, {pred_h2s.max():.5f}] => change: {pred_h2s.max()-pred_h2s.min():.5f}")
                if tf_res:
                    tf_pred_h2s = res['tf_predictions'][:, tf_res['y_sv'].index('B35_H2S')]
                    print(f"H2S TF  Pred: [{tf_pred_h2s.min():.5f}, {tf_pred_h2s.max():.5f}] => change: {tf_pred_h2s.max()-tf_pred_h2s.min():.5f}")
                results.append(res)
        
        # Plotting similar to dynamic predict_step_change
        for i in range(0, len(results), 4):
            batch_results = results[i:i+4]
            fig, axes = plt.subplots(4, 2, figsize=(16, 16))
            fig.suptitle(f"Tabular MLP Step-Change: {folder_name} (Part {i//4 + 1})", fontsize=14)
            axes = axes.flatten()

            for j, res in enumerate(batch_results):
                cond = parse_scenario_conditions(res['filename'])
                c_val = cond['change_val']
                val_str = f"+{c_val}" if isinstance(c_val, (int, float)) and c_val > 0 else str(c_val)
                title_str = f"[air2={cond['air2']} t2={cond['t2']} Δ{cond['change_var']}={val_str}]"
                
                ax_h2s = axes[j*2] if j*2 < len(axes) else None
                ax_so2 = axes[j*2 + 1] if j*2 + 1 < len(axes) else None

                if ax_h2s:
                    idx = target_cols.index('B35_H2S')
                    ax_h2s.plot(res['true_values'][:, idx], label="True (Aspen)", color='steelblue')
                    
                    tab_init = res['predictions'][0, idx]
                    tab_final = res['predictions'][-1, idx]
                    tab_dir = "Up" if tab_final - tab_init > 1e-4 else ("Down" if tab_final - tab_init < -1e-4 else "Flat")
                    
                    ax2_h2s = ax_h2s.twinx()
                    
                    tf_lines = []
                    tf_labels = []
                    
                    if res.get('tf_predictions') is not None:
                        tf_idx = tf_y_sv.index('B35_H2S')
                        tf_preds = res['tf_predictions'][:, tf_idx]
                        interval = max(1, len(res['true_values']) // len(tf_preds))
                        tf_x = [i * interval + (interval - 1) for i in range(len(tf_preds))]
                        line = ax2_h2s.plot(tf_x, tf_preds, label="TF Pred", color='tomato', linestyle='--')
                        tf_lines.extend(line)
                        tf_labels.append("TF Pred")
                    
                    title_color = 'red' if tab_dir == "Flat" else 'black'
                    ax_h2s.set_title(f"{title_str}  Tabular: {tab_dir}\nB35_H2S", fontsize=10, color=title_color)
                    ax_h2s.set_ylabel("B35_H2S (True)", color='steelblue')
                    ax_h2s.tick_params(axis='y', labelcolor='steelblue', color='steelblue')
                    ax_h2s.spines['left'].set_color('steelblue')

                    ax2_h2s.set_ylabel("TF Predicted", color='tomato')
                    ax2_h2s.tick_params(axis='y', labelcolor='tomato', color='tomato')
                    ax2_h2s.spines['right'].set_color('tomato')
                    
                    lines1, labels1 = ax_h2s.get_legend_handles_labels()
                    ax2_h2s.legend(lines1 + tf_lines, labels1 + tf_labels, loc='upper right')
                    
                    ax_h2s.grid(True, alpha=0.3)

                if ax_so2:
                    idx = target_cols.index('B35_SO2')
                    ax_so2.plot(res['true_values'][:, idx], label="True (Aspen)", color='steelblue')
                    
                    tab_init = res['predictions'][0, idx]
                    tab_final = res['predictions'][-1, idx]
                    tab_dir = "Up" if tab_final - tab_init > 1e-4 else ("Down" if tab_final - tab_init < -1e-4 else "Flat")
                    
                    ax2_so2 = ax_so2.twinx()
                    
                    tf_lines2 = []
                    tf_labels2 = []
                    
                    if res.get('tf_predictions') is not None:
                        tf_idx = tf_y_sv.index('B35_SO2')
                        tf_preds = res['tf_predictions'][:, tf_idx]
                        interval = max(1, len(res['true_values']) // len(tf_preds))
                        tf_x = [i * interval + (interval - 1) for i in range(len(tf_preds))]
                        line2 = ax2_so2.plot(tf_x, tf_preds, label="TF Pred", color='tomato', linestyle='--')
                        tf_lines2.extend(line2)
                        tf_labels2.append("TF Pred")
                        
                    title_color2 = 'red' if tab_dir == "Flat" else 'black'
                    ax_so2.set_title(f"{title_str}  Tabular: {tab_dir}\nB35_SO2", fontsize=10, color=title_color2)
                    ax_so2.set_ylabel("B35_SO2 (True)", color='steelblue')
                    ax_so2.tick_params(axis='y', labelcolor='steelblue', color='steelblue')
                    ax_so2.spines['left'].set_color('steelblue')

                    ax2_so2.set_ylabel("TF Predicted", color='tomato')
                    ax2_so2.tick_params(axis='y', labelcolor='tomato', color='tomato')
                    ax2_so2.spines['right'].set_color('tomato')
                    
                    lines3, labels3 = ax_so2.get_legend_handles_labels()
                    ax2_so2.legend(lines3 + tf_lines2, labels3 + tf_labels2, loc='upper right')
                    
                    ax_so2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            out_path = f"results/{exp}/{folder_name}/tabular_step_change_part{i//4 + 1}.png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to {out_path}")
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/tabular_mlp_claus.yaml')
    parser.add_argument('--tf-config', type=str, default='configs/transformer_layerwise_71var_decoder_input_sp.yaml')
    parser.add_argument('--tf-weights', type=str, default='saved_models/transformer_layerwise_71var_decoder_input_sp_PGIN_Finetuned.pth')
    args = parser.parse_args()
    main(args.config, args.tf_config, args.tf_weights)
