import os
import sys
import argparse
import yaml
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

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.models.tabular_mlp import TabularMLP

def generate_steady_state_batch(df_raw, batch_size, de_mv, all_cols, W, std_all):
    idx1 = np.random.choice(len(df_raw) - W - 1, batch_size)

    historical_dfs = []
    ss1_rows = []
    
    for i in idx1:
        hist_w = df_raw.iloc[i : i+W].reset_index(drop=True)
        historical_dfs.append(hist_w)
        ss1_rows.append(hist_w.iloc[-1:])
        
    historical_dfs = historical_dfs * 2
    ss1_p_df = pd.concat(ss1_rows * 2, ignore_index=True)
    ss2_p_df = ss1_p_df.copy()

    target_mvs = [col for col in ['air2_SP', 'HEATER2_output_T_SP'] if col in de_mv]     

    for b in range(batch_size):
        if target_mvs:
            var_to_perturb = target_mvs[b % len(target_mvs)]
        else:
            var_to_perturb = de_mv[b % len(de_mv)]
            
        std_val = std_all[var_to_perturb] if abs(std_all[var_to_perturb]) > 1e-6 else 1.0
        delta = 0.5 * std_val
        
        ss2_p_df.at[b, var_to_perturb] += delta
        ss2_p_df.at[b + batch_size, var_to_perturb] -= delta

    return historical_dfs, ss1_p_df, ss2_p_df

def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp = config['exp_name']
    pretrained_path = './saved_models/transformer_layerwise_71var_decoder_input_sp.pth'
    if os.path.exists(pretrained_path):
        exp = exp + "_Finetuned"
        print(f"[Info] Found pre-trained weights. Automatically renaming experiment to: {exp}")

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
      "acidgas_Fv", "acidgas_T", "acidgas_P", "air2_SP",
      "HEATER2_output_T_SP", "acidgas_CO2", "acidgas_H2O", "acidgas_H2S"
    ]
    tab_target_cols = ["B35_H2S", "B35_SO2"]

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
                
    if len(all_dfs_log) == 0:
        raise ValueError("No training data found to calculate Z-score stats.")
        
    df_all_log = pd.concat(all_dfs_log, ignore_index=True)
    mean_all, std_all = data_utils.calculate_zscore_stats(df_all_log)
    
    # 建立 PGIN 結果資料夾並儲存 stats
    zscore_dir = f'./results/{exp}/'
    os.makedirs(zscore_dir, exist_ok=True)
    mean_all.to_csv(os.path.join(zscore_dir, 'zscore_mean.csv'))
    std_all.to_csv(os.path.join(zscore_dir, 'zscore_std.csv'))
    print(f"  [Save] Z-score stats saved to {zscore_dir}")

    tab_mean_path = './results/Tabular_MLP_Claus_Final/zscore_mean.csv'
    tab_std_path = './results/Tabular_MLP_Claus_Final/zscore_std.csv'
    if os.path.exists(tab_mean_path) and os.path.exists(tab_std_path):
        tab_mean = pd.read_csv(tab_mean_path, index_col=0).squeeze("columns")
        tab_std = pd.read_csv(tab_std_path, index_col=0).squeeze("columns")
        for col in tab_input_cols + tab_target_cols:
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
    dataset_H = H_out * len(config['training'].get('loss_weighting', {}).get('weights', [1]))

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

    print('[Info] Preparing data for Physics-Informed Perturbation...')
    df_raw = df_all_log.copy()
    if 'air_acidgas_ratio' not in df_raw.columns and 'air2_SP' in df_raw.columns and 'acidgas_Fv' in df_raw.columns:
        df_raw['air_acidgas_ratio'] = df_raw['air2_SP'] / df_raw['acidgas_Fv']

    cols_to_check = [c for c in list(set(all_dynamic_cols + tab_input_cols + tab_target_cols)) if c in df_raw.columns]
    df_raw = df_raw.dropna(subset=cols_to_check)

    keep_cols = []
    for c in all_dynamic_cols + tab_input_cols:
        if c not in keep_cols:
            keep_cols.append(c)
    df_raw = df_raw[keep_cols]

    print("[Info] Initializing Transformer Model (with optional Pre-trained check)...")
    dynamic_model = get_model(config).to(device)
    pretrained_path = './saved_models/transformer_layerwise_71var_decoder_input_sp.pth'
    if os.path.exists(pretrained_path):
        print(f"  [Info] Found pre-trained MSE weights: {pretrained_path}")
        print(f"  [Info] Loading weights for Finetuning...")
        dynamic_model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("  [Info] No pre-trained weights found. Training entirely from scratch.")
    dynamic_model.train() 

    print("[Info] Loading Pre-trained TabularMLP Model...")
    target_mean_tensor = torch.tensor(mean_all[tab_target_cols].values, dtype=torch.float32, device=device)
    target_std_tensor = torch.tensor(std_all[tab_target_cols].values, dtype=torch.float32, device=device)
    mlp_model = TabularMLP(num_features=len(tab_input_cols), num_outputs=len(tab_target_cols),
                           hidden_dims=[256, 128, 64], dropout=0.05, activation='gelu',
                           target_mean=target_mean_tensor, target_std=target_std_tensor)
    mlp_path = f'./saved_models/Tabular_MLP_Claus_Final_tabular_mlp.pth'
    if not os.path.exists(mlp_path):
        mlp_path = f'./saved_models/{exp}_tabular_mlp.pth'
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.to(device)
    mlp_model.eval()
    mlp_model.requires_grad_(False)

    epochs = config['training'].get('epochs', 20)
    steps_per_epoch = min(config['training'].get('steps_per_epoch', 50), len(train_loader))
    base_lr = config['training'].get('learning_rate', 1e-4)
    # If using pretrained model, scale down LR to prevent destroying MSE state
    if os.path.exists('./saved_models/transformer_layerwise_71var_decoder_input_sp.pth'):
        base_lr = 1e-5
        print(f"  [Info] Auto-scaled Learning Rate for Fine-tuning: {base_lr}")

    optimizer = optim.Adam(dynamic_model.parameters(), lr=base_lr)
    
    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    H_ss = 100 
    warmup_steps = 100 

    best_loss = float('inf')
    early_stop_patience = config['training'].get('patience', 10)
    epochs_no_improve = 0

    y_mean_tensor = torch.tensor(mean_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(std_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_safe = torch.where(torch.abs(y_std_tensor) < 1e-6, torch.ones_like(y_std_tensor), y_std_tensor)

    target_std_safe = torch.where(torch.abs(target_std_tensor) < 1e-6, torch.ones_like(target_std_tensor), target_std_tensor)

    history_losses = []
    history_val_losses = []
    history_kci = [] 

    from src.engine import step_wise_rolling_at_loss_step
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        dynamic_model.train()
        epoch_gain_loss = 0.0
        epoch_correct_dir = 0
        epoch_total_eval = 0
        epoch_mse_loss = 0.0
        
        step_limit = min(steps_per_epoch, len(train_loader))
        pbar = tqdm(enumerate(train_loader), total=step_limit, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, mse_batch in pbar:
            if step >= step_limit:
                break
                
            optimizer.zero_grad()

            mse_loss_val = step_wise_rolling_at_loss_step(dynamic_model, mse_batch, criterion, device, config)

            ss_batch_size = 4
            historical_dfs, ss1_df, ss2_df = generate_steady_state_batch(df_raw, ss_batch_size, de_mv, keep_cols, W, std_all)
            B_actual = len(historical_dfs)

            with contextlib.redirect_stdout(io.StringIO()):
                x_en_z_list = []
                for b_df in historical_dfs:
                    b_z = data_utils.apply_zscore(b_df, mean_all, std_all).fillna(0.0)
                    x_en_z_list.append(torch.tensor(b_z[en_mv_and_sv].values, dtype=torch.float32, device=device))
                x_en_z_history = torch.stack(x_en_z_list) 
                
                ss1_z_df = data_utils.apply_zscore(ss1_df, mean_all, std_all).fillna(0.0)
                ss2_z_df = data_utils.apply_zscore(ss2_df, mean_all, std_all).fillna(0.0)

            ss1_de_p = torch.tensor(ss1_df[de_mv].values, dtype=torch.float32, device=device)
            ss2_de_p = torch.tensor(ss2_df[de_mv].values, dtype=torch.float32, device=device)

            delta_mv = ss2_de_p - ss1_de_p
            is_perturbed = torch.abs(delta_mv) > 1e-5
            delta_mv_safe = torch.where(is_perturbed, delta_mv, torch.sign(delta_mv) * 1e-6 + 1e-6)

            mlp_x_z_ss1 = torch.tensor(ss1_z_df[tab_input_cols].values, dtype=torch.float32, device=device)
            mlp_x_z_ss2 = mlp_x_z_ss1.clone()
            
            for col in de_mv:
                if col in tab_input_cols:
                    col_idx_in_tab = tab_input_cols.index(col)
                    col_idx_in_de = de_mv.index(col)
                    m = mean_all.get(col, 0.0)
                    s_val = std_all.get(col, 1.0)
                    s = s_val if abs(s_val) > 1e-6 else 1.0
                    mlp_x_z_ss2[:, col_idx_in_tab] = (ss2_de_p[:, col_idx_in_de] - m) / s

            with torch.no_grad():
                y_mlp_z_ss1 = mlp_model(mlp_x_z_ss1)
                y_mlp_p_ss1 = y_mlp_z_ss1 * target_std_safe + target_mean_tensor
                y_mlp_z_ss2 = mlp_model(mlp_x_z_ss2)
                y_mlp_p_ss2 = y_mlp_z_ss2 * target_std_safe + target_mean_tensor

            delta_y_mlp = y_mlp_p_ss2 - y_mlp_p_ss1 
            K_ss_matrix = delta_y_mlp.unsqueeze(2) / delta_mv_safe.unsqueeze(1)
            K_ss_direction = torch.sign(K_ss_matrix)

            ss1_en_z = torch.tensor(ss1_z_df[en_mv_and_sv].values, dtype=torch.float32, device=device)
            ss1_de_z = torch.tensor(ss1_z_df[de_mv].values, dtype=torch.float32, device=device)

            current_en_history = x_en_z_history.clone()
            steady_state_preds_ss1 = []
            all_preds_ss1_plot = []

            with torch.no_grad():
                for t in range(warmup_steps):
                    step_de_input = ss1_de_z.unsqueeze(1)
                    pred_z = dynamic_model(current_en_history, step_de_input)
                    pred_p_warmup = pred_z.squeeze(1) * y_std_safe + y_mean_tensor

                    log_cols_inv = [c for c in ['B35_H2S', 'B35_SO2'] if c in y_sv]
                    log_target_idx = [y_sv.index(c) for c in log_cols_inv]
                    if len(log_target_idx) > 0:
                        pred_p_warmup[:, log_target_idx] = torch.exp(pred_p_warmup[:, log_target_idx])

                    tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                    pred_p_warmup[:, tab_target_idx] = torch.clamp(pred_p_warmup[:, tab_target_idx], min=1e-6)

                    pred_p_targets = pred_p_warmup[:, tab_target_idx]
                    if step == 0:
                        all_preds_ss1_plot.append(pred_p_targets)

                    if t >= warmup_steps - 40:
                        steady_state_preds_ss1.append(pred_p_targets)

                    new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                    for c_idx, c_name in enumerate(en_mv_and_sv):
                        if c_name in de_mv:
                            new_step_features[:, 0, c_idx] = ss1_de_z[:, de_mv.index(c_name)]
                        elif c_name in y_sv:
                            new_step_features[:, 0, c_idx] = pred_z[:, 0, y_sv.index(c_name)].detach()
                        else:
                            new_step_features[:, 0, c_idx] = ss1_en_z[:, c_idx]

                    current_en_history = torch.cat([current_en_history[:, 1:, :], new_step_features], dim=1)

            y_dyn_ss1 = torch.stack(steady_state_preds_ss1).mean(dim=0)
            current_en_history = current_en_history.detach() 

            ss2_de_z_graph = torch.zeros(B_actual, len(de_mv), device=device)
            for col_idx_in_de, col in enumerate(de_mv):
                m = mean_all.get(col, 0.0)
                s_val = std_all.get(col, 1.0)
                s = s_val if abs(s_val) > 1e-6 else 1.0
                ss2_de_z_graph[:, col_idx_in_de] = (ss2_de_p[:, col_idx_in_de] - m) / s

            ss2_en_z_const = torch.tensor(ss2_z_df[en_mv_and_sv].values, dtype=torch.float32, device=device)
            steady_state_preds = []
            all_preds_ss2_plot = []

            for t in range(H_ss):
                step_de_input = ss2_de_z_graph.unsqueeze(1)
                pred_z = dynamic_model(current_en_history, step_de_input)
                pred_p = pred_z.squeeze(1) * y_std_safe + y_mean_tensor

                if len(log_target_idx) > 0:
                    pred_p[:, log_target_idx] = torch.exp(pred_p[:, log_target_idx])

                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                pred_p[:, tab_target_idx] = torch.clamp(pred_p[:, tab_target_idx], min=1e-6)

                pred_p_targets = pred_p[:, tab_target_idx]
                if step == 0:
                    all_preds_ss2_plot.append(pred_p_targets)

                if t >= 60:
                    steady_state_preds.append(pred_p_targets)

                new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                for c_idx, c_name in enumerate(en_mv_and_sv):
                    if c_name in de_mv:
                        new_step_features[:, 0, c_idx] = ss2_de_z_graph[:, de_mv.index(c_name)]
                    elif c_name in y_sv:
                        new_step_features[:, 0, c_idx] = pred_z[:, 0, y_sv.index(c_name)].detach()
                    else:
                        new_step_features[:, 0, c_idx] = ss2_en_z_const[:, c_idx]

                current_en_history = torch.cat([current_en_history[:, 1:, :], new_step_features], dim=1)

            if step == 0:
                t_ss1_full = torch.stack(all_preds_ss1_plot, dim=1).detach().cpu().numpy() # shape (B, T1, num_targets)
                t_ss2_full = torch.stack(all_preds_ss2_plot, dim=1).detach().cpu().numpy() # shape (B, T2, num_targets)
                
                timeline_full = np.concatenate([t_ss1_full, t_ss2_full], axis=1) # shape (B, T1+T2, num_targets)
                total_steps = timeline_full.shape[1]
                t_change_idx = t_ss1_full.shape[1]

                fig, axes = plt.subplots(len(tab_target_cols), 1, figsize=(10, 4*len(tab_target_cols)))
                if len(tab_target_cols) == 1: axes = [axes]
                
                for tgt_idx, tgt_name in enumerate(tab_target_cols):
                    axes[tgt_idx].plot(range(total_steps), timeline_full[0, :, tgt_idx], label=f'Model Prediction trajectory', color='purple', linewidth=2)
                    axes[tgt_idx].axvline(x=t_change_idx, color='red', linestyle='--', linewidth=1.5, label='Perturb MV (+0.5σ)')
                    axes[tgt_idx].text(t_change_idx + 1, timeline_full[0, :, tgt_idx].min(), "Step Change!", color='red', verticalalignment='bottom')

                    axes[tgt_idx].set_title(f"Continuous Step-Change Rollout Ep{epoch}_Step{step} - {tgt_name}")
                    axes[tgt_idx].set_xlabel("Future Time Steps")
                    axes[tgt_idx].legend()
                    axes[tgt_idx].grid(True)
                
                os.makedirs('./results/PGIN_Visualizations', exist_ok=True)
                plt.tight_layout()
                plt.savefig(f'./results/PGIN_Visualizations/train_rollout_ep{epoch}_step{step}.png')
                plt.close()
                print(f"\n  [Plot] Saved continuous step-change visualization at Ep{epoch} Step{step}!")

            steady_state_stack = torch.stack(steady_state_preds) 
            delta_y_dyn_stack = steady_state_stack - y_dyn_ss1.unsqueeze(0) 

            K_dyn_matrix_stack = delta_y_dyn_stack.unsqueeze(3) / delta_mv_safe.unsqueeze(0).unsqueeze(2)
            K_ss_direction_exp = K_ss_direction.unsqueeze(0) 
            
            loss_matrix_stack = torch.nn.functional.relu(-K_dyn_matrix_stack * K_ss_direction_exp)

            is_perturbed_mask = is_perturbed.unsqueeze(1).expand(-1, loss_matrix_stack.size(2), -1) 
            is_perturbed_mask_stack = is_perturbed_mask.unsqueeze(0).expand(loss_matrix_stack.size(0), -1, -1, -1) 
            
            valid_mlp_mask = (torch.abs(delta_y_mlp) >= 1e-5)
            valid_mlp_mask_expanded = valid_mlp_mask.unsqueeze(0).unsqueeze(3).expand(loss_matrix_stack.size(0), -1, -1, loss_matrix_stack.size(3))

            final_mask = is_perturbed_mask_stack & valid_mlp_mask_expanded
            correct_mask = (K_dyn_matrix_stack * K_ss_direction_exp > 0) & final_mask
            
            valid_items = final_mask.sum().item()
            epoch_correct_dir += correct_mask.sum().item()
            epoch_total_eval += valid_items

            if valid_items > 0:
                gain_loss_weight = config['training'].get('pgin_loss_weight', 0.3)
                loss_gain = torch.mean(loss_matrix_stack[final_mask]) * gain_loss_weight
            else:
                loss_gain = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss = mse_loss_val + loss_gain
            epoch_mse_loss += mse_loss_val.item()

            total_norm_val = 0.0
            if total_loss.item() > 0:
                total_loss.backward()
                # clip_grad_norm_ 會回傳被截斷之前的原始總梯度大小 (unclipped grad norm)
                total_norm = torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), 1.0)
                total_norm_val = total_norm.item()
                optimizer.step()

            epoch_gain_loss += loss_gain.item()
            pbar.set_postfix({
                'MSE': f"{mse_loss_val.item():.4f}", 
                'Gain': f"{loss_gain.item():.8f}",
                'GradNorm(Pre-Clip)': f"{total_norm_val:.2f}"
            })

        avg_loss = (epoch_gain_loss + epoch_mse_loss) / step_limit
        epoch_kci = epoch_correct_dir / epoch_total_eval if epoch_total_eval > 0 else 1.0
        print(f"Epoch [{epoch+1}/{epochs}] | Train Total L: {avg_loss:.6f} | KCI: {epoch_kci*100:.2f}%")
        history_losses.append(avg_loss)
        history_kci.append(epoch_kci)

        # Validation Phase
        dynamic_model.eval()
        val_mse_loss = 0.0
        if valid_loader is not None:
            with torch.no_grad():
                for mse_batch in valid_loader:
                    v_loss = step_wise_rolling_at_loss_step(dynamic_model, mse_batch, criterion, device, config)
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
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), history_losses, marker='o', color='purple', label='Train Total', linewidth=2)
    if len(history_val_losses) > 0 and valid_loader is not None:
        plt.plot(range(1, actual_epochs + 1), history_val_losses, marker='x', color='blue', label='Valid MSE', linewidth=2)
    plt.title(f'Training & Validation Loss - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    kci_percentages = [kci * 100 for kci in history_kci]
    plt.plot(range(1, actual_epochs + 1), kci_percentages, marker='s', color='orange', linewidth=2)
    plt.title(f'PGIN KCI Consistency - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Consistent Steps (%)')
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'training_loss_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[Done] Model training complete. Weights securely saved to {out_model_path}")
    print(f"[Info] Training metrics exported to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)


