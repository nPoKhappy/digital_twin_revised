# finetune_gain_inform.py - PGIN Finetuning for Transformer using Steady-State Gain from TabularMLP
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

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.models.tabular_mlp import TabularMLP

# ==============================================================================
# Helper Functions
# ==============================================================================
# def compute_jacobian(y, x, create_graph=False):
#     """
#     Compute the Jacobian matrix of y with respect to x (dy_i / dx_j).
#     Args:
#         y: Tensor of shape (Batch, N_y)
#         x: Tensor of shape (Batch, N_x) (Requires requires_grad=True)
#         create_graph: If True, graph of the derivative will be constructed, allowing to compute higher order derivative products.
#     Returns:
#         jacobian: Tensor of shape (Batch, N_y, N_x)
#     """
#     B, N_y = y.shape
#     _, N_x = x.shape
#     jacobian = torch.zeros(B, N_y, N_x, device=y.device)

#     for i in range(N_y):
#         grad_outputs = torch.ones_like(y[:, i])
#         grad = torch.autograd.grad(
#             outputs=y[:, i],
#             inputs=x,
#             grad_outputs=grad_outputs,
#             create_graph=create_graph, # Allow double differentiation for loss backward
#             retain_graph=True,
#             only_inputs=True
#         )[0]
#         jacobian[:, i, :] = grad

#     return jacobian

def generate_steady_state_batch(df_raw, batch_size, de_mv, all_cols, W, std_all):
    """
    Generate steady state simulation batches by sampling sequences.
    - historical_dfs: A list of DataFrames of length W representing the history before the step change.
    - SS1: The last row of historical_dfs.
    - SS2: Perform step change (+/-) symmetrically to generate pairs to prevent gradient drift.
    """
    idx1 = np.random.choice(len(df_raw) - W - 1, batch_size)

    historical_dfs = []
    ss1_rows = []
    
    for i in idx1:
        hist_w = df_raw.iloc[i : i+W].reset_index(drop=True)
        historical_dfs.append(hist_w)
        ss1_rows.append(hist_w.iloc[-1:])
        
    # Duplicate for symmetry: first half gets +dx, second half gets -dx on the EXACT same starting points.
    historical_dfs = historical_dfs * 2
    ss1_p_df = pd.concat(ss1_rows * 2, ignore_index=True)
    ss2_p_df = ss1_p_df.copy()

    # 根據您的需求，只允許這兩個變數發生獨立的 Step Change
    target_mvs = [col for col in ['air2_SP', 'HEATER2_output_T_SP'] if col in de_mv]     

    for b in range(batch_size):
        if target_mvs:
            # 依序平均分配，確保每個 batch 都能獨立擾動到這兩個變數
            var_to_perturb = target_mvs[b % len(target_mvs)]
        else:
            var_to_perturb = de_mv[b % len(de_mv)]
            
        std_val = std_all[var_to_perturb] if abs(std_all[var_to_perturb]) > 1e-6 else 1.0
        delta = 0.5 * std_val
        
        # Positive perturbation
        ss2_p_df.at[b, var_to_perturb] += delta
        # Negative perturbation
        ss2_p_df.at[b + batch_size, var_to_perturb] -= delta

    return historical_dfs, ss1_p_df, ss2_p_df

# ==============================================================================
# Main Finetuning Loop
# ==============================================================================
def main(config_path: str):
    # 1. Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    exp = config['exp_name']
    device = 'cuda' if torch.cuda.is_available() and config['training']['device'] == 'cuda' else 'cpu'

    print("=" * 70)
    print(f"Physics-Informed Gain Finetuning - Session: {exp}")
    print("=" * 70)

    # Load variable sizes
    cfg_data = config['data']
    de_mv, y_sv, non_used, en_mv_and_sv = variable_selection(cfg_data['variables_num'])

    # Deduplicate dynamic cols to avoid pandas reindex issues on z-score processing
    all_dynamic_cols = []
    for col in en_mv_and_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)
    for col in y_sv:
        if col not in all_dynamic_cols:
            all_dynamic_cols.append(col)

    # Enforce config dict numbers for the transformer get_model
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)

    # Setup TabularMLP input and target columns
    tab_input_cols = [
      "acidgas_Fv", "acidgas_T", "acidgas_P", "air2_SP",
      "HEATER2_output_T_SP", "acidgas_CO2", "acidgas_H2O", "acidgas_H2S"
    ]
    tab_target_cols = ["B35_H2S", "B35_SO2"]

    # Mapping indices for dynamic and TabularMLP
    col_to_dyn_idx = {col: i for i, col in enumerate(all_dynamic_cols)}
    de_mv_idx = [col_to_dyn_idx[col] for col in de_mv]
    y_sv_idx = [col_to_dyn_idx[col] for col in y_sv]
    col_to_tab_idx = {col: i for i, col in enumerate(all_dynamic_cols) if col in tab_input_cols}

    # 2. Load Z-score mean and std for un-normalizing Jacobians
    mean_path = f'./results/{exp}/zscore_mean.csv'
    std_path = f'./results/{exp}/zscore_std.csv'
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing normalization files in ./results/{exp}/")

    mean_all = pd.read_csv(mean_path, index_col=0).squeeze("columns")
    std_all = pd.read_csv(std_path, index_col=0).squeeze("columns")

    # Inherit standard tabular MLP normalization parameters for inputs
    tab_mean_path = './results/Tabular_MLP_Claus_Final/zscore_mean.csv'
    tab_std_path = './results/Tabular_MLP_Claus_Final/zscore_std.csv'
    if os.path.exists(tab_mean_path) and os.path.exists(tab_std_path):
        tab_mean = pd.read_csv(tab_mean_path, index_col=0).squeeze("columns")
        tab_std = pd.read_csv(tab_std_path, index_col=0).squeeze("columns")
        for col in tab_input_cols + tab_target_cols:
            if col not in mean_all.index and col in tab_mean.index:
                mean_all[col] = tab_mean[col]
                std_all[col] = tab_std[col]

    # 3. Load training data for steady-state samples
    df_raw_list = []
    if 'training_files' in cfg_data and cfg_data['training_files']:
        for fname in cfg_data['training_files']:
            fpath = os.path.join(cfg_data['path'], fname)
            if os.path.exists(fpath):
                print(f"[Info] Loading history data from: {fpath}")
                df_curr = data_utils.load_data(fpath)
                df_raw_list.append(df_curr)
    elif 'filename' in cfg_data:
        fpath = os.path.join(cfg_data['path'], cfg_data['filename'])
        print(f"[Info] Loading history data from: {fpath}")
        df_curr = data_utils.load_data(fpath)
        if 'point' in cfg_data:
            df_curr = df_curr.iloc[:cfg_data['point']]
        df_raw_list.append(df_curr)

    # Add data from step_change directory to expose the model to out-of-distribution dynamic changes
    step_change_dir = os.path.join(cfg_data['path'], 'step_change')
    if os.path.exists(step_change_dir):
        for root, dirs, files in os.walk(step_change_dir):
            for file in files:
                if file.endswith('.csv'):
                    fpath = os.path.join(root, file)
                    print(f"[Info] Loading step change history data from: {fpath}")
                    df_curr = pd.read_csv(fpath)
                    df_raw_list.append(df_curr)

    # NOTE: We only base our SS1 samples on the original training distribution and step changes.
    print(f"[Info] Total number of source files pooled: {len(df_raw_list)}")

    df_raw = pd.concat(df_raw_list, ignore_index=True)

    if 'air_acidgas_ratio' not in df_raw.columns and 'air2_SP' in df_raw.columns and 'acidgas_Fv' in df_raw.columns:
        df_raw['air_acidgas_ratio'] = df_raw['air2_SP'] / df_raw['acidgas_Fv']

    cols_to_check = [c for c in list(set(all_dynamic_cols + tab_input_cols + tab_target_cols)) if c in df_raw.columns]
    df_raw = df_raw.dropna(subset=cols_to_check)

    # Extract required features list combining both models
    keep_cols = []
    for c in all_dynamic_cols + tab_input_cols:
        if c not in keep_cols:
            keep_cols.append(c)

    df_raw = df_raw[keep_cols]

    # 4. Model Loading Strategy
    # Transformer Setup
    print("[Info] Loading Pre-trained Transformer Model...")
    dynamic_model = get_model(config).to(device)
    dynamic_path = f'./saved_models/{exp}.pth'
    dynamic_model.load_state_dict(torch.load(dynamic_path, map_location=device))
    dynamic_model.eval() # Use eval mode to prevent compound dropout noise during autoregressive BPTT

    # Tabular MLP Setup (Ground Truth Proxy)
    print(f"[Info] Loading Pre-trained TabularMLP Model...")
    
    target_mean_tensor = torch.tensor(mean_all[tab_target_cols].values, dtype=torch.float32, device=device)
    target_std_tensor = torch.tensor(std_all[tab_target_cols].values, dtype=torch.float32, device=device)
        
    mlp_model = TabularMLP(num_features=len(tab_input_cols),
                           num_outputs=len(tab_target_cols),
                           hidden_dims=[256, 128, 64],
                           dropout=0.05,
                           activation='gelu',
                           target_mean=target_mean_tensor,
                           target_std=target_std_tensor)
    mlp_path = f'./saved_models/Tabular_MLP_Claus_Final_tabular_mlp.pth' # Fixed MLP Path
    if not os.path.exists(mlp_path):
        print(f"[Warning] Path {mlp_path} not found. Attempting {exp}_tabular_mlp.pth")
        mlp_path = f'./saved_models/{exp}_tabular_mlp.pth'

    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.to(device)
    mlp_model.eval() # Freeze MLP model in evaluation mode
    mlp_model.requires_grad_(False)

    # 5. Training Hyperparameters
    epochs = config['training'].get('finetune_epochs', 20)
    steps_per_epoch = 100
    batch_size = 4 # Needs to be small for BPTT and full Jacobian unrolling

    base_lr = config['training'].get('learning_rate', 1e-4)
    finetune_lr = base_lr / 10.0 # Reduce LR for finetuning stage
    
    # --- FREEZE ENCODER / DECODER TO PREVENT CATASTROPHIC FORGETTING ---
    # To maintain autoregressive rolling stability, we only fine-tune the output projection heads.
    print("\n[Info] Freezing Transformer Feature Extractor to preserve rolli stability...")
    tunable_params = []
    for name, param in dynamic_model.named_parameters():
        if "output_dense" in name or "decoder_output_dense" in name or "de_output" in name:
            param.requires_grad = True
            tunable_params.append(param)
            print(f"  -> Fine-tuning layer: {name}")
        else:
            param.requires_grad = False
            
    optimizer = optim.Adam(tunable_params, lr=finetune_lr)

    W = int(config['window']['train_window_mins'] / config['window']['sampling_interval_min'])
    H_ss = 100 # Match the ~100 step horizon after the disturbance (t=40 to 140 in your graph)
    # Your predict script ran 60 warmup steps + 40 flat steps before the step change = 100 steps of pure SS1
    warmup_steps = 100 

    print(f"  - Optimizer LR: {finetune_lr}")
    print(f"  - Batch Size: {batch_size}, Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}")
    print(f"  - Warm-up Steps: {warmup_steps}, Step Change Length: {H_ss}")

    best_loss = float('inf')
    y_mean_tensor = torch.tensor(mean_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor(std_all[y_sv].values, dtype=torch.float32, device=device)
    y_std_safe = torch.where(torch.abs(y_std_tensor) < 1e-6, torch.ones_like(y_std_tensor), y_std_tensor)

    target_mean_tensor = torch.tensor(mean_all[tab_target_cols].values, dtype=torch.float32, device=device)
    target_std_tensor = torch.tensor(std_all[tab_target_cols].values, dtype=torch.float32, device=device)
    target_std_safe = torch.where(torch.abs(target_std_tensor) < 1e-6, torch.ones_like(target_std_tensor), target_std_tensor)


    history_losses = []

    for epoch in range(epochs):
        epoch_gain_loss = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for step in pbar:
            optimizer.zero_grad()

            # --- (A) Data Sampling ---
            historical_dfs, ss1_df, ss2_df = generate_steady_state_batch(df_raw, batch_size, de_mv, keep_cols, W, std_all)
            B_actual = len(historical_dfs)

            # Standardize logic without console print spams (Hidden Outputs)
            with contextlib.redirect_stdout(io.StringIO()):
                x_en_z_list = []
                for b_df in historical_dfs:
                    b_z = data_utils.apply_zscore(b_df, mean_all, std_all).fillna(0.0)
                    x_en_z_list.append(torch.tensor(b_z[en_mv_and_sv].values, dtype=torch.float32, device=device))
                x_en_z_history = torch.stack(x_en_z_list) # [Batch, W, Features]
                
                ss1_z_df = data_utils.apply_zscore(ss1_df, mean_all, std_all).fillna(0.0)
                ss2_z_df = data_utils.apply_zscore(ss2_df, mean_all, std_all).fillna(0.0)
            # --- (B) Compute MLP Steady State Gain Proxy (K_ss) ---
            # Using Finite Difference and Direction Consistency (Ribeiro et al. / Hsiao et al.)
            ss1_de_p = torch.tensor(ss1_df[de_mv].values, dtype=torch.float32, device=device)
            ss2_de_p = torch.tensor(ss2_df[de_mv].values, dtype=torch.float32, device=device)

            delta_mv = ss2_de_p - ss1_de_p
            # Create a mask for actually perturbed elements
            is_perturbed = torch.abs(delta_mv) > 1e-5
            delta_mv_safe = torch.where(is_perturbed, delta_mv, torch.sign(delta_mv) * 1e-6 + 1e-6)

            mlp_x_z_ss1 = torch.tensor(ss1_z_df[tab_input_cols].values, dtype=torch.float32, device=device)
            mlp_x_z_ss2 = mlp_x_z_ss1.clone()
            
            for col in de_mv:
                if col in tab_input_cols:
                    col_idx_in_tab = tab_input_cols.index(col)
                    col_idx_in_de = de_mv.index(col)
                    m = mean_all[col]
                    s = std_all[col] if abs(std_all[col]) > 1e-6 else 1.0
                    mlp_x_z_ss2[:, col_idx_in_tab] = (ss2_de_p[:, col_idx_in_de] - m) / s

            with torch.no_grad():
                # SS1
                y_mlp_z_ss1 = mlp_model(mlp_x_z_ss1)
                y_mlp_p_ss1 = y_mlp_z_ss1 * target_std_safe + target_mean_tensor
                # SS2
                y_mlp_z_ss2 = mlp_model(mlp_x_z_ss2)
                y_mlp_p_ss2 = y_mlp_z_ss2 * target_std_safe + target_mean_tensor

            delta_y_mlp = y_mlp_p_ss2 - y_mlp_p_ss1 # (Batch, N_targets)
            
            # K_ss calculations (Formula 6) -> Direction only (+1 or -1)
            # Shape: (Batch, N_targets, N_de_mvs)
            K_ss_matrix = delta_y_mlp.unsqueeze(2) / delta_mv_safe.unsqueeze(1)
            K_ss_direction = torch.sign(K_ss_matrix)

            # --- (C) Transformer Autoregressive Warmup ---
            ss1_en_z = torch.tensor(ss1_z_df[en_mv_and_sv].values, dtype=torch.float32, device=device)
            ss1_de_z = torch.tensor(ss1_z_df[de_mv].values, dtype=torch.float32, device=device)

            # 直接傳入真實震盪起點 (非平坦)，準備進行水波撫平
            current_en_history = x_en_z_history.clone()

            debug_warmup_preds = []
            steady_state_preds_ss1 = []

            # Autoregressive washout mode without gradient graphs
            with torch.no_grad():
                for t in range(warmup_steps):
                    step_de_input = ss1_de_z.unsqueeze(1)
                    pred_z = dynamic_model(current_en_history, step_de_input)
                    pred_p_warmup = pred_z.squeeze(1) * y_std_safe + y_mean_tensor

                    # Apply inverse log transform to log-scaled targets (H2S, SO2) BEFORE clamp
                    log_cols_inv = [c for c in ['B35_H2S', 'B35_SO2'] if c in y_sv]
                    log_target_idx = [y_sv.index(c) for c in log_cols_inv]
                    if len(log_target_idx) > 0:
                        pred_p_warmup[:, log_target_idx] = torch.exp(pred_p_warmup[:, log_target_idx])

                    # Ensure predictions don't fall below physical limits
                    tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                    pred_p_warmup[:, tab_target_idx] = torch.clamp(pred_p_warmup[:, tab_target_idx], min=1e-6)

                    # Collect tailing 20 steps of warmup for steady state calculation of SS1
                    if t >= warmup_steps - 40:
                        pred_p_targets = pred_p_warmup[:, tab_target_idx]
                        steady_state_preds_ss1.append(pred_p_targets)
                    
                    debug_warmup_preds.append(pred_p_warmup[0].detach().cpu().numpy())

                    # Expand sliding window with states
                    new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                    for c_idx, c_name in enumerate(en_mv_and_sv):
                        if c_name in de_mv:
                            new_step_features[:, 0, c_idx] = ss1_de_z[:, de_mv.index(c_name)]
                        elif c_name in y_sv:
                            new_step_features[:, 0, c_idx] = pred_z[:, 0, y_sv.index(c_name)]
                        else:
                            new_step_features[:, 0, c_idx] = ss1_en_z[:, c_idx]

                    current_en_history = torch.cat([current_en_history[:, 1:, :], new_step_features], dim=1)

            y_dyn_ss1 = torch.stack(steady_state_preds_ss1).mean(dim=0)

            # --- (D) Transformer Step Change & Dynamics ---
            current_en_history = current_en_history.detach() # Free warmup graph state

            # Setup new Step Change configuration matrix
            ss2_de_z_graph = torch.zeros(B_actual, len(de_mv), device=device)
            for col_idx_in_de, col in enumerate(de_mv):
                m = mean_all[col]
                s = std_all[col] if abs(std_all[col]) > 1e-6 else 1.0
                ss2_de_z_graph[:, col_idx_in_de] = (ss2_de_p[:, col_idx_in_de] - m) / s

            ss2_en_z_const = torch.tensor(ss2_z_df[en_mv_and_sv].values, dtype=torch.float32, device=device)

            dynamic_k_losses = []
            debug_step_preds = []
            steady_state_preds = []

            for t in range(H_ss):
                step_de_input = ss2_de_z_graph.unsqueeze(1)
                pred_z = dynamic_model(current_en_history, step_de_input)

                # Convert to absolute units
                pred_p = pred_z.squeeze(1) * y_std_safe + y_mean_tensor

                # Apply inverse log transform to log-scaled targets (H2S, SO2) BEFORE clamp
                if len(log_target_idx) > 0:
                    pred_p[:, log_target_idx] = torch.exp(pred_p[:, log_target_idx])

                # Ensure dynamic predictions don't fall below physical limits
                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                pred_p[:, tab_target_idx] = torch.clamp(pred_p[:, tab_target_idx], min=1e-6)

                debug_step_preds.append(pred_p[0].detach().cpu().numpy())

                # Check dynamic K Jacobian matrices:
                # Wait for the system to settle before enforcing steady-state constraints.
                # In your graph, after the step change at t=40, it takes a long time to reach the new plateau.
                # We enforce the constraint only on the trailing end of the step change (e.g., the last 40 steps).
                if t >= 60:
                    # Assess matrix against target H2S and SO2 only
                    tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                    pred_p_targets = pred_p[:, tab_target_idx]
                    steady_state_preds.append(pred_p_targets)

                # Renew trailing slide window states
                new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                for c_idx, c_name in enumerate(en_mv_and_sv):
                    if c_name in de_mv:
                        new_step_features[:, 0, c_idx] = ss2_de_z_graph[:, de_mv.index(c_name)]
                    elif c_name in y_sv:
                        new_step_features[:, 0, c_idx] = pred_z[:, 0, y_sv.index(c_name)]
                    else:
                        new_step_features[:, 0, c_idx] = ss2_en_z_const[:, c_idx]

                current_en_history = torch.cat([current_en_history[:, 1:, :], new_step_features], dim=1)

            # Finite Difference Dynamic Gain & Consistency Loss (Formula 7 & 8)
            y_dyn_ss2 = torch.stack(steady_state_preds).mean(dim=0)
            delta_y_dyn = y_dyn_ss2 - y_dyn_ss1 # (Batch, N_targets)

            # K_dyn calculations -> Magnitude (Formula 7)
            K_dyn_matrix = delta_y_dyn.unsqueeze(2) / delta_mv_safe.unsqueeze(1)

            # H = 1 if K_dyn and K_ss have same sign, 0 otherwise
            H_direction = (K_dyn_matrix * K_ss_direction > 0).float()

            # Loss = |K_dyn| * (1 - H)
            loss_matrix = torch.abs(K_dyn_matrix) * (1.0 - H_direction)

            # Filter ONLY the elements that were actually perturbed
            is_perturbed_mask = is_perturbed.unsqueeze(1).expand(-1, K_dyn_matrix.size(1), -1)
            
            if is_perturbed_mask.sum() > 0:
                loss_gain = torch.mean(loss_matrix[is_perturbed_mask])
            else:
                loss_gain = torch.tensor(0.0, device=device, requires_grad=True)

            if loss_gain.item() > 0:
                loss_gain.backward()
                optimizer.step()

            epoch_gain_loss += loss_gain.item()
            pbar.set_postfix({'Gain_Loss': f"{loss_gain.item():.6f}"})

            # -----------------------------------------------------
            # Plot training trajectories to verify shape and S.S. Direction
            # -----------------------------------------------------
            if epoch == 0: # 匯出整個第 0 Epoch 的所有 Step 圖片供手動確認
                import matplotlib.pyplot as plt
                plot_dir = f'./results/{exp}_PGIN_Finetuned/trajectories_ep{epoch}'
                os.makedirs(plot_dir, exist_ok=True)
                
                # Extract SS1 operational states for logging
                ss1_state_vals = ss1_df.iloc[0]
                state_text_lines = []
                for col_name in tab_input_cols:
                    state_text_lines.append(f"{col_name}: {ss1_state_vals[col_name]:.2f}")
                state_str = " | ".join(state_text_lines)

                perturbed_mv_name = "Unknown MV"
                mv_dir_str = ""
                p_idx = 0
                for i, is_p in enumerate(is_perturbed[0]):
                    if is_p:
                        perturbed_mv_name = de_mv[i]
                        p_idx = i
                        if delta_mv[0, i].item() > 0:
                            mv_dir_str = " (+dMV)"
                        else:
                            mv_dir_str = " (-dMV)"
                        break

                # 從 Tabular MLP 中取出 batch 0 的兩項指標增益方向 (+1 或 -1)
                h2s_ss_dir = "+ (UP)" if K_ss_direction[0, 0, p_idx].item() > 0 else "- (DOWN)"
                so2_ss_dir = "+ (UP)" if K_ss_direction[0, 1, p_idx].item() > 0 else "- (DOWN)"

                warmup_arr = np.array(debug_warmup_preds)
                step_arr = np.array(debug_step_preds)
                full_traj = np.concatenate([warmup_arr, step_arr], axis=0) # Shape: (200, N_y)

                h2s_idx = y_sv.index('B35_H2S')
                so2_idx = y_sv.index('B35_SO2')

                plt.figure(figsize=(16, 7))
                
                # B35_H2S 子圖
                plt.subplot(1, 2, 1)
                plt.plot(full_traj[:, h2s_idx], label='Dynamic Pred', color='blue', linewidth=2)
                plt.axvline(x=warmup_steps, color='r', linestyle=':', label='Step Change Init')
                plt.title(f'B35_H2S | Target S.S. Dir: {h2s_ss_dir}')
                plt.legend()
                
                # B35_SO2 子圖
                plt.subplot(1, 2, 2)
                plt.plot(full_traj[:, so2_idx], label='Dynamic Pred', color='green', linewidth=2)
                plt.axvline(x=warmup_steps, color='r', linestyle=':', label='Step Change Init')
                plt.title(f'B35_SO2 | Target S.S. Dir: {so2_ss_dir}')
                plt.legend()

                plt.suptitle(f'PGIN - Epoch {epoch} Step {step} | Perturbed MV: {perturbed_mv_name}{mv_dir_str} | Dynamics vs Static Direction\nSS1 State: {state_str}', fontsize=10)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                debug_plot_path = os.path.join(plot_dir, f'step_{step:03d}.png')
                plt.savefig(debug_plot_path, dpi=150)
                plt.close()

        avg_loss = epoch_gain_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{epochs}] | Avg L_gain: {avg_loss:.6f}")
        history_losses.append(avg_loss)

        # Dynamic Saving Mechanism - Only update params if validation improves
        out_model_path = f'./saved_models/{exp}_PGIN_Finetuned.pth'

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  [Save] New Best Model saved to {out_model_path} (Loss: {best_loss:.6f})")
            torch.save(dynamic_model.state_dict(), out_model_path)

    # Post process: Export plot
    import matplotlib.pyplot as plt
    out_dir = f'./results/{exp}_PGIN_Finetuned'
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history_losses, marker='o', color='purple', linewidth=2)
    plt.title(f'PGIN Finetuning Gain Loss - {exp}')
    plt.xlabel('Epoch')
    plt.ylabel('Gain MSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'finetune_loss_curve.png'), dpi=300)
    plt.close()

    print(f"\n[Done] Model finetuning complete. Weights securely saved to {out_model_path}")
    print(f"[Info] Training metrics exported to {out_dir}/finetune_loss_curve.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)


