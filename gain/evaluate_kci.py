import os
import torch
import numpy as np
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from src import utils as data_utils
from src.variable_selection import variable_selection
from src.models import get_model
from src.models.tabular_mlp import TabularMLP
from src.dataset import MultiStepS2SDataset

def evaluate_kci(config_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    exp = config.get('exp_name', 'transformer_layerwise_71var_decoder_input_sp_PGIN_From_Scratch')
    variables_num = config['data']['variables_num']
    de_mv, y_sv, _, en_mv_and_sv = variable_selection(variables_num)
    
    # Setup data parameters
    sampling_interval_min = config['window']['sampling_interval_min']
    train_window_mins = config['window']['train_window_mins']
    seq_len = train_window_mins // sampling_interval_min
    pred_len = config['window']['prediction_length']
    
    # Increase stride for faster evaluation (hardcoded to ignore config)
    stride = 25 

    print("Loading Z-scores...")
    zscore_dir = f'./results/{exp}/'
    if not os.path.exists(zscore_dir):
        zscore_dir = f'./results/{exp}_PGIN_From_Scratch/' # Fallback
        
    zscore_mean = pd.read_csv(f"{zscore_dir}zscore_mean.csv", index_col=0).squeeze()
    zscore_std = pd.read_csv(f"{zscore_dir}zscore_std.csv", index_col=0).squeeze()

    print("Loading test dataset...")
    test_files = config['data'].get('testing_files', [])
    if not test_files:
        test_files = [f for f in os.listdir('data/Claus_dynamic') if f.endswith('_converted.csv')]
        
    # Restore R5-7 and R5-8 per user request
    # test_files = [f for f in test_files if 'R=5-7' not in f and 'R=5-8' not in f]

    print(f"Testing on files: {test_files}")

    datasets = []
    for f in test_files:
        df = pd.read_csv(os.path.join('data/Claus_dynamic', f))
        target_features = en_mv_and_sv.copy()
        for col in target_features:
            if col not in df.columns:
                df[col] = 0.0

        if config['data'].get('apply_rolling_median', False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].rolling(window=config['data']['rolling_median_window'], min_periods=1).median()
            df = df.dropna().reset_index(drop=True)

        target_cols_log = ['B35_H2S', 'B35_SO2']
        df = data_utils.apply_log_transform(df, target_cols_log)

        for col in target_features:
            if col in zscore_mean.index and zscore_std[col] != 0:
                df[col] = (df[col] - zscore_mean[col]) / zscore_std[col]
            else:
                df[col] = 0.0

        # Since MultiStepS2SDataset doesn't support stride, we apply it via Subset
        ds = MultiStepS2SDataset(df, en_mv_and_sv, de_mv, y_sv, seq_len, pred_len)
        indices = list(range(0, len(ds), stride))
        ds_subset = torch.utils.data.Subset(ds, indices)
        datasets.append(ds_subset)

    test_loader = DataLoader(ConcatDataset(datasets), batch_size=config['training'].get('batch_size', 256), shuffle=False, drop_last=True)

    print("Loading Teacher MLP...")
    mlp_chkpt = f'./saved_models/Tabular_MLP_Claus_Final_tabular_mlp.pth'
    if not os.path.exists(mlp_chkpt):
        # Allow fallback
        mlp_chkpt = "./saved_models/Tabular_MLP_Claus_tabular_mlp.pth"

    tab_target_cols = ['B35_H2S', 'B35_SO2']
    tab_input_cols = config['data']['tabular_inputs'] if 'tabular_inputs' in config['data'] else de_mv

    target_mean_tensor = torch.tensor(zscore_mean[tab_target_cols].values, dtype=torch.float32, device=device)
    target_std_tensor = torch.tensor(zscore_std[tab_target_cols].values, dtype=torch.float32, device=device)
        
    mlp_teacher = TabularMLP(num_features=8, num_outputs=2,
                             hidden_dims=[256, 128, 64], dropout=0.05, activation='gelu',
                             target_mean=target_mean_tensor, target_std=target_std_tensor).to(device)
    mlp_teacher.load_state_dict(torch.load(mlp_chkpt, map_location=device))
    mlp_teacher.eval()

    print("Loading Transformer...")
    # Inject dimensions dynamically so get_model instantiates correctly
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)

    dynamic_model = get_model(config).to(device)
    dynamic_model.load_state_dict(torch.load(model_path, map_location=device))
    dynamic_model.eval()

    total_correct_dir = 0
    total_eval_items = 0

    y_mean_tensor = torch.tensor([zscore_mean.get(c, 0.0) for c in y_sv], dtype=torch.float32, device=device)
    y_std_tensor = torch.tensor([zscore_std.get(c, 1.0) for c in y_sv], dtype=torch.float32, device=device)
    y_std_safe = torch.where(torch.abs(y_std_tensor) < 1e-6, torch.tensor(1.0, device=device), y_std_tensor)

    de_mean_tensor = torch.tensor([zscore_mean.get(c, 0.0) for c in de_mv], dtype=torch.float32, device=device)
    de_std_tensor = torch.tensor([zscore_std.get(c, 1.0) for c in de_mv], dtype=torch.float32, device=device)
    de_std_safe = torch.where(torch.abs(de_std_tensor) < 1e-6, torch.tensor(1.0, device=device), de_std_tensor)

    print("Starting KCI Evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            en_x, de_x, y_true = [b.to(device) for b in batch]
            B_actual = en_x.size(0)

            ss1_de_z_graph = en_x[:, -1, [en_mv_and_sv.index(c) for c in de_mv]]

            # SYNTHETIC SINGLE-VARIABLE PERTURBATION (Identical to train_pgin_from_scratch.py)
            ss1_de_p = ss1_de_z_graph * de_std_safe + de_mean_tensor
            ss2_de_p = ss1_de_p.clone()
            
            # Determine which MVs to perturb based on config preference
            target_mvs = [col for col in ['air2_SP_m3', 'air2_SP', 'HEATER2_output_T_SP'] if col in de_mv]
            if not target_mvs:
                target_mvs = de_mv

            # Apply single variable perturbation per batch item
            for b in range(B_actual):
                var_to_perturb = target_mvs[b % len(target_mvs)]
                var_idx = de_mv.index(var_to_perturb)
                
                # Fetch 0.5 physical std delta
                std_val = de_std_safe[var_idx].item()
                std_val = std_val if abs(std_val) > 1e-6 else 1.0
                delta = 0.5 * std_val
                
                # Alternate + and - delta to thoroughly evaluate
                if b % 2 == 0:
                    ss2_de_p[b, var_idx] += delta
                else:
                    ss2_de_p[b, var_idx] -= delta
            
            # Convert back to standard Z-scores
            ss2_de_z_graph = (ss2_de_p - de_mean_tensor) / de_std_safe

            ss1_en_z_const = en_x[:, -1, [en_mv_and_sv.index(c) for c in en_mv_and_sv if c not in de_mv]]
            ss2_en_z_const = ss1_en_z_const # assume external var unchanged

            tab_input_cols = [
                "acidgas_Fv", "acidgas_T", "acidgas_P", "air2_SP",
                "HEATER2_output_T_SP", "acidgas_CO2", "acidgas_H2O", "acidgas_H2S"
            ]

            ss1_mlp_input = en_x[:, -1, [en_mv_and_sv.index(c) for c in tab_input_cols if c in en_mv_and_sv]]
            ss1_mlp_input = ss1_mlp_input.to(device)

            ss2_mlp_input = ss1_mlp_input.clone()
            for col in tab_input_cols:
                if col in de_mv:
                    col_idx_in_tab = tab_input_cols.index(col)
                    col_idx_in_de = de_mv.index(col)
                    ss2_mlp_input[:, col_idx_in_tab] = ss2_de_z_graph[:, col_idx_in_de]

            with torch.no_grad():
                y_mlp_z_ss1 = mlp_teacher(ss1_mlp_input)
                y_mlp_z_ss2 = mlp_teacher(ss2_mlp_input)
            
            y_mlp_p_ss1 = y_mlp_z_ss1 * target_std_tensor + target_mean_tensor
            y_mlp_p_ss2 = y_mlp_z_ss2 * target_std_tensor + target_mean_tensor
            delta_y_mlp = y_mlp_p_ss2 - y_mlp_p_ss1
            
            ss1_de_p = ss1_de_z_graph * de_std_safe + de_mean_tensor
            ss2_de_p = ss2_de_z_graph * de_std_safe + de_mean_tensor

            delta_mv = ss2_de_p - ss1_de_p
            is_perturbed = (torch.abs(delta_mv) > 1e-5)
            delta_mv_safe = torch.where(is_perturbed, delta_mv, torch.sign(delta_mv)*1e-6 + 1e-6)

            K_ss_matrix = delta_y_mlp.unsqueeze(2) / delta_mv_safe.unsqueeze(1)
            K_ss_direction = torch.sign(K_ss_matrix)

            # --- Dynamic Model Rollout ---
            ss2_pred_len = 100
            tab_target_cols = ['B35_H2S', 'B35_SO2']

            # ss1 predictions
            current_en_history_ss1 = en_x.clone()
            ss1_preds = []
            for t in range(100):
                pred_z_ss1 = dynamic_model(current_en_history_ss1, ss1_de_z_graph.unsqueeze(1))
                pred_p_ss1 = pred_z_ss1[:, 0, :].clone() * y_std_safe + y_mean_tensor
                
                log_target_idx = [y_sv.index(c) for c in set(['B35_H2S', 'B35_SO2']) & set(y_sv)]
                if len(log_target_idx) > 0:
                    pred_p_ss1[:, log_target_idx] = torch.exp(pred_p_ss1[:, log_target_idx])

                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                pred_p_ss1[:, tab_target_idx] = torch.clamp(pred_p_ss1[:, tab_target_idx], min=1e-6)

                if t >= 0: # take all 40 steps 
                    ss1_preds.append(pred_p_ss1[:, tab_target_idx])
                
                # Next step history
                new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                for c_idx, c_name in enumerate(en_mv_and_sv):
                    if c_name in de_mv:
                        new_step_features[:, 0, c_idx] = ss1_de_z_graph[:, de_mv.index(c_name)]
                    elif c_name in y_sv:
                        new_step_features[:, 0, c_idx] = pred_z_ss1[:, 0, y_sv.index(c_name)]
                    else:
                        new_step_features[:, 0, c_idx] = ss1_en_z_const[:, c_idx]

                current_en_history_ss1 = torch.cat([current_en_history_ss1[:, 1:, :], new_step_features], dim=1)

            y_dyn_ss1 = torch.stack(ss1_preds[-40:]).mean(dim=0)
            
            # Start ss2 rollout from the state after ss1 rollout
            current_en_history_ss2 = current_en_history_ss1.detach()
            steady_state_preds = []
            
            for t in range(ss2_pred_len):
                pred_z = dynamic_model(current_en_history_ss2, ss2_de_z_graph.unsqueeze(1))
                pred_p = pred_z[:, 0, :].clone() * y_std_safe + y_mean_tensor
                
                # Reverse log
                log_target_idx = [y_sv.index(c) for c in set(['B35_H2S', 'B35_SO2']) & set(y_sv)]
                if len(log_target_idx) > 0:
                    pred_p[:, log_target_idx] = torch.exp(pred_p[:, log_target_idx])

                tab_target_idx = [y_sv.index(c) for c in tab_target_cols]
                pred_p[:, tab_target_idx] = torch.clamp(pred_p[:, tab_target_idx], min=1e-6)

                if t >= 60:
                    steady_state_preds.append(pred_p[:, tab_target_idx])
                
                # Next step history
                new_step_features = torch.zeros(B_actual, 1, len(en_mv_and_sv), device=device)
                for c_idx, c_name in enumerate(en_mv_and_sv):
                    if c_name in de_mv:
                        new_step_features[:, 0, c_idx] = ss2_de_z_graph[:, de_mv.index(c_name)]
                    elif c_name in y_sv:
                        new_step_features[:, 0, c_idx] = pred_z[:, 0, y_sv.index(c_name)]
                    else:
                        new_step_features[:, 0, c_idx] = ss2_en_z_const[:, c_idx]

                current_en_history_ss2 = torch.cat([current_en_history_ss2[:, 1:, :], new_step_features], dim=1)

            y_dyn_ss2 = torch.stack(steady_state_preds).mean(dim=0)
            delta_y_dyn = y_dyn_ss2 - y_dyn_ss1

            K_dyn_matrix = delta_y_dyn.unsqueeze(2) / delta_mv_safe.unsqueeze(1)
            
            # valid_mlp_mask is [B, 2]
            valid_mlp_mask = (torch.abs(delta_y_mlp) >= 1e-5)
            # is_perturbed is [B, 8]
            
            # K_dyn_matrix is [B, 2, 8]
            final_mask = is_perturbed.unsqueeze(1) & valid_mlp_mask.unsqueeze(2)
            correct_mask = (K_dyn_matrix * K_ss_direction > 0) & final_mask

            total_correct_dir += correct_mask.sum().item()
            total_eval_items += final_mask.sum().item()

    if total_eval_items > 0:
        kci = (total_correct_dir / total_eval_items) * 100
        print(f"\nFinal Test KCI (Knowledge Consistency Index): {kci:.2f}%")
        print(f"Total Evaluated Physical Constraints: {total_eval_items}")
        print(f"Total Correct Directions: {total_correct_dir}")
    else:
        print("\nNo valid physical constraint points evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    
    evaluate_kci(args.config, args.weights)
