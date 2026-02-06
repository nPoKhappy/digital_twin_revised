# 這個腳本用於分析訓練集的敏感性，特別是對於H2S和SO2的影響。它將使用訓練集中的多個隨機窗口來計算每個輸入變量對未來預測的平均影響，並將結果保存為CSV文件以供後續分析和可視化。
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src import variable_selection
from src.models import get_model

# Force CPU or CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_normalization_stats(result_dir):
    """Load mean and std from CSV files in result directory"""
    mean_path = os.path.join(result_dir, 'zscore_mean.csv')
    std_path = os.path.join(result_dir, 'zscore_std.csv')
    
    # Read CSVs, assuming format: index is variable name, column '0' has value
    # But checking file content: line 1 is ",0", line 2 is "i,4.6...", so it seems header is row 0
    # and index is column 0.
    df_mean = pd.read_csv(mean_path, index_col=0)
    df_std = pd.read_csv(std_path, index_col=0)
    
    # Convert to dictionaries for easy access
    mean_dict = df_mean.iloc[:, 0].to_dict()
    std_dict = df_std.iloc[:, 0].to_dict()
    
    return mean_dict, std_dict

def normalize(df, mean_dict, std_dict, variables):
    """Normalize specific variables in the dataframe"""
    df_norm = df.copy()
    for var in variables:
        if var in mean_dict and var in std_dict:
            mu = mean_dict[var]
            sigma = std_dict[var]
            if sigma == 0:
                print(f"Warning: std dev for {var} is 0. Avoiding division by zero.")
                sigma = 1e-6
            df_norm[var] = (df_norm[var] - mu) / sigma
        else:
            print(f"Warning: Stats for {var} not found. Skipping normalization.")
    return df_norm

def main():
    config_path = 'configs/transformer_71var_vanilla.yaml'
    
    # 1. Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading experiment: {config['exp_name']}")
    
    # 2. Get Variable Config
    total_vars = config['data']['variables_num'] # 71
    de_mv, y_sv, _, en_mv_and_sv = variable_selection.variable_selection(total_vars)
    
    config['data']['num_en_input'] = len(en_mv_and_sv)
    config['data']['num_de_input'] = len(de_mv)
    config['data']['num_output'] = len(y_sv)
    
    # 3. Load Model
    model = get_model(config).to(device)
    model_path = os.path.join('saved_models', f"{config['exp_name']}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    # Check if checkpoint is dict or model
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded.")
    
    # 4. Load Stats for Normalization/Perturbation
    # Assuming result dir is results/exp_name
    result_dir = os.path.join('results', config['exp_name'])
    mean_dict, std_dict = load_normalization_stats(result_dir)
    
    # 5. Load Data Sample (Training Set R5)
    data_cfg = config['data']
    # Switch to Training Set
    # To get "Global/Overall" sensitivity, we should look at multiple regimes
    # Using the full list from config or a representative set
    training_files = config['data']['training_files']
    # If strictly using the files previously discussed (R=5 etc), we can stick to those.
    # config['data']['training_files'] usually contains the list:
    # ["Test_dataform_change_air2_R=5_converted.csv", "Test_dataform_change_air2_R=5-1_converted.csv", ...]
    
    print(f"Loading ALL TRAINING data for GLOBAL sensitivity analysis: {training_files}")
    dfs = []
    
    # Load first training file (usually enough and representative)
    # Or load all if you want comprehensive coverage
    for fname in training_files: # Load the specific test file
        fpath = os.path.join(data_cfg['path'], fname)
        if os.path.exists(fpath):
            print(f"  Reading {fpath}")
            dfs.append(pd.read_csv(fpath))
        else:
            print(f"  Warning: {fpath} not found.")
            
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total Training Rows: {len(df)}")
    
    # Normalize data
    # We need to normalize ALL columns that are in en_mv_and_sv
    df_norm = normalize(df, mean_dict, std_dict, en_mv_and_sv)
    
    # Select multiple windows for Global Sensitivity
    # Window size: train_window_mins / sampling_interval_min
    input_len = config['window']['train_window_mins'] // config['window']['sampling_interval_min'] # 18
    pred_len = config['window']['prediction_length'] # 18
    sampling = config['window']['sampling_interval_min']
    
    # Number of random samples to average over
    N_SAMPLES = 100 
    print(f"Sampling {N_SAMPLES} windows randomly for Global Sensitivity Analysis...")
    
    # Generate random indices
    max_idx = len(df_norm) - (input_len + pred_len)
    if max_idx <= 0:
        raise ValueError("Dataset too small for the requested window sizes.")
        
    np.random.seed(42) # For reproducibility
    start_indices = np.random.randint(0, max_idx, size=N_SAMPLES)
    
    # Prepare Batched Inputs
    # Encoder Input: (N, W, 71)
    # Decoder Input: (N, H, 12)
    
    enc_param_list = []
    dec_param_list = []
    
    for start_idx in start_indices:
        enc_data = df_norm.iloc[start_idx : start_idx + input_len][en_mv_and_sv].values
        dec_data = df_norm.iloc[start_idx + input_len : start_idx + input_len + pred_len][de_mv].values
        enc_param_list.append(enc_data)
        dec_param_list.append(dec_data)
        
    enc_input_origin = torch.tensor(np.array(enc_param_list), dtype=torch.float32).to(device) # (N, W, 71)
    dec_input = torch.tensor(np.array(dec_param_list), dtype=torch.float32).to(device) # (N, H, 12)
    
    print(f"Batch Input shape: {enc_input_origin.shape}")
    print(f"Batch Future shape: {dec_input.shape}")
    
    # 6. Baseline Prediction (Batched)
    with torch.no_grad():
        baseline_pred = model(enc_input_origin, dec_input) # (N, H, 59)
    
    # 7. Perturbation Loop (Revised for Bias/Step Test)
    # K: input variable
    # T: time step in prediction
    # J: output variable
    
    W = input_len
    K = len(en_mv_and_sv)
    H = pred_len
    J = len(y_sv)
    
    # K: Variables, H: Prediction Steps, J: Output Variables
    impact_tensor = torch.zeros((K, H, J)) 
    
    print("Running perturbation analysis (Step Bias Test) across batch...")
    
    for k in tqdm(range(K), desc="Variables"):
        var_name = en_mv_and_sv[k]
        
        # Determine perturbation delta (1.0 sigma)
        delta = 1.0 
        
        # Clone input
        enc_input_perturbed = enc_input_origin.clone()
        
        # Perturb: Change the WHOLE history window for variable k across ALL samples
        # enc_input_perturbed[batch, time, variable]
        enc_input_perturbed[:, :, k] += delta 
        
        # Predict
        with torch.no_grad():
            pred_perturbed = model(enc_input_perturbed, dec_input)
        
        # Calculate difference (Future Trajectory Deviation)
        diff = pred_perturbed - baseline_pred # (N, H, J)
        
        # Calculate Mean Absolute Impact across Batch (dim=0)
        # Result: (H, J) - Average impact for this variable k
        mean_abs_diff = torch.mean(torch.abs(diff), dim=0)
        
        # Store
        impact_tensor[k, :, :] = mean_abs_diff.cpu()
            
    # 8. Analyze for H2S and SO2
    # Find indices for B35_H2S and B35_SO2 in y_sv
    target_vars = ['B35_H2S', 'B35_SO2']
    target_indices = [i for i, v in enumerate(y_sv) if v in target_vars]
    
    if not target_indices:
        print("Targets B35_H2S/SO2 not found in output variables.")
        return

    # Create directory for saving results
    # Switching to separate folder for Training Set analysis
    save_dir = os.path.join(result_dir, 'sensitivity_analysis_train_set')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the full tensor if needed for debugging
    # torch.save(impact_tensor, os.path.join(save_dir, 'impact_tensor.pt'))
    
    # Summarize importance: Mean of Absolute Difference over Future Time T
    # Shape: (K, H, J) -> reduce H (dim=1) to get (K, J)
    # We use MEAN now, representing Average Deviation per unit perturbation
    total_impact = torch.mean(torch.abs(impact_tensor), dim=1) 
    print(f"Total impact shape: {total_impact.shape}") 
    
    # For each target variable
    for j_idx, target_idx in enumerate(target_indices):
        t_name = y_sv[target_idx] # e.g., B35_H2S
        
        print(f"\nTop 10 variables affecting {t_name} (Mean Absolute Impact):")
        
        # Get impact scores for this target
        scores = total_impact[:, target_idx].numpy() # (K,)
        
        # Sort indices
        sorted_indices = np.argsort(scores)[::-1]
        
        results_list = []
        for i in range(min(10, K)):
            idx = sorted_indices[i]
            var = en_mv_and_sv[idx]
            score = scores[idx]
            print(f"  {i+1}. {var}: {score:.4f}")
            results_list.append({'rank': i+1, 'variable': var, 'score': score})
            
        # Save detailed heatmap data for this target
        # For Step Test, "Tau" dimension is gone.
        # We can plot heatmap of Impact over Future Time (T) vs Variable (K)
        # Shape: (K, H)
        heatmap_data = torch.abs(impact_tensor[:, :, target_idx]).numpy() # (K, H)
        
        # Save heatmap data to csv
        # Rows: Variables, Cols: Future Time Steps
        heatmap_df = pd.DataFrame(heatmap_data, index=en_mv_and_sv, columns=[f"t+{i+1}" for i in range(H)])
        heatmap_df.to_csv(os.path.join(save_dir, f'heatmap_{t_name}.csv'))
        print(f"Saved heatmap data to {os.path.join(save_dir, f'heatmap_{t_name}.csv')}")

if __name__ == "__main__":
    main()
