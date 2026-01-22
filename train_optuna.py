
import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import traceback

# Reuse existing project modules
from src.dataset import MultiStepS2SDataset
from src.models import get_model
from src import engine
from src import utils as data_utils
from src import variable_selection

# ==============================================================================
# Global Settings & Data Cache
# ==============================================================================
SEED = 42
BASE_CONFIG_PATH = 'configs/transformer_layerwise_AT_Rolling_Aligned_GRU_Interval10_median.yaml'

# Cache for loaded and preprocessed dataframes to avoid I/O bottleneck
CACHED_DFS = {} 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Helper to load and preprocess data (Cached)
def get_processed_data(config):
    data_path = config['data']['path']
    train_files = config['data']['training_files']
    interval_min = config['window']['sampling_interval_min']
    
    # Check cache
    cache_key = (tuple(train_files), interval_min)
    if cache_key in CACHED_DFS:
        return CACHED_DFS[cache_key]
    
    dfs = []
    print("Loading Data for Optimization...")
    for f_name in train_files:
        f_path = os.path.join(data_path, f_name)
        if not os.path.exists(f_path):
            continue
            
        if f_name.endswith('.csv'):
            df = pd.read_csv(f_path)
        else:
            df = pd.read_excel(f_path)
            
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.sort_values('DateTime', inplace=True)
            
        # Resample
        if interval_min > 1:
            df_resampled = df.rolling(window=interval_min, min_periods=interval_min).median()
            df_resampled = df_resampled.iloc[interval_min-1::interval_min].reset_index(drop=True)
            df_resampled.dropna(inplace=True)
        else:
            df_resampled = df
        
        # Log Transform
        log_cols = ['B35_H2S', 'B35_SO2']
        df_resampled = data_utils.apply_log_transform(df_resampled, log_cols)
        
        dfs.append(df_resampled)
    
    # Concat for generic stats
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate Stats (Robust)
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
    median, iqr = data_utils.calculate_robust_stats(full_df[numeric_cols])
    
    # Scale All
    scaled_dfs = []
    for df in dfs:
        # Filter only numeric cols that we have stats for
        df_num = df[numeric_cols]
        df_z = data_utils.apply_robust_scale(df_num, median, iqr)
        
        # Keep DateTime if exists? The original script drops it mostly?
        # Let's keep numeric only for simplicity as model inputs are all numeric.
        scaled_dfs.append(df_z)
        
    CACHED_DFS[cache_key] = (scaled_dfs, median, iqr)
    return scaled_dfs, median, iqr

# ==============================================================================
# Objective Function
# ==============================================================================

def objective(trial):
    try:
        set_seed(SEED)
        
        # 1. Load Base Config
        config = load_config(BASE_CONFIG_PATH)
        
        # 2. Hyperparameters from Optuna
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        emb_dim = trial.suggest_categorical("embedding_dim", [16, 32])
        ffn_expansion = trial.suggest_categorical("ffn_expansion", [2, 4])
        
        # Apply to config
        config['training']['learning_rate'] = lr
        config['model']['dropout'] = dropout
        config['model']['embedding_dim'] = emb_dim
        config['model']['hidden_dim'] = emb_dim
        config['model']['ffn_expansion'] = ffn_expansion
        
        # 3. Data Preparation
        scaled_dfs, median, iqr = get_processed_data(config)
        
        # Define Variable Selection
        if len(scaled_dfs) == 0:
            raise ValueError("No data loaded!")
            
        sample_df = scaled_dfs[0]
        de_mv, y_sv, con_tag, en_mv_sv = variable_selection.variable_selection(
            config['data']['variables_num']
        )
        
        # Dataset Parameters
        H_out = config['window']['prediction_length']
        interval_min = config['window']['sampling_interval_min']
        W = config['window']['train_window_mins'] // interval_min
        loss_weighting = config['training'].get('loss_weighting', {'weights': [1.0]})
        num_windows = len(loss_weighting['weights'])
        dataset_H = H_out * num_windows if num_windows > 1 else H_out

        # Create Datasets
        all_datasets = []
        for df_z in scaled_dfs:
            if len(df_z) <= W + dataset_H: continue
            all_datasets.append(MultiStepS2SDataset(df_z, en_mv_sv, de_mv, y_sv, W, dataset_H))
            
        if not all_datasets:
            raise ValueError("Datasets empty after processing.")

        full_dataset = ConcatDataset(all_datasets)
        
        # Split
        total_len = len(full_dataset)
        val_len = int(total_len * config['data']['valid_data_split'])
        train_len = total_len - val_len
        train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))
        
        # Loaders
        batch_size = config['training']['batch_size']
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 4. Model Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_en_input = len(en_mv_sv) 
        num_output = len(y_sv)       
        
        config['data']['num_en_input'] = num_en_input
        config['data']['num_output'] = num_output
        
        model = get_model(config).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        # 5. Training Loop
        epochs = 50 
        
        # Select Step Function
        if num_windows > 1:
            step_fn = engine.step_wise_rolling_at_loss_step
        else:
            step_fn = engine.step_wise_rolling_training_step

        best_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss = engine.train_one_epoch(
                model, train_dl, optimizer, loss_fn, device, 
                step_fn,
                config
            )
            
            # Eval
            val_loss = engine.evaluate(
                model, val_dl, loss_fn, device,
                step_fn,
                config
            )

            # Keep best
            if val_loss < best_loss:
                best_loss = val_loss
            
            # Report to Optuna
            trial.report(val_loss, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return best_loss

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print("Exception in trial:")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    args = parser.parse_args()

    # Create Study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print(f"Starting Optimization with {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials, catch=(Exception,))
    
    print("\n==================================")
    print("Optimization Finished!")
    if len(study.trials) > 0:
        print("Best Trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    print("==================================")
