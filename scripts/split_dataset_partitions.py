# This script reads the training files specified in the config, splits them into valid and test partitions based on the specified ratios, and saves the new CSV files for valid and test sets.
import pandas as pd
import yaml
import os
import argparse

def split_partitions(config_path):
    # 1. Load Config
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    data_path = config['data']['path']
    training_files = config['data']['training_files']
    
    test_ratio = config['data'].get('test_data_split', 0.1)
    valid_ratio = config['data'].get('valid_data_split', 0.1)
    
    print(f"Config: {config_path}")
    print(f"Test Ratio: {test_ratio}")
    print(f"Valid Ratio: {valid_ratio} (of the remaining training set)")
    
    for filename in training_files:
        full_path = os.path.join(data_path, filename)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
            
        print(f"\nProcessing file: {full_path}")
        
        # Read Data
        # Try-except block for robust reading
        try:
            df = pd.read_csv(full_path)
        except:
             df = pd.read_csv(full_path, engine='python')

        total_len = len(df)
        
        # Calculate Logic consistent with train_claus_resampled.py
        # split_point1 = int(data_len * (1 - cfg_data['test_data_split']))
        # split_point2 = int(split_point1 * (1 - cfg_data['valid_data_split']))
        
        split_point1 = int(total_len * (1 - test_ratio))
        split_point2 = int(split_point1 * (1 - valid_ratio))
        
        print(f"  Total Rows: {total_len}")
        print(f"  Valid Range: {split_point2} -> {split_point1} ({split_point1 - split_point2} rows)")
        print(f"  Test Range:  {split_point1} -> {total_len} ({total_len - split_point1} rows)")
        
        # Extract Partitions
        valid_df = df.iloc[split_point2:split_point1].copy()
        test_df = df.iloc[split_point1:].copy()
        
        # Save Valid
        base_name = os.path.splitext(filename)[0]
        
        valid_filename = f"{base_name}_valid_split.csv"
        valid_path = os.path.join(data_path, valid_filename)
        valid_df.to_csv(valid_path, index=False)
        print(f"  Saved VALID split to: {valid_filename}")
        
        # Save Test (Optional, but good to have consistent)
        test_filename = f"{base_name}_test_split.csv"
        test_path = os.path.join(data_path, test_filename)
        test_df.to_csv(test_path, index=False)
        print(f"  Saved TEST split to:  {test_filename}")

if __name__ == "__main__":
    split_partitions('configs/transformer_layerwise_71var.yaml')
