# check_training_range.py - Quick script to find min/max of air2_SP and HEATER2_output_T_SP
# from Test_dataform_* CSV files in data/Claus_senpai_data

import os
import glob
import pandas as pd

def main():
    data_dir = "data/Claus_senpai_data"
    pattern = os.path.join(data_dir, "Test_dataform_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"[Error] No files found matching: {pattern}")
        return
    
    print(f"Found {len(files)} files\n")
    print("=" * 80)
    
    cols = ["air2_SP", "HEATER2_output_T_SP"]
    
    for f in files:
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[Error] Failed to read {filename}: {e}")
            continue
        
        print(f"\n📄 {filename}")
        print("-" * 40)
        
        for col in cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}:")
                print(f"    min = {min_val:.4f}")
                print(f"    max = {max_val:.4f}")
            else:
                print(f"  {col}: [NOT FOUND]")
        
        print()
    
    print("=" * 80)
    print("\n📊 Summary across ALL files:")
    print("-" * 40)
    
    # Aggregate across all files
    all_dfs = []
    for f in files:
        try:
            all_dfs.append(pd.read_csv(f))
        except:
            pass
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        for col in cols:
            if col in combined.columns:
                print(f"  {col}:")
                print(f"    GLOBAL min = {combined[col].min():.4f}")
                print(f"    GLOBAL max = {combined[col].max():.4f}")


if __name__ == "__main__":
    main()
