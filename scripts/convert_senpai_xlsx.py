# convert_senpai_xlsx.py - Convert senpai's xlsx files to match your CSV format
# 將學長的xlsx數據轉換成與你的CSV格式一致
# Target: 1440 rows (24 hours), step change at position 400

import pandas as pd
import os
import glob
import re

# Configuration
SENPAI_DIR = "data/Claus_dynamic/in_training_distribution_step_change/senpai_xlsx_file"
OUTPUT_DIR = "data/Claus_dynamic/in_training_distribution_step_change"
TARGET_LENGTH = 1440  # 24 hours in minutes
STEP_CHANGE_POS = 400  # Position where step change should occur

# Your reference columns (77 columns from your CSV)
REFERENCE_COLS = [
    'i', 'j', 'steps', 'acidgas_Fm', 'acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S', 
    'acidgas_T', 'acidgas_P', 'air', 'air_SP', 'second_air2', 'air2_SP', 'COG', 
    'COG_SP', 'burner_input_T_SP', 'burner_input_T_PV', 'burner_inputP', 
    'burner_output_T_SP', 'burner_output_T_PV', 'burner_output_P_SP', 
    'burner_output_P_PV', 'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 
    'fur_outputT', 'fur_outputP_SP', 'fur_outputP_PV', 'WHB_F', 'WHB_inputT', 
    'WHB_inputP', 'WHB_outputT', 'WHB_outputP', 'SEP1_F', 'SEP1_P_SP', 
    'SEP1_P_PV', 'SEP1_T', 'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 
    'HEATER1_output_T_SP', 'HEATER1_output_T_PV', 'HEATER1_output_P', 'cat1_F', 
    'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_output_P_SP', 
    'cat1_output_P_PV', 'cat1_deltaP', 'SEP2_F', 'SEP2_P_SP', 'SEP2_P_PV', 
    'SEP2_T', 'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 
    'HEATER2_output_T_SP', 'HEATER2_output_T_PV', 'HEATER2_output_P', 'cat2_F', 
    'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_output_P_SP', 
    'cat2_output_P_PV', 'cat2_deltaP', 'SEP3_F', 'SEP3_P_SP', 'SEP3_P_PV', 
    'SEP3_T', 'B35_H2S', 'B35_SO2', 'ratio', 'ratioSP', 'conv'
]


def find_step_change_index(df, change_col):
    """Find the index where the step change occurs in the MV column."""
    if change_col not in df.columns:
        return None
    
    # Look for significant change in the column
    changes = df[change_col].diff().abs() > 0.01
    if changes.any():
        return df.index[changes].tolist()[0]
    return None


def parse_filename(filename):
    """Parse filename to extract operating point and change info.
    Format: air2_190_t2_155_air2_change_-5.xlsx
    """
    # Handle typo in one file: "air_change" instead of "air2_change"
    filename = filename.replace("air_change", "air2_change")
    
    pattern = r"air2_(\d+)_t2_(\d+)_(air2|t2)_change_(-?\d+)"
    match = re.search(pattern, filename)
    if match:
        air2_base = int(match.group(1))
        t2_base = int(match.group(2))
        change_var = match.group(3)  # 'air2' or 't2'
        change_amount = int(match.group(4))
        return {
            'air2_base': air2_base,
            't2_base': t2_base,
            'change_var': change_var,
            'change_amount': change_amount
        }
    return None


def process_file(filepath):
    """Process a single senpai xlsx file."""
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    
    # Parse filename
    info = parse_filename(filename)
    if info is None:
        print(f"  [SKIP] Cannot parse filename format")
        return None
    
    print(f"  Operating point: air2={info['air2_base']}, t2={info['t2_base']}")
    print(f"  Step change: {info['change_var']} changes by {info['change_amount']}")
    
    # Load Excel file
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"  [SKIP] Error reading file: {e}")
        return None
    
    print(f"  Original shape: {df.shape}")
    
    # Check if we have the required columns (allow extra columns, we'll filter later)
    missing_cols = [c for c in REFERENCE_COLS if c not in df.columns]
    if missing_cols:
        print(f"  [SKIP] Missing required columns: {missing_cols[:5]}...")
        return None
    
    # Keep only the reference columns (77 columns)
    df = df[REFERENCE_COLS].copy()
    print(f"  Filtered shape: {df.shape}")
    
    # Determine which column has the step change
    change_col = 'air2_SP' if info['change_var'] == 'air2' else 'HEATER2_output_T_SP'
    
    # Find step change index in original data
    original_step_idx = find_step_change_index(df, change_col)
    if original_step_idx is None:
        print(f"  [SKIP] Cannot find step change in {change_col}")
        return None
    
    print(f"  Step change found at index: {original_step_idx}")
    print(f"  {change_col} before: {df.loc[original_step_idx-1, change_col]:.2f}")
    print(f"  {change_col} after: {df.loc[original_step_idx, change_col]:.2f}")
    
    # Calculate how many rows before and after step change
    rows_before_step = original_step_idx
    rows_after_step = len(df) - original_step_idx
    
    # We need: STEP_CHANGE_POS rows before, (TARGET_LENGTH - STEP_CHANGE_POS) rows after
    need_before = STEP_CHANGE_POS
    need_after = TARGET_LENGTH - STEP_CHANGE_POS
    
    print(f"  Need: {need_before} rows before step, {need_after} rows after step")
    print(f"  Available: {rows_before_step} rows before, {rows_after_step} rows after")
    
    # Check if we have enough data - use all available if not enough
    if rows_before_step < need_before:
        print(f"  [WARN] Not enough data before step change ({rows_before_step} < {need_before}), using all available")
        actual_before = rows_before_step
    else:
        actual_before = need_before
    
    if rows_after_step < need_after:
        print(f"  [WARN] Not enough data after step change ({rows_after_step} < {need_after}), using all available")
        actual_after = rows_after_step
    else:
        actual_after = need_after
    
    # Extract the window
    start_idx = original_step_idx - actual_before
    end_idx = original_step_idx + actual_after
    
    df_out = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    # Update i, j, steps columns to match your format
    df_out['i'] = 0.0
    df_out['j'] = df_out.index.astype(float)
    df_out['steps'] = df_out.index.astype(float)
    
    # Mark rows after step change with i=1 (optional, depending on your format)
    # df_out.loc[STEP_CHANGE_POS:, 'i'] = 1.0
    
    print(f"  Output shape: {df_out.shape}")
    
    # Verify step change is at correct position
    new_step_idx = find_step_change_index(df_out, change_col)
    print(f"  Step change now at index: {new_step_idx} (target: {STEP_CHANGE_POS})")
    
    if new_step_idx != STEP_CHANGE_POS:
        print(f"  [WARN] Step change position mismatch!")
    
    return df_out, info


def main():
    # Open log file
    log_file = open("convert_senpai_log.txt", "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
    
    log("=" * 60)
    log("Converting Senpai's XLSX files to your CSV format")
    log(f"Target: {TARGET_LENGTH} rows, step change at position {STEP_CHANGE_POS}")
    log("=" * 60)
    
    # Find all xlsx files
    files = glob.glob(os.path.join(SENPAI_DIR, "*.xlsx"))
    log(f"Found {len(files)} xlsx files")
    
    success_count = 0
    skip_count = 0
    skipped_files = []
    
    for f in files:
        result = process_file(f)
        
        if result is None:
            skip_count += 1
            skipped_files.append(os.path.basename(f))
            continue
        
        df_out, info = result
        
        # Generate output filename matching your format
        change_str = f"{info['change_var']}_change_{info['change_amount']}"
        out_filename = f"air2_{info['air2_base']}_t2_{info['t2_base']}_{change_str}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        
        # Save CSV
        df_out.to_csv(out_path, index=False)
        log(f"  [OK] Saved: {out_filename}")
        success_count += 1
    
    log("\n" + "=" * 60)
    log(f"Summary: {success_count} converted, {skip_count} skipped")
    if skipped_files:
        log(f"Skipped files: {skipped_files}")
    log("=" * 60)
    
    log_file.close()
    print("\nLog saved to convert_senpai_log.txt")


if __name__ == "__main__":
    main()
