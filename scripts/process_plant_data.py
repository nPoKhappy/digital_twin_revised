import pandas as pd
import numpy as np
import os

def process_plant_data():
    input_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area.xlsx'
    output_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area_processed.csv'
    
    print(f"Reading {input_path}...")
    # Read with header=1 to get Chinese names, skip row 2 (Tags) by dropping it later or filtering
def process_sheet(df, sheet_name, output_base_path):
    print(f"\nProcessing Sheet: {sheet_name}")
    print(f"Initial shape: {df.shape}")
    
    # Drop the first row which contains tags like 'W25_FI6509.PV' (it becomes row 0 after header=1)
    df = df.iloc[1:].reset_index(drop=True)
    print(f"Shape after removing tag row: {df.shape}")
    
    # Check actual column names (stripped)
    df.columns = [str(c).strip() for c in df.columns]
    
    print(f"Original Columns: {df.columns.tolist()}")

    rename_map = {
        'Unnamed: 0': 'DateTime',
        '流量': 'acidgas_Fm',
        '溫度': 'acidgas_T',
        '壓力': 'acidgas_P',
        '二次空氣': 'second_air2',
        '一次空氣': 'air',
        'Oven頂溫': 'burner_output_T_PV',
        '觸媒層溫度': 'fur_temp',
        'NO1入口溫度': 'cat1_input_temp',
        'NO1出口溫度': 'cat1_output_temp',
        'NO1壓差': 'cat1_deltaP',
        'NO1溫度': 'cat1_bed_temp',
        'NO2入口溫度': 'cat2_input_temp',
        'NO2出口溫度': 'cat2_output_temp',
        'NO2壓差': 'cat2_deltaP',
        'NO2溫度': 'cat2_bed_temp',
        'H2S': 'B35_H2S',
        'SO2': 'B35_SO2'
    }
    
    df.rename(columns=rename_map, inplace=True)
    # print(f"Renamed Columns: {df.columns.tolist()}")
    
    # Ensure DateTime is parsed with error handling
    print("Parsing DateTime...")
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    # Drop rows with invalid DateTime
    len_before = len(df)
    invalid_datetime_rows = df[df['DateTime'].isna()]
    if not invalid_datetime_rows.empty:
        print(f"\n[Warning] Found {len(invalid_datetime_rows)} rows with invalid DateTime:")
        print(invalid_datetime_rows.head())  # Print sample of dropped rows
    
    df.dropna(subset=['DateTime'], inplace=True)
    len_after = len(df)
    print(f"Dropped {len_before - len_after} rows with invalid DateTime.")
    
    df.sort_values('DateTime', inplace=True)

    # Convert numeric columns
    numeric_cols = [
        'acidgas_Fm', 'acidgas_T', 'acidgas_P', 'second_air2', 
        'cat1_input_temp', 'cat2_input_temp', 
        'B35_H2S', 'B35_SO2',
        'air', 'fur_top', 'fur_temp',
        'cat1_output_temp', 'cat1_deltaP', 'cat1_bed_temp',
        'cat2_output_temp', 'cat2_deltaP', 'cat2_bed_temp'
    ]
    
    # Filter numeric_cols to only those present in df to avoid KeyErrors
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with NaNs in critical columns
    len_before_nan = len(df)
    
    # Check which rows have NaNs
    nan_mask = df[numeric_cols].isna().any(axis=1)
    nan_rows = df[nan_mask]
    
    if not nan_rows.empty:
        print(f"\n[Warning] Found {len(nan_rows)} rows with NaNs in numeric columns:")
        # Check which columns specifically have NaNs
        nan_cols = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
        print(f"Columns containing NaNs: {nan_cols}")
        print("Sample of dropped rows (showing NaN columns):")
        print(nan_rows[nan_cols + ['DateTime']].head())
    
    df.dropna(subset=numeric_cols, inplace=True)
    len_after_nan = len(df)
    print(f"Dropped {len_before_nan - len_after_nan} rows with NaNs in numeric columns.")
    
    print(f"Final shape for {sheet_name}: {df.shape}")
    
    # ---------------------------------------------------------
    # Check for Time Continuity and Split if necessary
    # ---------------------------------------------------------
    df = df.reset_index(drop=True)
    time_diffs = df['DateTime'].diff()
    
    if len(df) > 1:
        median_interval = time_diffs.median()
        print(f"Median time interval: {median_interval}")
        
        # Define gap threshold (strict continuity: > 1.5 * median)
        # Assuming regular 1-min data, any gap >= 2 mins is a break.
        threshold = median_interval * 1.5
        
        # Find indices where gap occurs
        # gap_indices points to the first row of the NEW segment
        # shift(-1) or just check diff at current index? 
        # df['DateTime'].diff() at index i is time(i) - time(i-1).
        # If diff[i] > threshold, then there is a gap between i-1 and i.
        gap_indices = df.index[time_diffs > threshold].tolist()
    else:
        gap_indices = []
        
    # Construct output path base
    sheet_clean = sheet_name.replace(' ', '_')
    output_path_base = output_base_path.replace('.csv', f'_{sheet_clean}')
    
    if not gap_indices:
        # No gaps, save as single file
        final_output_path = f"{output_path_base}.csv"
        df.to_csv(final_output_path, index=False)
        print(f"Time is continuous. Saved to {final_output_path}")
    else:
        print(f"Found {len(gap_indices)} gaps. Splitting into {len(gap_indices) + 1} continuous segments.")
        
        start_idx = 0
        gap_indices.append(len(df)) # Add end index for loop
        
        for i, end_idx in enumerate(gap_indices):
            segment_df = df.iloc[start_idx:end_idx]
            
            # Save segment
            if not segment_df.empty:
                part_path = f"{output_path_base}_part{i+1}.csv"
                segment_df.to_csv(part_path, index=False)
                
                # Info
                start_time = segment_df['DateTime'].iloc[0]
                end_time = segment_df['DateTime'].iloc[-1]
                print(f"  Saved Part {i+1}: {len(segment_df)} rows ({start_time} to {end_time}) -> {part_path}")
            
            start_idx = end_idx

def process_plant_data():
    input_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area.xlsx'
    output_base_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area_processed.csv'
    
    print(f"Reading {input_path}...")
    
    # Load specific sheets
    sheets_to_process = ['Training Data', 'Test Data']
    
    try:
        # Read all sheets at once or loop
        xls = pd.ExcelFile(input_path)
        print(f"Found sheets: {xls.sheet_names}")
        
        for sheet in sheets_to_process:
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=1)
                process_sheet(df, sheet, output_base_path)
            else:
                print(f"Warning: Sheet '{sheet}' not found in Excel file.")
                
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    process_plant_data()
