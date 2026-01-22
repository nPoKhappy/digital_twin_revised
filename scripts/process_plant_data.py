import pandas as pd
import numpy as np
import os

def process_plant_data():
    input_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area.xlsx'
    output_path = 'data/Claus_dynamic/Claus_plant_data/W251_500area_processed_for_GRU.csv'
    
    print(f"Reading {input_path}...")
    # Read with header=1 to get Chinese names, skip row 2 (Tags) by dropping it later or filtering
    df = pd.read_excel(input_path, header=1)
    
    # Drop the first row which contains tags like 'W25_FI6509.PV' (it becomes row 0 after header=1)
    df = df.iloc[1:].reset_index(drop=True)
    
    # Rename columns map
    # '流量': 'acidgas_Fm'
    # '溫度': 'acidgas_T'
    # '壓力': 'acidgas_P'
    # '二次空氣': 'second_air2'
    # 'NO1入口溫度': 'HEATER1_output_T_PV' 
    # 'NO2入口溫度': 'HEATER2_output_T_PV' 
    # 'H2S': 'B35_H2S'
    # 'SO2': 'B35_SO2'
    
    # Check actual column names (stripped)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Rename columns map
    # ... (same as before)
    
    rename_map = {
        'Unnamed: 0': 'DateTime',
        '流量': 'acidgas_Fm',
        '溫度': 'acidgas_T',
        '壓力': 'acidgas_P',
        '二次空氣': 'second_air2',
        'NO1入口溫度': 'HEATER1_output_T_PV',
        'NO2入口溫度': 'HEATER2_output_T_PV',
        'H2S': 'B35_H2S',
        'SO2': 'B35_SO2'
    }
    
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure DateTime is parsed with error handling
    print("Parsing DateTime...")
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    # Drop rows with invalid DateTime
    len_before = len(df)
    df.dropna(subset=['DateTime'], inplace=True)
    len_after = len(df)
    print(f"Dropped {len_before - len_after} rows with invalid DateTime.")
    
    df.sort_values('DateTime', inplace=True)
    
    # Create missing SP columns by copying PVs
    # ... (rest is same)
    if 'air2_SP' not in df.columns:
        df['air2_SP'] = df['second_air2']
        
    if 'HEATER2_output_T_SP' not in df.columns:
        df['HEATER2_output_T_SP'] = df['HEATER2_output_T_PV']

    # Also map T2_SPRemote if needed, but our variable_selection uses HEATER2_output_T_SP
    df['T2_SPRemote'] = df['HEATER2_output_T_SP'] 
    df['T2'] = df['HEATER2_output_T_PV']

    # Convert numeric columns with unit conversion to match Aspen Training Data
    
    # Convert numeric columns with unit conversion to match Aspen Training Data
    
    # Convert numeric columns
    # NOTE: Keeping data in ORIGINAL PLANT UNITS as requested for analysis.
    # Flows: Nm3/hr
    # Pressure: mean 175 (likely kPa)
    # Composition: mean 0.49 (likely %)
    
    # No conversion applied here.
    
    numeric_cols = [
        'acidgas_Fm', 'acidgas_T', 'acidgas_P', 'second_air2', 'air2_SP',
        'HEATER1_output_T_PV', 'HEATER2_output_T_PV', 'HEATER2_output_T_SP',
        'B35_H2S', 'B35_SO2'
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with NaNs in critical columns
    df.dropna(subset=numeric_cols, inplace=True)
    
    print(f"Data shape after cleaning: {df.shape}")
    
    # Save processed CSV
    # We save it raw first (1-minute data). 
    # The training script (predict_resampled.py) will handle 10-min median downsampling automatically if configured!
    # But to be safe and consistent with "training files", maybe we can let the script handle it.
    # However, 'predict_resampled.py' has a flag 'use_median_downsampling: true' in config.
    # So we just save the 1-min data here.
    
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_plant_data()
