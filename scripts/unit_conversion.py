import pandas as pd
import numpy as np
import glob
import os

def calculate_conversion_factor():
    # User provided points for Air: kmol/hr -> m^3/hr
    # (kmol_val, m3_val)
    points = [
        (123.295, 2326.15),
        (140.4, 2649),
        (149.5, 2814),
        (158, 2980),
        (166.7, 3145)
    ]
    
    # Calculate ratios
    ratios = [m3 / kmol for kmol, m3 in points]
    factor = np.mean(ratios)
    
    # Optional: Check consistency
    print("Conversion Factor Calculation:")
    for (kmol, m3), r in zip(points, ratios):
        print(f"  {kmol} -> {m3} (Ratio: {r:.4f})")
    print(f"  Average Factor: {factor:.6f}")
    
    return factor

def convert_units():
    search_pattern = 'data/Claus_dynamic/Test_dataform_change_air2_R=*.csv'
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching {search_pattern}")
        return

    factor = calculate_conversion_factor()
    
    # Columns to convert
    # User asked for 'acidgas_Fm' and 'air'
    # 'acidgas_Fm' logic is inferred to use the same factor as 'air' based on request context.
    cols_to_convert = ['acidgas_Fm', 'air']
    
    for file_path in files:
        print(f"\nProcessing {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Check if columns exist
            processing_cols = [c for c in cols_to_convert if c in df.columns]
            
            if not processing_cols:
                print(f"  Skipping: Targeted columns {cols_to_convert} not found.")
                continue
                
            for col in processing_cols:
                print(f"  Converting column '{col}'...")
                # Apply conversion
                df[col] = df[col] * factor
                
            # Save the file (Overwrite or new file?)
            # I will save to a new file to be safe: filename_converted.csv
            # But the user might want inplace. Let's create a new file for now.
            
            # Construct output path
            dir_name, file_name = os.path.split(file_path)
            base_name, ext = os.path.splitext(file_name)
            output_path = os.path.join(dir_name, f"{base_name}_converted{ext}")
            
            df.to_csv(output_path, index=False)
            print(f"  Saved converted file to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

if __name__ == "__main__":
    convert_units()
