# inspect_senpai_xlsx.py - Quick inspection of senpai xlsx files
import pandas as pd
import os
import glob

senpai_dir = "data/Claus_dynamic/in_training_distribution_step_change/senpai_xlsx_file"

# Check the two skipped files
skip_files = ['air2_190_t2_155_t2_change_-5.xlsx', 'air2_270_t2_155_air2_change_-5.xlsx']

with open("senpai_inspection.txt", "w") as out:
    for fname in skip_files:
        f = os.path.join(senpai_dir, fname)
        if not os.path.exists(f):
            out.write(f"File not found: {fname}\n\n")
            continue
            
        out.write(f"File: {fname}\n")
        df = pd.read_excel(f)
        out.write(f"Shape: {df.shape}\n")
        
        # Check step change position for air2_SP
        if 'air2_SP' in df.columns:
            air2_changes = df['air2_SP'].diff().abs() > 0.1
            if air2_changes.any():
                idx = df.index[air2_changes].tolist()[0]
                out.write(f"air2_SP step change at index: {idx}\n")
                out.write(f"  Rows before: {idx}, Rows after: {len(df) - idx}\n")
                out.write(f"  Need before: 400, Need after: 1040\n")
            else:
                out.write(f"air2_SP: NO step change found\n")
        
        # Check step change position for HEATER2_output_T_SP
        if 'HEATER2_output_T_SP' in df.columns:
            t2_changes = df['HEATER2_output_T_SP'].diff().abs() > 0.1
            if t2_changes.any():
                idx = df.index[t2_changes].tolist()[0]
                out.write(f"HEATER2_output_T_SP step change at index: {idx}\n")
                out.write(f"  Rows before: {idx}, Rows after: {len(df) - idx}\n")
                out.write(f"  Need before: 400, Need after: 1040\n")
            else:
                out.write(f"HEATER2_output_T_SP: NO step change found\n")
        
        out.write("\n")

print("Results saved to senpai_inspection.txt")

