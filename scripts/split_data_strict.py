
import pandas as pd
import os

# Config
data_dir = r"c:\Users\Administrator\Desktop\digital_twin_revised\data\Claus_dynamic"
source_file = "Test_dataform_change_air2_R=5.csv"
total_points = 15000
train_ratio = 0.8
split_idx = int(total_points * train_ratio)

# Read
path = os.path.join(data_dir, source_file)
df = pd.read_csv(path)

# Ensure data length
if len(df) > total_points:
    df = df.iloc[:total_points]

# Split
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

# Save
train_fname = "R5_Train_Part.csv"
test_fname = "R5_ID_Test_Part.csv"

df_train.to_csv(os.path.join(data_dir, train_fname), index=False)
df_test.to_csv(os.path.join(data_dir, test_fname), index=False)

print(f"Split done.")
print(f"Train part ({train_fname}): {len(df_train)} rows")
print(f"Test part ({test_fname}): {len(df_test)} rows")
