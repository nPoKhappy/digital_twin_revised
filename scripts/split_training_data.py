# split_training_data.py - 將合併的訓練數據拆分成獨立的段
import pandas as pd
import os

# 配置
INPUT_FILE = "data/Claus_dynamic/training.csv"
OUTPUT_DIR = "data/Claus_dynamic/"
SEGMENT_SIZE = 14400  # 每段的數據點數
NUM_SEGMENTS = 3      # 段數

def main():
    print(f"讀取 {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"總行數: {len(df)}")
    
    for i in range(NUM_SEGMENTS):
        start_idx = i * SEGMENT_SIZE
        end_idx = (i + 1) * SEGMENT_SIZE
        
        if end_idx > len(df):
            print(f"[WARN] 段 {i+1} 數據不足，只有 {len(df) - start_idx} 行")
            end_idx = len(df)
        
        df_segment = df.iloc[start_idx:end_idx]
        
        output_file = os.path.join(OUTPUT_DIR, f"training_segment_{i+1}.csv")
        df_segment.to_csv(output_file, index=False)
        print(f"已保存 {output_file}: {len(df_segment)} 行")

if __name__ == "__main__":
    main()
