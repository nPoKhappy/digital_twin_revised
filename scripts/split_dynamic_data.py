import os
import pandas as pd
import glob

def main():
    # 設定路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'Claus_dynamic')
    
    # 尋找目標檔案
    pattern = os.path.join(data_dir, 'Test_dataform_change_air2_R=*.csv')
    files = glob.glob(pattern)
    
    # 過濾掉已經分割過的檔案
    target_files = [f for f in files if '_split' not in os.path.basename(f)]
    
    for file_path in target_files:
        print(f"正在處理: {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path)
        
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_valid = int(n_total * 0.1)
        
        # 依序切分 (保留時間序列順序)
        train_df = df.iloc[:n_train]
        valid_df = df.iloc[n_train:n_train + n_valid]
        test_df = df.iloc[n_train + n_valid:]
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        train_path = os.path.join(data_dir, f"{base_name}_train_split.csv")
        valid_path = os.path.join(data_dir, f"{base_name}_valid_split.csv")
        test_path = os.path.join(data_dir, f"{base_name}_test_split.csv")
        
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"  -> Train (0.8): {len(train_df)} 筆, 儲存為 {os.path.basename(train_path)}")
        print(f"  -> Valid (0.1): {len(valid_df)} 筆, 儲存為 {os.path.basename(valid_path)}")
        print(f"  -> Test  (0.1): {len(test_df)} 筆, 儲存為 {os.path.basename(test_path)}")
        print("-" * 50)

if __name__ == '__main__':
    main()
