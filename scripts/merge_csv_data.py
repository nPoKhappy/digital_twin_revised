# merge_csv_data.py - 合併多個 CSV 文件 # 合併3個訓練文件和1個測試文件 (測試數據是訓練數據分布外的 動態數據)
# claus 製程專用
import pandas as pd
import os
from pathlib import Path

def merge_csv_files(input_folder, file_patterns, output_file, test_file=None, test_output_file=None):
    """
    合併多個 CSV 文件
    
    Args:
        input_folder: 輸入文件夾路徑
        file_patterns: 要合併的文件名列表 (例如 ['5', '5-2', '5-6'])
        output_file: 輸出的合併文件路徑
        test_file: 測試文件名 (例如 '5-1')，如果提供則會單獨處理
        test_output_file: 測試文件輸出路徑
    """
    
    print("="*70)
    print("CSV 文件合併工具")
    print("="*70)
    
    # 創建輸出目錄（如果不存在）
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 儲存所有數據框
    dfs = []
    total_rows = 0
    
    print(f"\n輸入文件夾: {input_folder}")
    print(f"輸出文件: {output_file}")
    
    # 讀取並合併訓練文件
    print(f"\n{'='*70}")
    print(f"讀取訓練數據文件:")
    print(f"{'='*70}")
    
    for pattern in file_patterns:
        # 嘗試不同的文件擴展名
        possible_files = [
            os.path.join(input_folder, f"Test_dataform_change_air2_R={pattern}.csv"),
            os.path.join(input_folder, f"Test_dataform_change_air2_R={pattern}.CSV"),
        ]
        
        file_path = None
        for pf in possible_files:
            if os.path.exists(pf):
                file_path = pf
                break
        
        if file_path is None:
            print(f"⚠️  找不到文件: Test_dataform_change_air2_R={pattern}.csv")
            continue
        
        try:
            df = pd.read_csv(file_path)
            rows = len(df)
            cols = len(df.columns)
            
            print(f"✓ Test_dataform_change_air2_R={pattern}.csv: {rows:,} 行 × {cols} 列")
            
            # 顯示欄位名稱（僅第一個文件）
            if len(dfs) == 0:
                print(f"\n  欄位名稱:")
                for i, col in enumerate(df.columns, 1):
                    print(f"    {i:2d}. {col}")
            
            dfs.append(df)
            total_rows += rows
            
        except Exception as e:
            print(f"✗ 讀取 Test_dataform_change_air2_R={pattern}.csv 失敗: {e}")
    
    if not dfs:
        print("\n✗ 沒有成功讀取任何文件！")
        return False
    
    # 合併數據
    print(f"\n{'='*70}")
    print(f"合併數據:")
    print(f"{'='*70}")
    
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    print(f"合併後總行數: {len(merged_df):,}")
    print(f"合併後總列數: {len(merged_df.columns)}")
    
    # 檢查缺失值
    missing_counts = merged_df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  發現缺失值:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count} 個缺失值")
    else:
        print(f"✓ 無缺失值")
    
    # 保存合併後的文件
    try:
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✓ 訓練數據已保存至: {output_file}")
        print(f"  文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"\n✗ 保存文件失敗: {e}")
        return False
    
    # 處理測試文件（如果提供）
    if test_file:
        print(f"\n{'='*70}")
        print(f"處理測試數據:")
        print(f"{'='*70}")
        
        test_path = os.path.join(input_folder, f"Test_dataform_change_air2_R={test_file}.csv")
        if not os.path.exists(test_path):
            test_path = os.path.join(input_folder, f"Test_dataform_change_air2_R={test_file}.CSV")
        
        if os.path.exists(test_path):
            try:
                test_df = pd.read_csv(test_path)
                test_rows = len(test_df)
                
                print(f"✓ Test_dataform_change_air2_R={test_file}.csv: {test_rows:,} 行 × {len(test_df.columns)} 列")
                
                # 保存測試文件到指定路徑
                if test_output_file is None:
                    test_output_file = os.path.join('data', 'rolling_data.csv')
                
                # 創建測試輸出目錄（如果不存在）
                test_output_dir = Path(test_output_file).parent
                test_output_dir.mkdir(parents=True, exist_ok=True)
                
                test_df.to_csv(test_output_file, index=False, encoding='utf-8')
                
                print(f"✓ 測試數據已保存至: {test_output_file}")
                print(f"  文件大小: {os.path.getsize(test_output_file) / 1024 / 1024:.2f} MB")
                
            except Exception as e:
                print(f"✗ 處理測試文件失敗: {e}")
        else:
            print(f"⚠️  找不到測試文件: Test_dataform_change_air2_R={test_file}.csv")
    
    # 數據摘要
    print(f"\n{'='*70}")
    print(f"數據摘要:")
    print(f"{'='*70}")
    print(f"訓練數據: {len(merged_df):,} 行")
    if test_file:
        print(f"測試數據: {test_rows:,} 行")
    print(f"\n數據統計:")
    print(merged_df.describe())
    
    print(f"\n{'='*70}")
    print(f"✓ 數據合併完成！")
    print(f"{'='*70}")
    
    return True

def main():
    """主函數"""
    
    # 配置參數
    input_folder = "data/senpai_data"  # 輸入文件夾
    train_patterns = ['5', '5-2', '5-6']  # 訓練數據文件名（不含副檔名）
    test_pattern = '5-1'  # 測試數據文件名
    output_file = "data/senpai_data/training.csv"  # 輸出的訓練文件
    test_output_file = "data/senpai_data/testing.csv"  # 輸出的測試文件

    print("\n配置資訊:")
    print(f"  輸入文件夾: {input_folder}")
    print(f"  訓練文件: {', '.join(train_patterns)}")
    print(f"  測試文件: {test_pattern}")
    print(f"  輸出訓練文件: {output_file}")
    print(f"  輸出測試文件: {test_output_file}")
    
    # 檢查輸入文件夾是否存在
    if not os.path.exists(input_folder):
        print(f"\n✗ 錯誤: 找不到文件夾 '{input_folder}'")
        print(f"請確認路徑是否正確。")
        return
    
    # 合併文件
    success = merge_csv_files(
        input_folder=input_folder,
        file_patterns=train_patterns,
        output_file=output_file,
        test_file=test_pattern,
        test_output_file=test_output_file
    )
    
    if success:
        print(f"\n💡 下一步:")
        print(f"  1. 檢查合併後的文件: {output_file}")
        print(f"  2. 確認欄位名稱與配置文件一致")
        print(f"  3. 開始訓練:")
        print(f"     python train.py --config configs/ann_claus_experiment.yaml")
    else:
        print(f"\n✗ 數據合併失敗，請檢查錯誤訊息")

if __name__ == "__main__":
    main()
