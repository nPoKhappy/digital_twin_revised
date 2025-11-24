"""
計算輸入變數之間的相關性
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_correlation(csv_path: str, output_dir: str = './results/correlation'):
    """
    計算並視覺化變數之間的相關性
    
    Args:
        csv_path: 資料檔案路徑
        output_dir: 輸出目錄
    """
    # 讀取資料
    print(f"[Info] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 定義要分析的欄位
    input_cols = [
        'acidgas_Fm',          # 酸氣流量
        'acidgas_T',           # 酸氣溫度
        'acidgas_P',           # 酸氣壓力
        'air2_SP',             # MV1
        'HEATER2_output_T_SP'  # MV2
    ]
    
    # 檢查欄位是否存在
    missing_cols = [c for c in input_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # 選取相關欄位並移除缺失值
    df_selected = df[input_cols].copy()
    print(f"[Info] Original shape: {df_selected.shape}")
    df_selected = df_selected.dropna()
    print(f"[Info] After dropna: {df_selected.shape}")
    
    # 計算 Pearson 相關係數
    corr_pearson = df_selected.corr(method='pearson')
    print("\n=== Pearson Correlation Matrix ===")
    print(corr_pearson)
    
    # 計算 Spearman 相關係數（對非線性關係更敏感）
    corr_spearman = df_selected.corr(method='spearman')
    print("\n=== Spearman Correlation Matrix ===")
    print(corr_spearman)
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存相關係數矩陣
    corr_pearson.to_csv(os.path.join(output_dir, 'correlation_pearson.csv'))
    corr_spearman.to_csv(os.path.join(output_dir, 'correlation_spearman.csv'))
    print(f"\n[OK] Saved correlation matrices to: {output_dir}")
    
    # 視覺化：Pearson
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_pearson.png'), dpi=150)
    print(f"[OK] Saved Pearson heatmap to: {os.path.join(output_dir, 'correlation_pearson.png')}")
    plt.close()
    
    # 視覺化：Spearman
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_spearman, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_spearman.png'), dpi=150)
    print(f"[OK] Saved Spearman heatmap to: {os.path.join(output_dir, 'correlation_spearman.png')}")
    plt.close()
    
    # 找出高度相關的變數對（絕對值 > 0.7）
    print("\n=== Highly Correlated Pairs (|r| > 0.7) ===")
    high_corr = []
    for i in range(len(corr_pearson.columns)):
        for j in range(i+1, len(corr_pearson.columns)):
            corr_val = corr_pearson.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append({
                    'Var1': corr_pearson.columns[i],
                    'Var2': corr_pearson.columns[j],
                    'Pearson_r': corr_val
                })
    
    if high_corr:
        df_high_corr = pd.DataFrame(high_corr)
        print(df_high_corr)
        df_high_corr.to_csv(os.path.join(output_dir, 'high_correlation_pairs.csv'), index=False)
    else:
        print("No variable pairs with |correlation| > 0.7")
    
    # 基本統計資訊
    print("\n=== Descriptive Statistics ===")
    stats = df_selected.describe()
    print(stats)
    stats.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))
    
    print(f"\n[Done] All results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='計算輸入變數之間的相關性')
    parser.add_argument('--csv', type=str, default='data/my_own_data/training.csv',
                        help='輸入 CSV 檔案路徑')
    parser.add_argument('--output', type=str, default='results/correlation',
                        help='輸出目錄')
    args = parser.parse_args()
    
    compute_correlation(args.csv, args.output)


if __name__ == '__main__':
    main()
