# debug_initial_point.py - 檢查預測起始點對應關係
import pandas as pd
import numpy as np

# 讀取一個預測結果
result_file = 'results/step_change_predictions/in_training_distribution/air2_180_t2_150_air2_change_10/predictions.csv'
df_pred = pd.read_csv(result_file)

# 讀取原始數據
orig_file = 'data/Claus_dynamic/step_change/in_training_distribution/air2_180_t2_150_air2_change_10.csv'
df_orig = pd.read_csv(orig_file)

print('=== 預測結果檔案 ===')
print(f'Shape: {df_pred.shape}')
print(f'前5行 B35_H2S (true vs pred):')
print(df_pred[['B35_H2S', 'B35_H2S_pred']].head())

print('\n=== 原始數據 ===')
print(f'Shape: {df_orig.shape}')

# 窗口大小 W = 18
W = 18
print(f'\n=== 對應關係 (W={W}) ===')
print(f'Encoder 使用原始數據 index 0~{W-1} (共 {W} 行) 作為歷史窗口')
print(f'預測從原始數據 index {W} 開始')
print()
print(f'原始數據 index {W} 的 B35_H2S = {df_orig["B35_H2S"].iloc[W]:.8f}')
print(f'預測結果 index 0 的 B35_H2S (true) = {df_pred["B35_H2S"].iloc[0]:.8f}')
print(f'預測結果 index 0 的 B35_H2S (pred) = {df_pred["B35_H2S_pred"].iloc[0]:.8f}')
print()

# 計算差異
true_first = df_pred["B35_H2S"].iloc[0]
pred_first = df_pred["B35_H2S_pred"].iloc[0]
diff = abs(pred_first - true_first)
print(f'第一個預測點的誤差: {diff:.8f}')
print(f'相對誤差: {diff/true_first*100:.2f}%')

# 看看最後一個 encoder 輸入點
print(f'\n=== Encoder 輸入的最後一個點 (index {W-1}) ===')
print(f'原始數據 index {W-1} 的 B35_H2S = {df_orig["B35_H2S"].iloc[W-1]:.8f}')

# 這個才是模型「看到」的最後一個真實值
# 模型應該要從這個點預測下一個點
print(f'\n=== 問題分析 ===')
print('Sliding window 策略:')
print(f'  1. Encoder 輸入: 原始數據 [0:{W}], 最後看到 index {W-1}')
print(f'  2. 第一個預測: 應該是 index {W} 的值')
print(f'  3. 如果初始點對不上, 可能是模型沒學好這個轉換')
print()

# 比較 encoder 最後一點 vs 預測第一點
enc_last = df_orig["B35_H2S"].iloc[W-1]
print(f'Encoder 最後看到的值: {enc_last:.8f}')
print(f'預測的第一個值:       {pred_first:.8f}')
print(f'真實的第一個值:       {true_first:.8f}')
print(f'真實值變化量:         {true_first - enc_last:.8f}')
print(f'預測值變化量:         {pred_first - enc_last:.8f}')
