# src/utils.py
import pandas as pd
import numpy as np
import torch
import os
from math import sqrt
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# ==============================================================================
# --- 數據處理工具 (來自原 data_utils.py) ---
# ==============================================================================

def load_data(file_path, datetime_tag='DateTime', index_tag='DateTime', slice_interval=1):
    """
    載入 CSV 或 Excel 檔案，解析日期並設置索引。
    如果沒有指定的日期時間列，則使用預設的數值索引。
    """
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)
    
    if datetime_tag in data.columns:
        try:
            # 將日期字符串轉換為 datetime 對象，格式: '年/月/日 時:分'
            data[datetime_tag] = pd.to_datetime(data[datetime_tag], format='%Y/%m/%d %H:%M')
            # 將日期時間欄位設為索引，便於時間序列操作
            data.set_index(index_tag, inplace=True)
            print(f"成功設置 {datetime_tag} 為時間索引")
        except (ValueError, pd.errors.ParserError) as e:
            print(f"警告：無法解析日期格式 {datetime_tag}，使用原始數據: {e}")
    else:
        print(f"注意：數據中沒有找到 '{datetime_tag}' 列，使用數值索引")
    
    return data[::slice_interval]

def select_date_range(df, start_date, end_date):
    """根據開始和結束日期篩選 DataFrame。"""
    df_copy = df.copy()
    index_to_drop = (df_copy.index > end_date) | (df_copy.index < start_date)
    return df_copy.drop(df_copy.index[index_to_drop])

def remove_event_periods(df, event_periods=None):
    """移除指定事件期間的數據。"""
    if event_periods is None or np.size(event_periods) == 0:
        return df
    
    df_copy = df.copy()
    for start_event, end_event in event_periods:
        index_to_drop = (df_copy.index > start_event) & (df_copy.index < end_event)
        df_copy = df_copy.drop(df_copy.index[index_to_drop])
    return df_copy

def shift_time(df, delta_minutes, datetime_tag='DateTime', index_tag='DateTime'):
    """對 DataFrame 的時間戳進行平移。"""
    df_copy = df.copy().reset_index()
    df_copy[datetime_tag] = df_copy[datetime_tag] + timedelta(minutes=delta_minutes)
    return df_copy.set_index(index_tag)

def load_data_safe(file_path, datetime_tag='DateTime', index_tag='DateTime', slice_interval=1):
    """安全地載入 CSV 或 Excel 檔案，自動處理是否存在日期時間列的情況。"""
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)
    has_datetime_index = False
    
    if datetime_tag in data.columns:
        try:
            date_formats = ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']
            for fmt in date_formats:
                try:
                    data[datetime_tag] = pd.to_datetime(data[datetime_tag], format=fmt)
                    break
                except ValueError:
                    continue
            else:
                data[datetime_tag] = pd.to_datetime(data[datetime_tag], infer_datetime_format=True)
            
            data.set_index(index_tag, inplace=True)
            has_datetime_index = True
            print(f"✓ 成功設置 {datetime_tag} 為時間索引")
            
        except (ValueError, pd.errors.ParserError) as e:
            print(f"⚠ 警告：無法解析日期格式，保持數值索引: {e}")
            has_datetime_index = False
    else:
        print(f"ℹ 注意：數據中沒有 '{datetime_tag}' 列，使用數值索引")
        has_datetime_index = False
    
    sampled_data = data[::slice_interval]
    print(f"✓ 成功載入數據：{sampled_data.shape[0]} 行 × {sampled_data.shape[1]} 列")
    return sampled_data, has_datetime_index

def load_data_resample_median(file_path, interval_minutes=10, datetime_tag='DateTime', index_tag='DateTime'):
    """
    載入數據並進行移動窗口中位數重採樣 (無重疊窗口)。
    例如 interval_minutes=10, 則每 10 行取一個中位數作為代表。
    原數據頻率若為 1 分鐘，則輸出後時間跨度變為原來的 1/10，行數減少為 1/10。
    """
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        try:
            data = pd.read_csv(file_path)
        except:
             data = pd.read_csv(file_path, engine='python')
             
    has_datetime_index = False
    
    # 嘗試解析時間列
    if datetime_tag in data.columns:
        try:
            date_formats = ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']
            for fmt in date_formats:
                try:
                    data[datetime_tag] = pd.to_datetime(data[datetime_tag], format=fmt)
                    break
                except ValueError:
                    continue
            else:
                data[datetime_tag] = pd.to_datetime(data[datetime_tag], infer_datetime_format=True)
            
            data.set_index(index_tag, inplace=True)
            has_datetime_index = True
            
        except (ValueError, pd.errors.ParserError):
            print(f"⚠ 警告：無法解析日期格式，保持數值索引")
    
    # --- 核心邏輯：滾動中位數重採樣 ---
    # 使用 rolling().median() 會產生每一行的滾動中位數
    # 我們需要的是無重疊的窗口 (Non-overlapping window)
    # 所以先 rolling，然後按照間隔切片 (slicing)
    
    # 注意：numeric_only=True 是為了防止非數值列報錯
    resampled_data = data.rolling(window=interval_minutes, min_periods=interval_minutes).median()
    
    # 取樣：從第 interval_minutes-1 行開始取，每隔 interval_minutes 取一次
    # 例如 10 分鐘窗口，第 9 行是第一個完整窗口的中位數 (0-9)，第 19 行是第二個 (10-19)...
    resampled_data = resampled_data.iloc[interval_minutes-1::interval_minutes].copy()
    
    # 重設索引以保持連續性 (如果需要的話，或者保留原來的時間標籤)
    # 如果原本有時間索引，這裡切片後時間索引會是該窗口的最後一個時間點，這通常是合理的
    
    resampled_data.dropna(inplace=True)
    print(f"✓ 成功載入並重採樣 ({interval_minutes} min Median)：{resampled_data.shape[0]} 行 (原 {data.shape[0]} 行)")
    
    return resampled_data, has_datetime_index

def calculate_zscore_stats(df):
    """計算 Z-score 標準化所需的均值和標準差。"""
    return df.mean(), df.std()

def apply_zscore(df, mean, std):
    """應用 Z-score 標準化。"""
    std_safe = std + 1e-8
    df_z = (df - mean) / std_safe
    if df_z.isnull().values.any():
        nan_cols = df_z.columns[df_z.isnull().any()].tolist()
        print(f"錯誤：標準化後在以下欄位中發現 NaN: {nan_cols}")
        problem_std_cols = std[std < 1e-9].index.tolist()
        if problem_std_cols:
            print(f"原因分析：以下欄位的標準差接近於零，可能導致數值不穩定: {problem_std_cols}")
    return df_z

def inverse_zscore(df_scaled, mean, std):
    """還原 Z-score 標準化: x * std + mean"""
    return df_scaled * std + mean

def calculate_robust_stats(df):
    """計算 Robust Scaling 所需的中位數和 IQR (Q75-Q25)。"""
    median = df.median()
    q75 = df.quantile(0.75)
    q25 = df.quantile(0.25)
    iqr = q75 - q25
    return median, iqr

def apply_robust_scale(df, median, iqr):
    """應用 Robust Scaling: (x - median) / iqr"""
    iqr_safe = iqr.replace(0, 1) # 避免除以零
    return (df - median) / iqr_safe

def inverse_robust_scale(df_scaled, median, iqr):
    """還原 Robust Scaling"""
    return df_scaled * iqr + median

def apply_log_transform(df, columns):
    """對指定欄位應用 Log 變換 (log(x))。注意：需確保 x > 0"""
    df_new = df.copy()
    for col in columns:
        if col in df_new.columns:
            # 加上一個極小值防止 log(0)
            df_new[col] = np.log(df_new[col] + 1e-9)
    return df_new

def inverse_log_transform(df, columns):
    """還原 Log 變換 (exp(x))"""
    df_new = df.copy()
    for col in columns:
        if col in df_new.columns:
            df_new[col] = np.exp(df_new[col])
    return df_new

# ==============================================================================
# --- 評估與預測工具 (來自原 utils.py) ---
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """計算單一序列的 MAE, RMSE, R2, MAPE 並回傳字典"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), epsilon)))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def generate_results(model, loader, device, config, mean, std, y_tags, de_mv_tags, prefix, set_name):
    """(舊代碼) 批量生成結果並保存圖表和指標"""
    print(f"開始評估 {set_name}...")
    model.eval()
    
    all_predictions_list = []
    all_targets_list = []

    with torch.no_grad():
        for en_input, de_inputs, targets in tqdm(loader, desc=f"Predicting on {set_name} set"):
            en_input = en_input.to(device)
            all_future_mvs = de_inputs.to(device)
            
            _, encoder_hiddens = model.encoder(en_input)
            all_preds_tensor = model.decoder(all_future_mvs, encoder_hiddens)
            
            all_predictions_list.append(all_preds_tensor.cpu().numpy())
            all_targets_list.append(targets.numpy())

    y_pred_cov = np.concatenate(all_predictions_list, axis=0)
    y_true_cov = np.concatenate(all_targets_list, axis=0)
    
    y_mean_series = mean[y_tags]
    y_std_series = std[y_tags]
    y_pred_cov = y_pred_cov * y_std_series.values + y_mean_series.values
    y_true_cov = y_true_cov * y_std_series.values + y_mean_series.values
    
    pred_len = config['window']['prediction_length']
    num_output = len(y_tags)
    
    for t in tqdm(range(pred_len), desc=f"為 {set_name} 繪圖並計算指標"):
        metrics = {'R2': [], 'MAPE': [], 'RMSE': [], 'MAE': []}
        
        for yi in range(num_output):
            true = y_true_cov[:, t, yi]
            pred = y_pred_cov[:, t, yi]

            metrics['RMSE'].append(sqrt(mean_squared_error(true, pred)))
            metrics['R2'].append(r2_score(true, pred))
            non_zero_mask = true != 0
            if np.any(non_zero_mask):
                metrics['MAPE'].append(mean_absolute_percentage_error(true[non_zero_mask], pred[non_zero_mask]))
            else:
                metrics['MAPE'].append(0.0)
            metrics['MAE'].append(mean_absolute_error(true, pred))

        for metric_name, values in metrics.items():
            df = pd.DataFrame([values], columns=y_tags, index=[f't+{(t + 1)}'])
            path = os.path.join(config['output']['results_dir'], prefix, f'{metric_name}_timestep_{set_name}.csv')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if t == 0:
                df.to_csv(path)
            else:
                df.to_csv(path, mode='a', header=False)

def one_shot_forecast(model, encoder_input_initial, decoder_inputs_future, device):
    """
    執行一次性的編碼-解碼預測。
    適用於標準 Transformer 架構推論。
    """
    model.eval()
    encoder_input = encoder_input_initial.to(device)
    decoder_inputs = decoder_inputs_future.to(device)
    with torch.no_grad():
        _, encoder_hiddens = model.encoder(encoder_input)
        predictions = model.decoder(decoder_inputs, encoder_hiddens)
    return predictions
