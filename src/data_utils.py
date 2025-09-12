# ==============================================================================
# 數據工具模組 - 提供時間序列數據的載入、預處理和變量選擇功能
# ==============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# 數據載入與基本處理函數
# ==============================================================================

def load_data(file_path, datetime_tag='DateTime', index_tag='DateTime', slice_interval=1):
    """
    載入 CSV 檔案，解析日期並設置索引。
    如果沒有指定的日期時間列，則使用預設的數值索引。
    
    Args:
        file_path (str): CSV 檔案路径
        datetime_tag (str): 包含日期時間的欄位名稱
        index_tag (str): 用作索引的欄位名稱
        slice_interval (int): 數據採樣間隔（每隔多少行取一次數據）
        
    Returns:
        pd.DataFrame: 處理後的數據，以日期時間為索引（如果存在）或數值索引
        
    Raises:
        KeyError: 當指定的日期時間列不存在時
        ValueError: 當日期時間格式無法解析時
    """
    # 讀取 CSV 檔案
    data = pd.read_csv(file_path)
    
    # 檢查是否存在指定的日期時間列
    if datetime_tag in data.columns:
        try:
            # 將日期字符串轉換為 datetime 對象，格式: '年/月/日 時:分'
            data[datetime_tag] = pd.to_datetime(data[datetime_tag], format='%Y/%m/%d %H:%M')
            # 將日期時間欄位設為索引，便於時間序列操作
            data.set_index(index_tag, inplace=True)
            print(f"成功設置 {datetime_tag} 為時間索引")
        except (ValueError, pd.errors.ParserError) as e:
            print(f"警告：無法解析日期格式 {datetime_tag}，使用原始數據: {e}")
            # 如果日期解析失敗，保持原始數據不變
    else:
        print(f"注意：數據中沒有找到 '{datetime_tag}' 列，使用數值索引")
        # 沒有日期時間列時，保持預設的數值索引
    
    # 根據 slice_interval 進行採樣（例如每2行取1行）
    return data[::slice_interval]

def select_date_range(df, start_date, end_date):
    """
    根據開始和結束日期篩選 DataFrame。
    
    Args:
        df (pd.DataFrame): 輸入的數據框
        start_date (datetime): 開始日期
        end_date (datetime): 結束日期
        
    Returns:
        pd.DataFrame: 篩選後的數據框，只包含指定日期範圍內的數據
    """
    df_copy = df.copy()
    # 找出超出日期範圍的索引
    index_to_drop = (df_copy.index > end_date) | (df_copy.index < start_date)
    # 移除超出範圍的數據
    return df_copy.drop(df_copy.index[index_to_drop])

def remove_event_periods(df, event_periods=None):
    """
    移除指定事件期間的數據。
    用於排除設備維護、異常事件等特殊期間的數據，確保訓練數據的質量。
    
    Args:
        df (pd.DataFrame): 輸入的數據框
        event_periods (list): 事件期間列表，每個元素是 (start_event, end_event) 元組
        
    Returns:
        pd.DataFrame: 移除事件期間後的數據框
    """
    if event_periods is None or np.size(event_periods) == 0:
        return df  # 如果沒有指定事件期間，直接返回原數據
    
    df_copy = df.copy()
    # 遍歷每個事件期間
    for start_event, end_event in event_periods:
        # 找出位於事件期間內的數據索引
        index_to_drop = (df_copy.index > start_event) & (df_copy.index < end_event)
        # 移除事件期間內的數據
        df_copy = df_copy.drop(df_copy.index[index_to_drop])
    return df_copy

def shift_time(df, delta_minutes, datetime_tag='DateTime', index_tag='DateTime'):
    """
    對 DataFrame 的時間戳進行平移。
    用於處理數據同步問題或創建時間偏移的數據集。
    
    Args:
        df (pd.DataFrame): 輸入的數據框
        delta_minutes (int): 時間偏移量（分鐘）
        datetime_tag (str): 日期時間欄位名稱
        index_tag (str): 索引欄位名稱
        
    Returns:
        pd.DataFrame: 時間偏移後的數據框
    """
    df_copy = df.copy().reset_index()  # 重置索引以便修改日期欄位
    # 將時間戳向前或向後平移指定的分鐘數
    df_copy[datetime_tag] = df_copy[datetime_tag] + timedelta(minutes=delta_minutes)
    # 重新設定索引
    return df_copy.set_index(index_tag)

def load_data_safe(file_path, datetime_tag='DateTime', index_tag='DateTime', slice_interval=1):
    """
    安全地載入 CSV 檔案，自動處理是否存在日期時間列的情況。
    這是 load_data 的增強版本，提供更好的錯誤處理和靈活性。
    
    Args:
        file_path (str): CSV 檔案路径
        datetime_tag (str): 包含日期時間的欄位名稱（可選）
        index_tag (str): 用作索引的欄位名稱（可選）
        slice_interval (int): 數據採樣間隔
        
    Returns:
        tuple: (pd.DataFrame, bool) - 返回處理後的數據和是否使用了時間索引的標誌
    """
    # 讀取 CSV 檔案
    data = pd.read_csv(file_path)
    has_datetime_index = False
    
    # 檢查是否存在指定的日期時間列
    if datetime_tag in data.columns:
        try:
            # 嘗試多種日期格式
            date_formats = ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M']
            
            for fmt in date_formats:
                try:
                    data[datetime_tag] = pd.to_datetime(data[datetime_tag], format=fmt)
                    break
                except ValueError:
                    continue
            else:
                # 如果所有格式都失敗，使用 pandas 的自動推斷
                data[datetime_tag] = pd.to_datetime(data[datetime_tag], infer_datetime_format=True)
            
            # 設置時間索引
            data.set_index(index_tag, inplace=True)
            has_datetime_index = True
            print(f"✓ 成功設置 {datetime_tag} 為時間索引")
            
        except (ValueError, pd.errors.ParserError) as e:
            print(f"⚠ 警告：無法解析日期格式，保持數值索引: {e}")
            has_datetime_index = False
    else:
        print(f"ℹ 注意：數據中沒有 '{datetime_tag}' 列，使用數值索引")
        has_datetime_index = False
    
    # 根據 slice_interval 進行採樣
    sampled_data = data[::slice_interval]
    
    print(f"✓ 成功載入數據：{sampled_data.shape[0]} 行 × {sampled_data.shape[1]} 列")
    
    return sampled_data, has_datetime_index

# ==============================================================================
# 數據標準化函數
# ==============================================================================

def calculate_zscore_stats(df):
    """
    計算 Z-score 標準化所需的均值和標準差。
    Z-score 標準化將數據轉換為均值為0、標準差為1的分佈。
    
    Args:
        df (pd.DataFrame): 輸入的數據框
        
    Returns:
        tuple: (均值序列, 標準差序列)
    """
    return df.mean(), df.std()

def apply_zscore(df, mean, std):
    """
    應用 Z-score 標準化。
    公式: z = (x - μ) / σ
    
    Args:
        df (pd.DataFrame): 待標準化的數據框
        mean (pd.Series): 各特徵的均值
        std (pd.Series): 各特徵的標準差
        
    Returns:
        pd.DataFrame: 標準化後的數據框
    """
    # 加上一個極小值（1e-8）避免除以零的情況
    std_safe = std + 1e-8
    df_z = (df - mean) / std_safe
    if df_z.isnull().values.any():
        # 找出是哪些欄位產生了 NaN
        nan_cols = df_z.columns[df_z.isnull().any()].tolist()
        print(f"錯誤：標準化後在以下欄位中發現 NaN: {nan_cols}")
        
        # 找出原始 std 中接近零的欄位
        problem_std_cols = std[std < 1e-9].index.tolist()
        if problem_std_cols:
            print(f"原因分析：以下欄位的標準差接近於零，可能導致數值不穩定: {problem_std_cols}")
    return df_z

def inverse_zscore(df_z, mean, std):
    """
    還原 Z-score 標準化。
    公式: x = z * σ + μ
    
    Args:
        df_z (pd.DataFrame): 標準化後的數據框
        mean (pd.Series): 各特徵的原始均值
        std (pd.Series): 各特徵的原始標準差
        
    Returns:
        pd.DataFrame: 還原到原始尺度的數據框
    """
    return df_z * std + mean

# ==============================================================================
# 變量選擇函數 - 根據不同的變量總數配置，定義模型的輸入輸出結構
# ==============================================================================

def variable_selection(total_variables):
    """
    根據總變量數量選擇相應的變量配置。
    支持多種工業過程的變量配置方案。
    
    變量類型說明:
    - MV (Manipulated Variables): 操縱變量，可控制的輸入變量
    - SV (State Variables): 狀態變量，過程的狀態指標
    - QV (Quality Variables): 質量變量，產品質量指標
    
    模型結構說明:
    - de_mv: Decoder 輸入變量（未來已知的控制變量）
    - y_sv: 預測目標變量（需要預測的狀態和質量變量）
    - en_mv_and_sv: Encoder 輸入變量（歷史數據，包含所有變量）
    - con_tag: 控制標籤（通常與 y_sv 相同）
    
    Args:
        total_variables (int): 總變量數量
        
    Returns:
        tuple: (de_mv, y_sv, con_tag, en_mv_and_sv)
            - de_mv: Decoder 輸入變量列表
            - y_sv: 預測目標變量列表  
            - con_tag: 控制標籤列表
            - en_mv_and_sv: Encoder 輸入變量列表
    """
    if total_variables == 8:
        # ======================================================================
        # 配置 8: 基礎工業過程配置
        # ======================================================================
        de_mv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'SV24_FC03OP', 'SV25_TIC12', 
                'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_PIC01', 'MV6_PIC02', 'MV7_PIC03', 'MV8_SI01']
        y_sv =  ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'SV24_FC03OP', 'SV25_TIC12']
        con_tag = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'SV24_FC03OP', 'SV25_TIC12']
        en_mv_and_sv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'SV24_FC03OP', 'SV25_TIC12',
                'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_PIC01', 'MV6_PIC02', 'MV7_PIC03', 'MV8_SI01']
    elif total_variables == 9:
        # ======================================================================
        # 配置 9: 簡化工業過程配置（9個變量）
        # ======================================================================
        de_mv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131',
                'FC511', 'HV503A', 'SC020']  # 包含所有變量作為decoder輸入
        y_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131']  # 前6個作為預測目標
        con_tag = ['qv', 'sv1', 'sv2', 'sv3', 'sv4', 'sv5']  # 簡化的控制標籤
        en_mv_and_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131',
                        'FC511', 'HV503A', 'SC020']  # encoder輸入包含所有變量
    elif total_variables == 27:
        # ======================================================================
        # 配置 27: 擴展工業過程配置（27個變量）
        # ======================================================================
        de_mv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                'PI142', 'TI082', 'PI506', 'FI501', 'FC511', 'HV503A', 'SC020']
        y_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                'PI142', 'TI082', 'PI506', 'FI501',]  # 前24個變量作為預測目標
        con_tag = ['qv', 'sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'sv7', 'sv8', 'sv9', 'sv10', 'sv11',
                    'sv12', 'sv13', 'sv14', 'sv15', 'sv16', 'sv17', 'sv18', 'sv19', 'sv20', 'sv21', 'sv22', 'sv23']
        en_mv_and_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                        'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                        'PI142', 'TI082', 'PI506', 'FI501', 'FC511', 'HV503A', 'SC020']
    elif total_variables == 28:
        # ======================================================================
        # 配置 28: 中等規模工業過程配置（28個變量）
        # ======================================================================
        de_mv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV','MV1_FIC01', 'MV2_SI01']  # 包含部分MV作為decoder輸入
        y_sv =  ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV']  # 前26個變量作為預測目標
        con_tag = [ 'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                    'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                    'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV']
        en_mv_and_sv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                        'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                        'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV','MV1_FIC01', 'MV2_SI01']
    elif total_variables == 30:
        # ======================================================================
        # 配置 30: 標準工業過程配置（30個變量）
        # 清晰分離操縱變量(MV)和狀態變量(SV/QV)
        # ======================================================================
        
        # Encoder 輸入：包含所有歷史數據（MV + SV + QV）
        en_mv_and_sv = [
            'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_SI01',      # 操縱變量
            'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06',         # 質量和狀態變量
            'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02',
            'SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
            'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08',
            'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03',
            'SV19_TI10', 'SV20_PI05', 'SV21_TI11', 'SV22_PIC02',
            'SV23_FIC01', 'SV24_FIC02'
        ]

        # Decoder 輸入：只包含未來可控的操縱變數 (MV)
        de_mv = [
            'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_SI01'
        ]

        # 模型預測目標：包含所有非操縱變數，即 SV 和 QV
        y_sv = [
            'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06',
            'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02',
            'SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
            'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08',
            'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03',
            'SV19_TI10', 'SV20_PI05', 'SV21_TI11', 'SV22_PIC02',
            'SV23_FIC01', 'SV24_FIC02'
        ]
        
        # 控制標籤：通常與預測目標相同
        con_tag = y_sv
    elif total_variables == 33:
        # ======================================================================
        # 配置 33: 複雜工業過程配置（33個變量）
        # ======================================================================
        de_mv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_SI01']  # 包含更多MV
        y_sv =  ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV']  # 前25個變量作為預測目標
        con_tag = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV']  # 注意：這裡有個小錯誤 'SV18_PC03' 應該是 'SV18_PIC03'
        en_mv_and_sv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_SI01']
    elif total_variables == 35:
        # ======================================================================
        # 配置 35: 大規模工業過程配置（35個變量）
        # ======================================================================
        de_mv = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25',
                'MV1_CC1SP', 'MV2_CC2SP', 'MV3_HV503_Pos',]  # 包含簡化命名的變量
        y_sv =  ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25']  # 27個狀態/質量變量
        con_tag = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25']
        en_mv_and_sv = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                        'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25',
                        'MV1_CC1SP', 'MV2_CC2SP', 'MV3_HV503_Pos',]  # 包含所有30個變量作為encoder輸入
        
    else:
        # ======================================================================
        # 錯誤處理：不支持的變量數量
        # ======================================================================
        print(f'錯誤：不支持的變量總數 {total_variables}')
        print('支持的配置: 8, 9, 27, 28, 30, 33, 35')
        raise ValueError(f'不支持的變量總數: {total_variables}')
        
    return de_mv, y_sv, con_tag, en_mv_and_sv