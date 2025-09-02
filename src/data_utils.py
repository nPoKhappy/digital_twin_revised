# data_utils.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(file_path, datetime_tag='DateTime', index_tag='DateTime', slice_interval=1):
    """載入 CSV 檔案，解析日期並設置索引。"""
    data = pd.read_csv(file_path)
    data[datetime_tag] = pd.to_datetime(data[datetime_tag], format='%Y/%m/%d %H:%M')
    data.set_index(index_tag, inplace=True)
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

def calculate_zscore_stats(df):
    """計算 Z-score 所需的均值和標準差。"""
    return df.mean(), df.std()

def apply_zscore(df, mean, std):
    """應用 Z-score 標準化。"""
    # 加上一個極小值避免除以零
    return (df - mean) / (std + 1e-8)

def inverse_zscore(df_z, mean, std):
    """還原 Z-score 標準化。"""
    return df_z * std + mean

# 這裡可以繼續添加其他預處理函數，如 outlier_remove 等

def variable_selection(total_variables):
    if total_variables == 8:
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
        de_mv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131',
                'FC511', 'HV503A', 'SC020']
        y_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131']
        con_tag = ['qv', 'sv1', 'sv2', 'sv3', 'sv4', 'sv5']
        en_mv_and_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131',
                        'FC511', 'HV503A', 'SC020']
    elif total_variables == 27:
        de_mv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                'PI142', 'TI082', 'PI506', 'FI501', 'FC511', 'HV503A', 'SC020']
        y_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                'PI142', 'TI082', 'PI506', 'FI501',]
        con_tag = ['qv', 'sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'sv7', 'sv8', 'sv9', 'sv10', 'sv11',
                    'sv12', 'sv13', 'sv14', 'sv15', 'sv16', 'sv17', 'sv18', 'sv19', 'sv20', 'sv21', 'sv22', 'sv23']
        en_mv_and_sv = ['AI503', 'FI547', 'FI015CA', 'TI508', 'TI502', 'PI131', 'AT501' ,'TI546', 'TI501', 'FT505', 
                        'TI510', 'TI503', 'TI574', 'FT503', 'TI511', 'TI514', 'TI512', 'PI148', 'PI141', 'TI081', 
                        'PI142', 'TI082', 'PI506', 'FI501', 'FC511', 'HV503A', 'SC020']
    elif total_variables == 28:
        de_mv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV','MV1_FIC01', 'MV2_SI01']
        y_sv =  ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV']
        con_tag = [ 'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                    'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                    'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV']
        en_mv_and_sv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                        'SV11_TI05', 'SV12_TI07', 'SV13_FIC02PV', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                        'SV21_TI11', 'SV22_PIC02', 'SV23_FIC01PV', 'SV24_FC03PV','MV1_FIC01', 'MV2_SI01']
    elif total_variables == 30:
        en_mv_and_sv = [
            'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_SI01',
            'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06',
            'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02',
            'SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
            'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08',
            'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03',
            'SV19_TI10', 'SV20_PI05', 'SV21_TI11', 'SV22_PIC02',
            'SV23_FIC01', 'SV24_FIC02'
        ]

        # Decoder 輸入 (de_mv): 只包含未來可控的操縱變數 (MV)
        de_mv = [
            'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_SI01'
        ]

        # 模型預測目標 (y_sv): 包含所有非操縱變數，即 SV 和 QV
        y_sv = [
            'QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06',
            'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02',
            'SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
            'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08',
            'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03',
            'SV19_TI10', 'SV20_PI05', 'SV21_TI11', 'SV22_PIC02',
            'SV23_FIC01', 'SV24_FIC02'
        ]
        
        # con_tag: 通常與 y_sv 相同
        con_tag = y_sv
    elif total_variables == 33:
        de_mv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_SI01']
        y_sv =  ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PIC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV']
        con_tag = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_FIC02', 'SV14_TI08', 'SV15_TIC12', 'SV16_TI09', 'SV17_PI04', 'SV18_PC03', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV']
        en_mv_and_sv = ['QV1_AI01', 'QV2_AI02', 'SV1_FI05', 'SV2_FI06', 'SV3_TI06', 'SV4_TI01', 'SV5_PI06', 'SV6_AT02','SV7_TI02', 'SV8_TI03', 'SV9_FT04', 'SV10_TI04',
                'SV11_TI05', 'SV12_TI07', 'SV13_PIC01', 'SV14_PIC02', 'SV15_PIC03', 'SV16_TI08', 'SV17_TI09', 'SV18_PI04', 'SV19_TI10', 'SV20_PI05',
                'SV21_TI11', 'SV22_FC03SP', 'SV23_FC03PV', 'MV1_AIC01', 'MV2_FIC01', 'MV3_FIC02', 'MV4_TIC12', 'MV5_SI01']
    elif total_variables == 35:
        de_mv = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25',
                'MV1_CC1SP', 'MV2_CC2SP', 'MV3_HV503_Pos',]
        y_sv =  ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25']
        con_tag = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25']
        en_mv_and_sv = ['QV1', 'QV2', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6', 'SV7', 'SV8', 'SV9', 'SV10', 'SV11', 'SV12',
                        'SV13', 'SV14', 'SV15', 'SV16', 'SV17', 'SV18', 'SV19', 'SV20', 'SV21', 'SV22', 'SV23', 'SV24', 'SV25',
                        'MV1_CC1SP', 'MV2_CC2SP', 'MV3_HV503_Pos',]
        
    else:
        print('wrong amount')
    return de_mv, y_sv, con_tag, en_mv_and_sv