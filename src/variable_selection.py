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
        list: [de_mv, y_sv, con_tag, en_mv_and_sv]
            - de_mv: Decoder 輸入變量列表
            - y_sv: 預測目標變量列表  
            - con_tag: 控制標籤列表
            - en_mv_and_sv: Encoder 輸入變量列表
    """
    if total_variables == 8:
        # ======================================================================
        # 配置 8: Claus 過程 (移除 SP，使用 PV 作為未來輸入)
        # ======================================================================
        
        # Encoder Input: 8 variables (3 MVs + 3 SVs + 2 QVs)
        en_mv_and_sv = [
            'acidgas_Fm',          # MV3
            'acidgas_T',           # MV4
            'acidgas_P',           # MV5
            'HEATER1_output_T_PV', # SV1
            'HEATER2_output_T_PV', # SV2 (Will be used as De Input)
            'second_air2',         # SV3 (Will be used as De Input)
            'B35_H2S',             # QV1
            'B35_SO2'              # QV2
        ]
        
        # Decoder Input: future knowns (MVs + selected PVs)
        de_mv = [
            'acidgas_Fm', 
            'acidgas_T', 
            'acidgas_P',
            'HEATER2_output_T_PV', # Using PV as predictor
            'second_air2'          # Using PV as predictor
        ]
        
        # Prediction Targets: Remaining SVs and QVs
        y_sv = [
            'HEATER1_output_T_PV', 
            'B35_H2S', 
            'B35_SO2'
        ]
        
        con_tag = y_sv
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
    
    elif total_variables == 10:
        # ======================================================================
        # 配置 10: Claus 過程硫回收單元配置（10個變量 - Updated with CSV Headers）
        # MAP:
        # T2_SP   -> HEATER2_output_T_SP
        # T1.PV   -> HEATER1_output_T_PV
        # T2.PV   -> HEATER2_output_T_PV
        # AIR2.PV -> second_air2
        # ======================================================================
        
        # Encoder 輸入：包含所有變量 [MV (5) + SV/QV (5)]
        # 順序: [MV..., SV...]
        en_mv_and_sv = [
            'air2_SP',             # MV1
            'HEATER2_output_T_SP', # MV2 (T2_SP)
            'acidgas_Fm',          # MV3
            'acidgas_T',           # MV4
            'acidgas_P',           # MV5
            'HEATER1_output_T_PV', # SV1 (T1.PV)
            'HEATER2_output_T_PV', # SV2 (T2.PV)
            'second_air2',         # SV3 (AIR2.PV)
            'B35_H2S',             # QV1
            'B35_SO2'              # QV2
        ]
        
        # Decoder 輸入：只包含操縱變量 (MV)
        # Decoder 輸入：只包含操縱變量 (MV)，Keras GRU 模型只用了 SPs
        de_mv = [
            'air2_SP', 
            'HEATER2_output_T_SP'
            # 'acidgas_Fm', # Removed to match Keras GRU
            # 'acidgas_T', 
            # 'acidgas_P'
        ]
        
        # 模型預測目標：Keras GRU 模型只預測 H2S 和 SO2
        y_sv = [
            # 'HEATER1_output_T_PV', # Removed to match Keras GRU
            # 'HEATER2_output_T_PV', 
            # 'second_air2', 
            'B35_H2S', 
            'B35_SO2'
        ]
        
        # 控制標籤：與預測目標相同
        con_tag = y_sv
        
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
        
    elif total_variables == 17:
        # ======================================================================
        # 配置 17: Claus Plant Data (17 變量)
        # ======================================================================
        
        # Decoder 輸入：未來已知的操作變數 (MVs)
        de_mv = [
            'acidgas_Fm', 
            'acidgas_T', 
            'acidgas_P',
            'second_air2',
            'air'
        ]
        
        # 預測目標：其餘所有狀態變數 (SVs) 和質量變數 (QVs)
        y_sv = [
            'cat1_input_temp',
            'cat2_input_temp',
            'B35_H2S',
            'B35_SO2',
            'burner_output_T_PV',
            'fur_temp',
            'cat1_output_temp',
            'cat1_deltaP',
            'cat1_bed_temp',
            'cat2_output_temp',
            'cat2_deltaP',
            'cat2_bed_temp'
        ]
        
        # Encoder 輸入：包含所有 17 個變數
        # 順序建議：先放 de_mv 變數，再放其他，或者依照物理意義排序
        # 這裡簡單起見將 de_mv 和 y_sv 串接，或是依照數據中的順序
        # 為了保證對應正確，這裡明確列出所有
        en_mv_and_sv = de_mv + y_sv
        
        con_tag = y_sv

    elif total_variables == 71:
        # ======================================================================
        # 配置 71: Claus 過程全變量配置 (from simulation data)
        # ======================================================================
        de_mv = [
            'acidgas_Fm', 'acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S', 'acidgas_T', 'acidgas_P', 
            'air', 'air_SP', 'second_air2', 'air2_SP', 'COG', 'COG_SP'
        ]
        y_sv = [
            'burner_input_T_SP', 'burner_input_T_PV', 'burner_inputP', 
            'burner_output_T_SP', 'burner_output_T_PV', 'burner_output_P_SP', 'burner_output_P_PV', 
            'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 'fur_outputT', 'fur_outputP_SP', 'fur_outputP_PV', 
            'WHB_F', 'WHB_inputT', 'WHB_inputP', 'WHB_outputT', 'WHB_outputP', 
            'SEP1_F', 'SEP1_P_SP', 'SEP1_P_PV', 'SEP1_T', 
            'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 'HEATER1_output_T_SP', 'HEATER1_output_T_PV', 'HEATER1_output_P', 
            'cat1_F', 'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_output_P_SP', 'cat1_output_P_PV', 'cat1_deltaP', 
            'SEP2_F', 'SEP2_P_SP', 'SEP2_P_PV', 'SEP2_T', 
            'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 'HEATER2_output_T_SP', 'HEATER2_output_T_PV', 'HEATER2_output_P', 
            'cat2_F', 'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_output_P_SP', 'cat2_output_P_PV', 'cat2_deltaP', 
            'SEP3_F', 'SEP3_P_SP', 'SEP3_P_PV', 'SEP3_T', 
            'B35_H2S', 'B35_SO2'
        ]
        con_tag = y_sv
        en_mv_and_sv = de_mv + y_sv


    elif total_variables == 57:
        # ======================================================================
        # 配置 57: Claus 過程 (移除 SP，使用 PV 作為未來輸入)
        # ======================================================================
        de_mv = [
            # 原 de_mv (移除 SP)
            'acidgas_Fm', 'acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S', 'acidgas_T', 'acidgas_P', 
            'air', 'second_air2', 'COG',
            # 從 y_sv 移來的 PV (原 SP 對應)
            'burner_input_T_PV', 'burner_output_T_PV', 'burner_output_P_PV', 
            'fur_outputP_PV', 'SEP1_P_PV', 'HEATER1_output_T_PV', 
            'cat1_output_P_PV', 'SEP2_P_PV', 'HEATER2_output_T_PV', 
            'cat2_output_P_PV', 'SEP3_P_PV'
        ]
        y_sv = [
            'burner_inputP', 
            'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 'fur_outputT', 
            'WHB_F', 'WHB_inputT', 'WHB_inputP', 'WHB_outputT', 'WHB_outputP', 
            'SEP1_F', 'SEP1_T', 
            'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 'HEATER1_output_P', 
            'cat1_F', 'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_deltaP', 
            'SEP2_F', 'SEP2_T', 
            'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 'HEATER2_output_P', 
            'cat2_F', 'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_deltaP', 
            'SEP3_F', 'SEP3_T', 
            'B35_H2S', 'B35_SO2'
        ]
        con_tag = y_sv
        en_mv_and_sv = de_mv + y_sv

    elif total_variables == 72:
        # ======================================================================
        # 配置 58: Claus 過程 (71 變量資料，SP 作為 Decoder 輸入，PV 作為預測目標)
        # 邏輯：SP = 未來已知 → de_mv；PV / 流量響應 = 預測目標 → y_sv
        # air, second_air2, COG 為響應量，移至 y_sv
        # de_mv: 9 (acidgas + air_SP/air2_SP/COG_SP) + 11 SP = 20 vars
        # y_sv : 48 (原 PV) + 3 (air/second_air2/COG) = 51 vars
        # en_mv_and_sv: 20 + 51 = 71 vars (同 71-var 資料檔)
        # 使用方式：YAML 設定 variables_num: 58
        # ======================================================================

        # Decoder 輸入：acidgas MV + SP 設定值 (純未來可知量，不含流量響應)
        de_mv = [
            # ── acidgas 操作條件 (9 vars) ────────────────────────────
            'acidgas_Fm', 'acidgas_CO2', 'acidgas_H2O', 'acidgas_H2S',
            'acidgas_T', 'acidgas_P',
            'air_SP', 'air2_SP', 'COG_SP',
            # ── 從 y_sv 移來的 11 個 SP ──────────────────────────────
            'burner_input_T_SP', 'burner_output_T_SP', 'burner_output_P_SP',
            'fur_outputP_SP', 'SEP1_P_SP',
            'HEATER1_output_T_SP', 'cat1_output_P_SP',
            'SEP2_P_SP', 'HEATER2_output_T_SP', 'cat2_output_P_SP', 'SEP3_P_SP',
        ]

        # 預測目標：PV + 空氣/COG 實際流量響應 (51 vars)
        y_sv = [
            # ── 移入的流量響應 (3 vars) ──────────────────────────────
            'air', 'second_air2', 'COG',
            # ── 原有 PV 及狀態變量 (48 vars) ─────────────────────────
            'burner_input_T_PV', 'burner_inputP',
            'burner_output_T_PV', 'burner_output_P_PV',
            'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 'fur_outputT', 'fur_outputP_PV',
            'WHB_F', 'WHB_inputT', 'WHB_inputP', 'WHB_outputT', 'WHB_outputP',
            'SEP1_F', 'SEP1_P_PV', 'SEP1_T',
            'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 'HEATER1_output_T_PV', 'HEATER1_output_P',
            'cat1_F', 'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_output_P_PV', 'cat1_deltaP',
            'SEP2_F', 'SEP2_P_PV', 'SEP2_T',
            'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 'HEATER2_output_T_PV', 'HEATER2_output_P',
            'cat2_F', 'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_output_P_PV', 'cat2_deltaP',
            'SEP3_F', 'SEP3_P_PV', 'SEP3_T',
            'B35_H2S', 'B35_SO2',
        ]
        con_tag = y_sv
        en_mv_and_sv = de_mv + y_sv  # 20 + 51 = 71 個變量，使用與 71-var 相同的資料檔

    elif total_variables == 54:
        # ======================================================================
        # 配置 54: Claus 過程 (57 變量基礎上移除酸氣組成變量)
        # ======================================================================
        de_mv = [
            # 57 變量配置移除 acidgas_CO2/H2O/H2S
            'acidgas_Fm', 'acidgas_T', 'acidgas_P', 
            'air', 'second_air2', 'COG',
            # 同樣使用 PV 作為未來已知量
            'burner_input_T_PV', 'burner_output_T_PV', 'burner_output_P_PV', 
            'fur_outputP_PV', 'SEP1_P_PV', 'HEATER1_output_T_PV', 
            'cat1_output_P_PV', 'SEP2_P_PV', 'HEATER2_output_T_PV', 
            'cat2_output_P_PV', 'SEP3_P_PV'
        ]
        y_sv = [
            'burner_inputP', 
            'fur_F', 'fur_inputT', 'fur_inputP', 'fur_temp', 'fur_outputT', 
            'WHB_F', 'WHB_inputT', 'WHB_inputP', 'WHB_outputT', 'WHB_outputP', 
            'SEP1_F', 'SEP1_T', 
            'HEATER1_F', 'HEATER1_input_T', 'HEATER1_input_P', 'HEATER1_output_P', 
            'cat1_F', 'cat1_input_temp', 'cat1_output_temp', 'cat1_input_P', 'cat1_deltaP', 
            'SEP2_F', 'SEP2_T', 
            'HEATER2_F', 'HEATER2_input_T', 'HEATER2_input_P', 'HEATER2_output_P', 
            'cat2_F', 'cat2_input_temp', 'cat2_output_temp', 'cat2_input_P', 'cat2_deltaP', 
            'SEP3_F', 'SEP3_T', 
            'B35_H2S', 'B35_SO2'
        ]
        con_tag = y_sv
        en_mv_and_sv = de_mv + y_sv

    else:
        # ======================================================================
        # 錯誤處理：不支持的變量數量
        # ======================================================================
        print(f'錯誤：不支持的變量總數 {total_variables}')
        print('支持的配置: 8, 9, 10, 17, 27, 28, 30, 33, 35, 54, 57, 58, 71')
        raise ValueError(f'不支持的變量總數: {total_variables}')
        
    return de_mv, y_sv, con_tag, en_mv_and_sv
