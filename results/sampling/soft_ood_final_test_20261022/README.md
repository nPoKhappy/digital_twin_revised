# 新增 Soft-OOD test 操作條件

[操作條件 CSV](../../../data/sampling/LHS_air2_t2_soft_ood_final_test_20261022.csv) 使用 `scripts/sampling/generate_lhs_air2_t2_soft_ood.py` 的隔離執行副本產生。原程式與舊資料保留；執行副本只更換輸入／輸出／圖檔路徑，並設定各區域的 LHS 與擾動抽取種子。原有範圍、樣本配置、擾動及裁切邏輯不變。

## 資料組成

共 500 個 LHS 基準操作點與 4,500 列局部擾動，合計 5,000 列。包含 17 個物理輸入與 6 個追蹤欄位：`case_id`、`base_id`、`case_type`、`perturbed_mv`、`direction`、`delta`。

| 區域 | 基準點 | 輸出列數（含擾動） |
| --- | --- | --- |
| Air2 低側外插，T2 保持內插 | 100 | 1,000 |
| Air2 高側外插，T2 保持內插 | 100 | 1,000 |
| T2 低側外插，Air2 保持內插 | 100 | 1,000 |
| T2 高側外插，Air2 保持內插 | 100 | 1,000 |
| Air2 低側／T2 低側 | 25 | 250 |
| Air2 低側／T2 高側 | 25 | 250 |
| Air2 高側／T2 低側 | 25 | 250 |
| Air2 高側／T2 高側 | 25 | 250 |

因此單變數 OOD 共 4,000 列，同時變動的 joint-OOD 共 1,000 列。單變數區域的另一變數保持在安全內部範圍：Air2 142.5–297.5、T2 142.5–237.5。Air2 低側為 110–140、高側為 300–340；T2 低側為 100–140、高側為 240–280。

| 變數 | 整體範圍（含邊界） | 局部擾動 |
| --- | --- | --- |
| Air2 SP（B33.SPo.SPo） | 110–340 | ±2.5 |
| T2 SP（B20.SPo.SPo） | 100–280 | ±2.5 |
| 酸氣流量（B34.SPo.SPo） | 110.5–160.5 | ±0.5 |
| 酸氣溫度（ACIDGAS.T.T） | 82.6–84.6 | ±0.2 |
| 酸氣壓力（ACIDGAS.P.P） | 1.66–1.69 | ±0.005 |

所有操作條件均在上述五輸入範圍內，且 Air2 或 T2 至少一項位於 core 外側；沿用評估器定義，Air2 ≤140／≥300、T2 ≤140／≥240 含邊界視為 core 外。其他 12 個物理輸入使用原始 dynamic CSV 的中位數設定。

原流程會抽減擾動列以維持每區域目標列數，因此部分基準點的正負擾動組不完整。5,000 列不是 5,000 個獨立基準點；同一 `base_id` 是同一操作點家族，若日後切分資料，應整組保留。

## 驗證及紀錄

- [generation_manifest.json](generation_manifest.json)：每區域兩個隨機種子、原始檔案與輸出 SHA-256、範圍、擾動量、固定值、區域配置及套件版本。
- [generation_validation.json](generation_validation.json)：列數、single／joint-OOD 配置、逐區域 LHS 分層及種子重現、範圍、唯一性、擾動方向／delta、固定值與原始檔案保護檢查。
- [region_range_checks.csv](region_range_checks.csv)：8 區域 × 5 變動輸入的範圍與分層檢查。
- [overlap_checks.csv](overlap_checks.csv)：與既有 sampling CSV、steady-state XLSX 及固定值來源的操作條件重複檢查；五輸入四捨五入至小數點後 8 位後比較，XLSX 包含可解析數值的操作條件，不限完成狀態。
- [generation.log](generation.log)：原生程式執行紀錄。
- [LHS_Air2_T2_Soft_OOD_Check_Plots](LHS_Air2_T2_Soft_OOD_Check_Plots)：17 張基準點分布圖；Air2、T2 已檢視。整體分布因單變數／joint-OOD 配額而在外側密度較高，LHS 分層是在各區域內檢查。

## 接續使用

目前是 **Aspen 輸入操作條件 CSV**，尚未執行物理模擬或 KCI。CSV 不含完整模型變數；需先執行 Aspen 穩態模擬、保存完成狀態並匯出模型所需資料，才能接續 KCI 評估。

若用作最終 test，先固定 ratio／模型選擇，再評估這批新資料。
