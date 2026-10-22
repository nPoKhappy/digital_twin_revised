# Seed42 五模型比較

同時呈現 No-Gain、Gain、Gain＋SS=0.01、Difficult-GradNorm ratio=0.015，以及 Difficult-GradNorm ratio=0.01。原本的四模型圖保留在原資料夾。本資料夾沒有重新訓練或更新模型。

| 模型 | 圖中樣式 |
| --- | --- |
| No-Gain | 深灰方形、虛線 |
| Gain | 藍色圓形、實線 |
| Gain＋SS=0.01 | 橘色三角形、點劃線 |
| Difficult-GradNorm ratio=0.015 | 桃紅色菱形、點線 |
| Difficult-GradNorm ratio=0.01 | 紫色粗十字、虛線 |

兩個 Difficult-GradNorm 版本皆為 seed42、無 SS loss。它們的 ratio 是梯度目標比例；「Gain＋SS=0.01」的 0.01 是 SS loss 權重，兩者代表不同參數。

## 圖表

每張圖均提供 PNG 與向量 PDF。External 圖較寬，放大 PNG 或開啟 PDF 可閱讀各案例。

| 內容 | PNG | PDF |
| --- | --- | --- |
| Internal test：H₂S／SO₂，H1／H18／TS | [查看](five_model_raw_rmse_internal_test.png) | [PDF](five_model_raw_rmse_internal_test.pdf) |
| External R5-19～26：H₂S／SO₂，H1／H18／TS | [查看](five_model_raw_rmse_external_all_data.png) | [PDF](five_model_raw_rmse_external_all_data.pdf) |
| In-range／Soft-OOD 四個 gain pairs 的 all18 KCI | [查看](five_model_all18_kci.png) | [PDF](five_model_all18_kci.pdf) |
| 依 teacher gain 大小分組的 KCI | [查看](conditional_kci_by_gain_magnitude.png) | [PDF](conditional_kci_by_gain_magnitude.pdf) |
| 依 teacher gain 正負號分組的 KCI | [查看](conditional_kci_by_teacher_sign.png) | [PDF](conditional_kci_by_teacher_sign.pdf) |
| Teacher gain 大小與符號分布 | [查看](teacher_gain_landscape.png) | [PDF](teacher_gain_landscape.pdf) |

RMSE 圖使用原始數值，保留每個案例，沒有跨案例平均。Internal 只取 R5、R5-1、R5-2、R5-6 的 test split；external 取 R5-19～26 的 all-data。各 RMSE 面板的縱軸範圍不同，請在同一面板內比較方法。

KCI 是有效 ANN teacher gain 比較的嚴格同號比例，動態 gain 為零計為不一致。條件式 KCI 從五個模型的完整探測紀錄重新分組，沿用相同 teacher、有效門檻及四分位界線；air2→SO₂ 的 teacher 負組沒有有效資料，顯示 N/A。這些線圖沒有 error bar，也不表示跨訓練種子的信賴區間。Teacher 分布圖是共用參考資料，沒有模型比較線。

## 數值與驗證

- [five_model_rmse_values.csv](five_model_rmse_values.csv)：360 筆 RMSE／R²（5 模型 × 12 案例 × 2 品質 × 3 評估）。
- [five_model_all18_kci_values.csv](five_model_all18_kci_values.csv)：40 筆（5 模型 × 2 分布 × 4 配對），含有效比較數、coverage、TP／TN／FP／FN。
- [conditional_kci_by_gain_magnitude.csv](conditional_kci_by_gain_magnitude.csv)：160 筆大小分組數值。
- [conditional_kci_by_teacher_sign.csv](conditional_kci_by_teacher_sign.csv)：80 筆符號分組數值，空組 KCI 為空值。
- [teacher_gain_summary.csv](teacher_gain_summary.csv)、[gain_magnitude_quartile_edges.csv](gain_magnitude_quartile_edges.csv)：共用 teacher 分布及分組界線。
- [figure_validation.json](figure_validation.json)：逐案例指標、KCI 分組總數、來源雜湊與圖表檢查。
- [figure_provenance.json](figure_provenance.json)：五模型實際來源路徑及繪圖程式。

已將 360 筆 RMSE／R² 分別核對到各模型的案例分表；其中原本四模型的 288 筆與舊圖 CSV 一致。40 筆 KCI 與逐探測點聚合結果一致，正負分組及大小分組的有效數、正確數均能還原官方摘要。既有圖表與讀取的來源檔案均以 SHA-256 確認未修改。

繪圖程式：`scripts/plotting/plot_five_models_gradnorm_ratios_seed42.py`。完整預測評估報告見 [evaluation_report.md](../../pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss/evaluation_report.md)。
