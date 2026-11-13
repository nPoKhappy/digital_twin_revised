# Gain+SS validation 結果解讀

五組皆使用相同的 Gain／SS 權重、seed42，SS 限制全部 48 個模型輸出。動態內插為 R5-3／4／5、外插為 R5-27／28／29；沿用 Gain、SS-only 的 No-Gain 基準與逐案例比值，AvgRelRMSE 內外各 50%。

| Gain = SS | 內插 AvgRelRMSE | 外插 AvgRelRMSE | 整體 AvgRelRMSE | ANN-reference KCI (%) |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.064559 | 0.909433 | 0.983944 | 92.474863 |
| 2.5 | 1.221701 | 0.588772 | **0.848117** | 94.707833 |
| 25 | 1.298095 | 0.613893 | 0.892688 | 97.029306 |
| 250 | 1.610619 | 0.615484 | 0.995646 | **98.463036** |
| 2500 | 2.148662 | 0.927503 | 1.411698 | 94.150887 |

2.5 的整體 AvgRelRMSE 最低，相對 No-Gain=1 低約 15.19%。這是幾何聚合指標的改善，不能直接稱為所有原始 RMSE 平均降低 15.19%。其內插分數比基準高約 22.17%，外插則低約 41.12%；改善主要來自外插。五組的內插分數皆高於 1，因此沒有一組在兩個區域都優於 No-Gain。

250 的 KCI 最高，但動態整體分數接近 No-Gain，且不及 2.5。較高的 ANN-reference 方向一致性，在這些 checkpoint 上沒有同時對應較低的動態預測誤差。2500 的整體分數大於 1，並且 KCI 也未超過 250，沒有顯示繼續加大權重的優勢。

## 與 Gain、SS-only 比較

各家族依同一個 validation AvgRelRMSE 選出的最低分設定：

| 家族 | Gain weight | SS weight | 整體 AvgRelRMSE | ANN-reference KCI (%) |
| --- | ---: | ---: | ---: | ---: |
| Gain-only | 250 | 0 | 0.953755 | 98.941025 |
| SS-only | 0 | 2.5 | **0.758654** | 94.784198 |
| Gain+SS | 2.5 | 2.5 | 0.848117 | 94.707833 |

目前 SS-only=2.5 的整體 AvgRelRMSE 仍最低。同為 SS=2.5 時，這次 Gain+SS=2.5 的固定 checkpoint 沒有取得較低的 AvgRelRMSE，KCI 也略低。這是本次 checkpoint 與 validation 資料上的比較；沒有新增其他 seed 或進行顯著性檢定。

[全部 15 組 validation 分數](three_family_validation_comparison.csv)／[跨家族資料與基準核對](cross_family_compatibility.json)。

## 圖表與核對

- [內插／外插雙面板](regional_kci_vs_avgrelrmse.png)
- [整體 KCI × AvgRelRMSE](overall_kci_vs_avgrelrmse.png)
- [H1／H18／TS 六面板](horizon_kci_vs_avgrelrmse.png)
- [原單位 RMSE，含 No-Gain](validation_rmse_summary.png)
- [完整計算定義與重跑指令](README.md)

穩態 KCI 使用 `lhs_valid_in_range.xlsx`、`lhs_valid_soft_ood.xlsx` 的 4,404／2,585 個探測點重新評估；5,032,080 筆方向／時刻比較已與保存的增益陣列及逐點 CSV 核對。動態 1,368 項 RMSE 由預測陣列獨立重算。原有 Gain、SS-only 的 214 個產物及本次使用的 60 個受保護輸入均保持不變。

以上是單一 seed42 的 validation 分析，並非 test 結果。KCI 以 ANN 為方向參考，不代表物理增益正確性或閉迴路控制效能。
