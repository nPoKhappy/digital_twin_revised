# Difficult-GradNorm ratio=0.01，seed42，無 SS

動態預測及比較報告：[evaluation_report.md](../../pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss/evaluation_report.md)。

- 配置：`configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p01_no_ss_seed42.yaml`。
- 權重：`saved_models/pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss.pth`。
- In-range：`data/Claus_steady_state/lhs_test_in_range.xlsx`，過濾後 4,385 點。
- Soft-OOD：`data/Claus_steady_state/lhs_test_soft_ood_3.xlsx`，2,692 點。
- `gain_evaluation_detail.csv`：兩分布 × 四配對，8 筆。
- `gain_evaluation_summary.csv`：兩分布的 micro KCI、coverage 及零 gain 數。
- `gain_probe_diagnostics.csv`：1,019,088 筆逐探測點、擾動方向、pair、post-step 結果。
- `comparison_vs_ratio0p015.csv`、`overall_comparison_vs_ratio0p015.csv`：相較原版的 KCI 百分點變化。
- `kci_by_teacher_sign.csv`：正負 teacher 組的有效數與 KCI；空組為 N/A。

只有一個模型，原生 `gain_vs_no_gain_comparison.csv` 為空；此檔不代表已執行 No-Gain 比較。KCI 是 ANN-reference 方向一致性，與動態 RMSE 使用不同的評估資料。

重跑命令（於專案根目錄執行）：

```powershell
python scripts/evaluation/evaluate_gain_in_out_distribution.py --in-range data/Claus_steady_state/lhs_test_in_range.xlsx --soft-ood data/Claus_steady_state/lhs_test_soft_ood_3.xlsx --configs configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p01_no_ss_seed42.yaml --max-points 0 --batch-size 64 --device cuda --output-dir results/gain_eval_all18/difficult_gradnorm_ratio0p01_no_ss_seed42
```
