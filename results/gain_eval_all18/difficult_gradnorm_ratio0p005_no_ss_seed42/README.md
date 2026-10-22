# Difficult-GradNorm ratio=0.005，seed42，無 SS

預測及三個 ratio 比較：[evaluation_report.md](../../pgin_gain005_ratio0p005_seed42_difficult_group_gradnorm_no_ss/evaluation_report.md)。

- `gain_evaluation_detail.csv`：兩分布 × 四 gain pairs，共 8 筆，含 KCI、coverage 及 confusion counts。
- `gain_evaluation_summary.csv`：兩分布的整體 micro KCI。
- `gain_probe_diagnostics.csv`：1,019,088 筆完整逐探測點、方向、pair、post-step 結果。
- `three_ratio_pair_comparison.csv`、`three_ratio_overall_comparison.csv`：0.005／0.01／0.015 的配對及整體比較。
- `three_ratio_kci_by_teacher_sign.csv`：三個 ratio 的正負 teacher 組，各組有有效分母與正確數。
- `kci_change_vs_existing_ratios.csv`：0.005 相較 0.01／0.015 的 KCI 百分點變化。
- `evaluation_validation.json`：全量 KCI 重新聚合的核對結果。

使用 `data/Claus_steady_state/lhs_test_in_range.xlsx`（過濾後 4,385 點）及 `lhs_test_soft_ood_3.xlsx`（2,692 點），與其他 ratio 完全相同。有效 teacher 門檻為絕對值 1e-5，零動態 gain 計為不一致，採 all18 嚴格同號。

只執行一個新模型，原生 `gain_vs_no_gain_comparison.csv` 為空；六模型比較另外整合既有官方摘要，見圖表資料夾。KCI 為 ANN-reference 方向一致性。

```powershell
python scripts/evaluation/evaluate_gain_in_out_distribution.py --in-range data/Claus_steady_state/lhs_test_in_range.xlsx --soft-ood data/Claus_steady_state/lhs_test_soft_ood_3.xlsx --configs configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p005_no_ss_seed42.yaml --max-points 0 --batch-size 64 --device cuda --output-dir results/gain_eval_all18/difficult_gradnorm_ratio0p005_no_ss_seed42
```
