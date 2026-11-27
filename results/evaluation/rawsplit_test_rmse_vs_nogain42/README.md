# Rawsplit test RMSE compared with No-Gain seed 42

All five validation-selected families and seeds 42, 43, 44 are included.
The baseline is the corresponding No-Gain seed 42 RMSE for the same partition, case, target and evaluation.
`delta_percent = 100 * (RMSE / baseline_rmse - 1)`; negative means smaller RMSE.
The all-horizons-and-TS per-seed CSV has 6840 rows: 6480 for H1-H18 and 360 for TS.
Its three-seed summary has 2280 groups. The separate H1/H18/TS files have 1080 rows and 360 groups.
Summary values are arithmetic means and sample standard deviations across seeds (ddof=1).
No-Gain seed 43 and 44 are also compared against seed 42 and need not equal zero.
RMSE values are in original output units (mole fraction, kmol/kmol).
Internal test contains R5, R5-1, R5-2, R5-6; external contains R5-19 through R5-26.
H1-H18 are horizon-specific; TS is the corresponding time-series evaluation.
These test cases were previously used and are not a new blind holdout.
Rebuild with `python -m scripts.analysis.report_rawsplit_test_rmse_vs_nogain42`.
