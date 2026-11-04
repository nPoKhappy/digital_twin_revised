# JTICE figures: panel map and source

All panels use source CSV values without averaging or retraining. The labels below identify **cases**, and both H2S and SO2 axes in the same row refer to the same case. Each line shows H1, H18 and TS, and each panel has its own y-axis scale.

- `rmse_internal_validation`: (a) R5-3; (b) R5-4; (c) R5-5.
- `rmse_external_validation`: (a) R5-13; (b) R5-14.
- `rmse_internal_test`: (a) R5; (b) R5-1; (c) R5-2; (d) R5-6.
- `rmse_external_test_a_d`: (a) R5-19; (b) R5-20; (c) R5-21; (d) R5-22.
- `rmse_external_test_e_h`: (e) R5-23; (f) R5-24; (g) R5-25; (h) R5-26.
- `ann_parity`: (a) H2S; (b) SO2.
- `nearseries`: (a) R5-19 low Air2; (b) R5-20 high Air2; (c) R5-21 low T2; (d) R5-22 high T2; (e) R5-23 low Air2/high T2; (f) R5-24 high Air2/high T2; (g) R5-25 low Air2/low T2; (h) R5-26 high Air2/low T2.

`validation.json` records exact plot-coordinate checks and SHA-256 source hashes. Original input files are read only. DG means Difficult-GradNorm; rho is the target gradient ratio.
