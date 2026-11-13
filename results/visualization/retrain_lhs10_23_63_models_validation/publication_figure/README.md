# 63 models: regional KCI–AvgRelRMSE figure

`regional_63_models_4x2.png` / `.pdf` is the complete eight-panel figure: four model families by two validation regions. Each family panel contains 15 points (five parameter values × three trained seeds) and three grey No-Gain references. These reference points repeat across the family panels; they are not additional training runs. Colour indicates the loss weight for Gain, SS-only and Gain+SS, and ρ for Difficult-GradNorm. Marker shape indicates seed. Gain+SS uses equal Gain and SS weights. These ρ values are target gradient-norm ratios, not loss weights.

All panels in every output, including the split figures, use identical x- and y-axis limits for direct comparisons. The `regional_63_models_4x2_common_y` files are retained under their original names and now have the same scale as `regional_63_models_4x2`. `regional_63_models_part1` and `regional_63_models_part2` split the eight panels over two figures with larger printing size. The figure labels the x-axis `KCI(%)`; explanatory text remains in this README rather than beneath the figure.

All raw positions are preserved without jitter. Some markers may overlap at the final display size; consult `../regional_coordinates_63_models.csv` or the interactive report for the complete 126 coordinates. There are 63 independently trained dynamic models, evaluated in two regions; all share one ANN teacher. The figure shows no mean, uncertainty estimate, or statistical test.

**X-axis:** ANN-reference KCI computed over in-range or soft-OOD steady-state probes, respectively. KCI measures agreement with the ANN's gain direction; it is not a direct measurement of Aspen's true gain or closed-loop performance.

**Y-axis:** for each region, geometric mean of 18 same-case ratios (three cases × two targets × H1/H18/TS) between a model's RMSE and the corresponding RMSE of the same seed's No-Gain model. The dashed line is the No-Gain reference at 1. The inner cases are R5-3, R5-4 and R5-5; the outer cases are R5-27, R5-28 and R5-29. R5-5 crosses the defined core boundary (approximately 33.19% of its raw time points are outside), so “interpolation” denotes the designated case group, not a guarantee that every point lies in the core.

The two axes use different validation datasets. Reproduce from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m scripts.plotting.plot_retrain_63_publication --overwrite
```

`figure_provenance.json` records source and figure SHA-256 hashes. The source CSV is never changed by the plotting script.
