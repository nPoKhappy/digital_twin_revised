# Raw-split three-seed ANN-reference KCI

Validation-selected model settings: {'no_gain': 'no_gain', 'gain': 'gain25', 'ss_only': 'ss2p5_only', 'gain_ss': 'gain250_ss250', 'difficult_gradnorm': 'dg_ratio1p5'}. All 15 dynamic models were evaluated on the raw-split experiment; no previous model predictions were reused. Each point is the mean of seed 42, 43 and 44, and error bars show one sample standard deviation (ddof=1), not a bootstrap confidence interval. The seeds share one ANN and the same test operating points. Every valid operating point contributes 2 perturbation directions × 18 prediction steps; ANN |gain| below 1e-5 is excluded, and zero dynamic gain is incorrect. Empty groups are missing values, not zero KCI.

The magnitude quartiles are calculated separately per region and gain pair from normalized ANN |gain| using shared boundaries. See the quartile-edges CSV for bounds and probe counts. KCI measures agreement with ANN directions, not independent proof of Aspen physical directions. Test probes were previously used and are not a fresh blind holdout.

Rebuild after the raw-split selected test evaluation with `python -m scripts.analysis.plot_rawsplit_teacher_gain_kci`.
