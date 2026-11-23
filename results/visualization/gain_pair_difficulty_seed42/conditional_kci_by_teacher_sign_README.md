# Three-seed ANN gain sign KCI

Models: No-Gain and validation-selected settings {'gain': 'gain25', 'ss_only': 'ss2p5_only', 'gain_ss': 'gain25_ss25', 'difficult_gradnorm': 'dg_ratio0p15'}. Points are the mean KCI across training seeds 42, 43, 44; bars are one sample SD (ddof=1), not bootstrap intervals. The three seeds share the ANN and validation/test operating points.

Each valid point has absolute ANN gain at least 1e-5 and contributes 36 comparisons (two step directions by 18 prediction steps). Dynamic zero gain counts as incorrect. Air2 to SO2 has no negative ANN-gain probes in either region; these groups are missing, not zero. The plotted statistics retain all valid points. KCI measures agreement to the ANN reference, not direct physical gain validation. The test probes have been used previously; they are not a fresh blind holdout.

This figure uses the updated validation selection. Some historical validation source files were deleted, so the full historical provenance is unavailable; current test NPZ files and their remaining source hashes have been verified. The folder name includes seed42 but this figure uses three seeds.

Rebuild: python -m scripts.analysis.plot_updated_test_teacher_sign_three_seeds
