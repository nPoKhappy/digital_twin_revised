# ????

?????? YAML ????????????????????????????????????? YAML ???`exp_name` ? seed ????

## ?????

| ?? | seed | ?? | exp_name | ?? | ????? |
| --- | --- | --- | --- | --- | --- |
| `difficult_gradnorm` | 42 | [YAML](../configs/pgin/difficult_gradnorm/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss/zscore_std.csv) |
| `gain` | 42 | [YAML](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42/zscore_std.csv) |
| `gain_ss` | 42 | [YAML](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss0001_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss0001_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss0001_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss0001_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss0001_seed42/zscore_std.csv) |
| `gain_ss` | 42 | [YAML](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss001_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss001_seed42/zscore_std.csv) |
| `gain_ss` | 42 | [YAML](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss005_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42/zscore_std.csv) |
| `no_gain` | 42 | [YAML](../configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42/zscore_std.csv) |
| `ss_only` | 42 | [YAML](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss0001_only_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss0001_only_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss0001_only_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss0001_only_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss0001_only_seed42/zscore_std.csv) |
| `ss_only` | 42 | [YAML](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss005_only_seed42.yaml) | `transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42` | [??](../saved_models/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42.pth) | [mean](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42/zscore_mean.csv) / [std](../results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42/zscore_std.csv) |
| `steady_state` | ? | [YAML](../configs/steady_state/tabular_mlp_claus.yaml) | `Tabular_MLP_Claus_Final` | [??](../saved_models/Tabular_MLP_Claus_Final_tabular_mlp.pth) | ????? |

## ?? ANN teacher

PGIN ????? teacher ? `saved_models/Tabular_MLP_5in_2out_QV.pth`??????? `results/Tabular_MLP_New/zscore_mean.csv` ? `zscore_std.csv`?`train_tabular_v2` ??????????????? YAML?

## ??????

| ??? | ??? |
| --- | --- |
| `train.py` | [scripts/training/train.py](../scripts/training/train.py) |
| `predict.py` | [scripts/prediction/predict.py](../scripts/prediction/predict.py) |
| `predict_step_change.py` | [scripts/prediction/predict_step_change.py](../scripts/prediction/predict_step_change.py) |
| `Claus_steady_state/train_tabular_v2.py` | [scripts/training/train_tabular_v2.py](../scripts/training/train_tabular_v2.py) |
| `Claus_steady_state/predict_tabular_v2.py` | [scripts/prediction/predict_tabular_v2.py](../scripts/prediction/predict_tabular_v2.py) |
| `Claus_steady_state/diagnose_ann_gain_pair.py` | [scripts/evaluation/diagnose_ann_gain_pair.py](../scripts/evaluation/diagnose_ann_gain_pair.py) |
| `gain/evaluate_gain_in_out_distribution.py` | [scripts/evaluation/evaluate_gain_in_out_distribution.py](../scripts/evaluation/evaluate_gain_in_out_distribution.py) |
| `LHS/generate_lhs_air2_t2_in_range.py` | [scripts/sampling/generate_lhs_air2_t2_in_range.py](../scripts/sampling/generate_lhs_air2_t2_in_range.py) |
| `LHS/generate_lhs_air2_t2_soft_ood.py` | [scripts/sampling/generate_lhs_air2_t2_soft_ood.py](../scripts/sampling/generate_lhs_air2_t2_soft_ood.py) |
| `LHS/plot_data11_concentration_surfaces.py` | [scripts/plotting/plot_data11_concentration_surfaces.py](../scripts/plotting/plot_data11_concentration_surfaces.py) |
| `logs/annotate_difficult_gradnorm.py` | [scripts/maintenance/annotate_difficult_gradnorm.py](../scripts/maintenance/annotate_difficult_gradnorm.py) |
| `logs/check_ann_step_data_20261013.py` | [scripts/maintenance/check_ann_step_data_20261013.py](../scripts/maintenance/check_ann_step_data_20261013.py) |
| `gain/train_pgin_autoregressive_gain.py` | [scripts/training/train_pgin_autoregressive_gain.py](../scripts/training/train_pgin_autoregressive_gain.py) |
| `gain/train_pgin_from_scratch.py` | [scripts/training/train_pgin_from_scratch.py](../scripts/training/train_pgin_from_scratch.py) |
| `gain/train_pgin_stepwise_gain.py` | [scripts/training/train_pgin_stepwise_gain.py](../scripts/training/train_pgin_stepwise_gain.py) |
| `gain/train_pgin_stepwise_gain_difficult_group_gradnorm.py` | [scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py](../scripts/training/train_pgin_stepwise_gain_difficult_group_gradnorm.py) |
| `gain/train_pgin_stepwise_gain_ss_loss.py` | [scripts/training/train_pgin_stepwise_gain_ss_loss.py](../scripts/training/train_pgin_stepwise_gain_ss_loss.py) |
| `scripts/analyze_distribution_html.py` | [scripts/analysis/analyze_distribution_html.py](../scripts/analysis/analyze_distribution_html.py) |
| `scripts/analyze_gain_pair_difficulty.py` | [scripts/analysis/analyze_gain_pair_difficulty.py](../scripts/analysis/analyze_gain_pair_difficulty.py) |
| `scripts/plot_gain_vs_no_gain_comparison.py` | [scripts/plotting/plot_gain_vs_no_gain_comparison.py](../scripts/plotting/plot_gain_vs_no_gain_comparison.py) |
| `scripts/plot_kci_pair_comparison.py` | [scripts/plotting/plot_kci_pair_comparison.py](../scripts/plotting/plot_kci_pair_comparison.py) |
| `scripts/plot_plant_simulated_distribution.py` | [scripts/plotting/plot_plant_simulated_distribution.py](../scripts/plotting/plot_plant_simulated_distribution.py) |
| `scripts/plot_ss_weight_horizon_comparison.py` | [scripts/plotting/plot_ss_weight_horizon_comparison.py](../scripts/plotting/plot_ss_weight_horizon_comparison.py) |
| `scripts/split_dataset_partitions.py` | [scripts/preprocessing/split_dataset_partitions.py](../scripts/preprocessing/split_dataset_partitions.py) |
| `scripts/split_dynamic_data.py` | [scripts/preprocessing/split_dynamic_data.py](../scripts/preprocessing/split_dynamic_data.py) |

## ??????

- `LHS/LHS_air2_t2_in_range_perturbation.csv` ? `data/sampling/LHS_air2_t2_in_range_perturbation.csv`?
- `LHS/LHS_air2_t2_soft_ood_perturbation.csv` ? `data/sampling/LHS_air2_t2_soft_ood_perturbation.csv`?
- ? `LHS/*Check_Plots/` ? `results/sampling/*Check_Plots/`?

??????????????????????????

## Seed43 / Seed44 archive (2026-10-21)

Seed43 and seed44 configurations, checkpoints and results were copied to the local Desktop `seed43_44` archive and SHA-256 verified before their experiment-specific files were removed from this repository. The archive preserves the original relative paths and includes `EXPORT_README.md` and `export_manifest.json`. It is outside Git; shared data, source code and seed42 experiments remain here.
