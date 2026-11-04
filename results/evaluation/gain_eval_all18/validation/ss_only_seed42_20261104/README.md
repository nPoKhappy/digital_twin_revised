# SS-only validation KCI

Five retained frozen SS-only checkpoints (SS=0.05, 2.5, 25, 250, 2500); Gain loss=0; seed42. Same full validation probes as Gain: in-range 4404, Soft-OOD 2585. Warmup35; post-step1-18; both perturbation signs; strict agreement with ANN reference.

5,032,080 diagnostic rows recounted for the retained models; all per-model validations and protected input hashes passed.

[Figures and scores](../../../../visualization/ss_only_all_weights_casewise_region50_validation_seed42/README.md)

[Pair counts](ss_only_pair_kci_comparison.csv) | [Regional KCI](ss_only_overall_kci_comparison.csv) | [Verification](evaluation_validation.json)

Raw native training_type is based on experiment name and labels non-No-Gain names as gain. Actual SS-only objectives were verified from configs (gain_loss_weight=0); canonical objective metadata is available in the figure output ss_only_kci_components.csv.
