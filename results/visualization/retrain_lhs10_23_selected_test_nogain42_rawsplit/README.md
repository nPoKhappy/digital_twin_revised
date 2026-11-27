# Raw-split validation-selected test comparison

Selection was frozen from the raw-split 63-model validation results before running test inference.
Per family, the three-seed means of the combined regional AvgRelRMSE and KCI are normalized
within that family; the setting closest to lowest error and highest KCI is selected.

Each selected setting has seeds 42, 43, and 44; No-Gain also has all three seeds.
All 15 models were evaluated from their new checkpoints without reusing previous predictions.
AvgRelRMSE uses the matching test case, output and horizon of the new No-Gain seed 42 as baseline.
KCI is directional agreement with the shared ANN teacher on steady-state test probes.
The dynamic test cases and steady-state probes were previously used; this is not a new blind test.
Dynamic inference uses recorded future decoder inputs. Seeds quantify dynamic-training variation only.

Selected validation aliases:

- gain: gain25
- ss_only: ss2p5_only
- gain_ss: gain250_ss250
- difficult_gradnorm: dg_ratio1p5

The interactive HTML and PNG/PDF summary are under `three_seed_report/`;
the journal-ready figure and separate caption are under `jtice_figures/`.
Use `independent_npz_audit.json` for the independent recomputation of output metrics.
