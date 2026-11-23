# Selected test results

Use [`updated_validation_best/`](updated_validation_best/) for the current
three-seed test evaluation after the DG 95th-percentile cap retraining. The
validation-selected families are No-Gain, Gain 25, SS-only 2.5, Gain+SS
25/25, and Difficult-GradNorm rho=0.15, each for seeds 42, 43, and 44.
See `updated_validation_best/README_test_protocol.md` and
`updated_validation_best/selected_models.json` for the protocol and exact
model selection.

The latest selected models, predictions, figures, and provenance are kept in
`updated_validation_best/`. Older test result files have been removed.

## Historical provenance limitations

The retained `../retrain_lhs10_23_dg95_validation/reference_non_dg/previous_provenance.json`
is an unchanged snapshot of an earlier analysis. Its recorded source hashes
do not mean those historical files remain available. The following five files
under `../retrain_lhs10_23_63_models_validation_nogain42_baseline_previous_before_dg_expansion/`
were already unavailable before the latest cleanup: `provenance.json`,
`analysis_validation.json`, `baseline_raw_rmse.csv`,
`relative_rmse_components.csv`, and `regional_coordinates_63_models.csv`.

The latest cleanup additionally removed these three historical files from
this parent directory. Their original SHA-256 values, as recorded in the
snapshot, are:

| Historical file | Original SHA-256 |
| --- | --- |
| `validation_selection_plan.json` | `30382d60f5e186c19eb16e8465d2700cff12538276cc1843d260d1217336f88f` |
| `selected_models.json` | `2fde8ca7245d0e121ad5dac292e0e8eadf9f0cc80fa96e0bb513f37f5a3dd8be` |
| `run_protocol.json` | `1d16600d39c327905d5680219678d7abc79ab222926638722ffa13c276536d15` |

Files with the same names in `updated_validation_best/` are the current
results; they do not replace or verify the deleted historical versions. The
current test protocol's active sources and both current KCI plot provenance
records were checked after cleanup and their recorded hashes matched.
