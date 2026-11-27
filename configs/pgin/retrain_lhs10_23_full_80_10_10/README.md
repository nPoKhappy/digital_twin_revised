# Full-trajectory 80/10/10 retraining

This separate experiment trains the same five model families and three seeds
using the complete R5, R5-1, R5-2, and R5-6 dynamic CSVs. For each raw
trajectory, the first 80% is training, the next 10% is the training-time
validation set for checkpoint selection and early stopping, and the last 10%
is held out from training. Each segment is downsampled independently after
splitting, and dynamic scaler statistics are fitted only on the training
segment. The existing `*_test_split.csv` files match the raw held-out segments.
These previously used test cases are not a fresh blind test.

Each seed has 21 configurations: No-Gain (1), Gain (5), SS-only (5), Gain+SS
(5), and Difficult-GradNorm (5). Existing hyperparameters, including DG caps
estimated from the previous training protocol, are retained; the caps have not
been recalibrated for this split. The shared ANN teacher is unchanged.

From the repository root, check or run one seed at a time per terminal:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.training.run_retrain_lhs10_23_full_80_10_10 --seed 42 --check-only
.\.venv\Scripts\python.exe -u -m scripts.training.run_retrain_lhs10_23_full_80_10_10 --seed 42
```

Replace `42` with `43` or `44`. Runs write new checkpoints and outputs without
overwriting the earlier experiment. A failed or interrupted batch does not
automatically resume; inspect its record and active processes before retrying.
The per-model prediction entry point is
`scripts.prediction.predict_full_80_10_10`, which checks checkpoint and scaler
hashes before inference. The earlier 63-model validation and selected-test
aggregation scripts assume old checkpoints and cached predictions; they must be
adapted and rerun before reporting results from this experiment.
