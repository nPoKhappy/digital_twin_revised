## Environment constraints
- The development environment is Windows.
- The primary shell is PowerShell, not Bash or zsh.
- Python is managed using `uv`.
- When running Python scripts or tools, always use `uv run ...`.
- Do NOT suggest `source`, `activate`, or Unix/macOS-specific shell commands.
- Assume Python is already installed and managed by `uv`.
- Prefer one-line commands that work directly in PowerShell.

## Big picture architecture
- This repo has two ML workflows:
	- Dynamic time-series seq2seq forecasting (main): `train.py` + `predict.py`.
	- Steady-state tabular regression: `Claus_steady_state/train_tabular.py` + `Claus_steady_state/predict_tabular.py`.
- Shared core modules live in `src/`:
	- Data shaping: `src/dataset.py` (`MultiStepS2SDataset`) and `src/dataset_tabular.py`.
	- Training/eval loops: `src/engine.py`.
	- Variable schema routing by `variables_num`: `src/variable_selection.py`.
	- Model factory and implementations: `src/models/__init__.py` and `src/models/*`.

## Critical data/feature flow (dynamic pipeline)
- `variables_num` in YAML selects `(de_mv, y_sv, en_mv_and_sv)` via `variable_selection()`; this drives model I/O dimensions.
- Training preprocessing order in `train.py` is intentional and should stay aligned with inference:
	1) per-file downsampling (`rolling(...).median(numeric_only=True)` when enabled),
	2) `dropna`,
	3) log transform on `B35_H2S`/`B35_SO2`,
	4) global Z-score stats across concatenated training segments,
	5) per-segment scaling.
- Inference (`predict.py`) must mirror the same transform order and reuse saved stats from `results/<exp_name>/zscore_mean.csv` and `zscore_std.csv`.

## Training strategy conventions
- `training.loss_weighting.weights` controls AT rolling mode in `src/engine.py`:
	- If multiple weights exist, training uses block replacement (`step_wise_rolling_at_loss_step`).
	- Weight count must match the number of predicted blocks (`total_pred_len / prediction_length`).
- `window.train_window_mins / sampling_interval_min` defines encoder history length `W`.
- `window.prediction_length` defines block size `H`.

## Developer workflows used in this repo
- Dynamic training: `uv run python train.py --config configs/<config>.yaml`
- Dynamic prediction: `uv run python predict.py --config configs/<config>.yaml`
- Optuna tuning: `uv run python train_optuna.py --trials <N>`
- Steady-state tabular: 
	- `uv run python Claus_steady_state/train_tabular.py --config configs/<config>.yaml`
	- `uv run python Claus_steady_state/predict_tabular.py --config configs/<config>.yaml`

## Project-specific pitfalls to respect
- Active scripts are `train.py` and `predict.py`.
- Keep encoder feature order consistent with `[de_mv, y_sv]` assumptions used when rolling history is rebuilt in training and prediction.
- `exp_name` is the run key: models save to `saved_models/<exp_name>.pth`; metrics/plots save under `results/<exp_name>/`.
- Some older files reference `src.data_utils`; canonical utilities are in `src/utils.py` (often imported as `data_utils`).

