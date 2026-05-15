# predict_step_change.py
# Run a trained Transformer on step-change CSV files and save prediction plots.

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# Add the repository root to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils as data_utils
from src.models import get_model
from src.utils import calculate_metrics
from src.variable_selection import variable_selection


CONFIG_PATH = "configs/transformer_layerwise_57var.yaml"
STEP_CHANGE_BASE_DIR = "data/Claus_dynamic/step_change"

DISTRIBUTION_DIRS = {
    "in_training": "in_training_distribution",
    "out_of_training": "out_of_training_distribution",
    "acidgas_fm_170": "acidgas_fm=170",
}

TRUE_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"


def style_prediction_axis(ax, y_ticks=5):
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.9, colors="#1f2937")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#1f2937")
        spine.set_linewidth(1.0)


def predict_block_replacement(model, initial_en_input, future_de_inputs, device, H):
    """Predict in blocks of length H, then replace the encoder window by the predicted block."""
    model.eval()

    num_pred_steps = future_de_inputs.shape[1]
    if num_pred_steps % H != 0:
        num_pred_steps = (num_pred_steps // H) * H
        future_de_inputs = future_de_inputs[:, :num_pred_steps, :]

    num_windows_to_predict = num_pred_steps // H
    predictions_all_windows = []
    current_en_input = initial_en_input.clone().to(device)

    with torch.no_grad():
        for i in range(num_windows_to_predict):
            start_idx = i * H
            end_idx = (i + 1) * H
            de_input_block = future_de_inputs[:, start_idx:end_idx, :].to(device)
            prediction_block = model(current_en_input, de_input_block)
            predictions_all_windows.append(prediction_block)

            # Encoder order is [MV, y_sv].
            current_en_input = torch.cat([de_input_block, prediction_block], dim=2)

    return torch.cat(predictions_all_windows, dim=1)


def predict_sliding_window(model, initial_en_input, future_de_inputs, device, num_output_features):
    """Autoregressive one-step sliding-window prediction."""
    model.eval()

    num_pred_steps = future_de_inputs.shape[1]
    predictions = torch.zeros(1, num_pred_steps, num_output_features).to(device)
    current_en_input = initial_en_input.clone().to(device)

    with torch.no_grad():
        for t in range(num_pred_steps):
            single_step_de_input = future_de_inputs[:, t, :].unsqueeze(1).to(device)
            single_step_prediction = model(current_en_input, single_step_de_input)
            predictions[:, t, :] = single_step_prediction

            next_en_input_history = current_en_input[:, 1:, :]
            new_step_features = torch.cat([single_step_de_input, single_step_prediction], dim=2)
            current_en_input = torch.cat([next_en_input_history, new_step_features], dim=1)

    return predictions


def process_single_file(
    filepath,
    model,
    config,
    mean_all,
    std_all,
    device,
    de_mv,
    y_sv,
    en_mv_and_sv,
    W,
    H,
    inference_strategy,
    warmup_steps=0,
):
    """Load one step-change file, run prediction, inverse-transform outputs, and compute metrics."""
    filename = os.path.basename(filepath)
    df_raw = pd.read_csv(filepath)

    interval = config["window"].get("sampling_interval_min", 1)
    use_median = config["window"].get("use_median_downsampling", True)
    if interval > 1:
        if use_median:
            df_raw = df_raw.rolling(window=interval, min_periods=interval).median(numeric_only=True)
            df_raw = df_raw.iloc[interval - 1 :: interval].reset_index(drop=True)
        else:
            df_raw = df_raw.iloc[::interval].reset_index(drop=True)
        df_raw.dropna(inplace=True)

    if len(df_raw) < W + H:
        print(f"  [SKIP] {filename}: length {len(df_raw)} is shorter than required {W + H}.")
        return None

    all_needed_cols = list(set(en_mv_and_sv + de_mv + y_sv))
    missing_cols = [c for c in all_needed_cols if c not in df_raw.columns]
    if missing_cols:
        print(f"  [SKIP] {filename}: missing columns: {missing_cols}")
        return None

    # Important: use the raw CSV scale here. The saved z-score stats were trained on raw scale.
    df_subset = df_raw[all_needed_cols].copy()

    if df_subset.isnull().any().any():
        print(f"  [WARN] {filename}: data contains NaN; applying forward/backward fill.")
        df_subset = df_subset.ffill().bfill()

    log_cols = [c for c in ["B35_H2S", "B35_SO2"] if c in df_subset.columns]
    if log_cols:
        df_subset = data_utils.apply_log_transform(df_subset, log_cols)

    mean_subset = mean_all[all_needed_cols]
    std_subset = std_all[all_needed_cols]
    std_safe = std_subset.mask(std_subset.abs() < 1e-6, 1.0)
    df_z = (df_subset - mean_subset) / std_safe

    ss1_en_z = df_z.iloc[0][en_mv_and_sv].values
    ss1_de_z = df_z.iloc[0][de_mv].values
    initial_history_np = np.tile(ss1_en_z, (W, 1))
    warmup_de_np = np.tile(ss1_de_z, (warmup_steps, 1))

    real_de_np = df_z.iloc[0:][de_mv].values
    real_targets_np = df_z.iloc[0:][y_sv].values
    full_de_np = np.concatenate([warmup_de_np, real_de_np], axis=0)

    initial_en_input = torch.tensor(initial_history_np, dtype=torch.float32).unsqueeze(0)
    full_de_inputs = torch.tensor(full_de_np, dtype=torch.float32).unsqueeze(0)

    if warmup_steps > 0:
        print(
            f"  [WARM-UP] encoder repeats SS1 for W={W}; "
            f"running {warmup_steps} SS1 decoder steps before prediction."
        )

    if inference_strategy == "block_replacement":
        all_preds_z = predict_block_replacement(model, initial_en_input, full_de_inputs, device, H)
    else:
        all_preds_z = predict_sliding_window(model, initial_en_input, full_de_inputs, device, len(y_sv))

    predictions_z = all_preds_z[:, warmup_steps:, :]
    num_actual_preds = predictions_z.shape[1]
    true_targets_np = real_targets_np[:num_actual_preds, :]

    predictions_np = predictions_z.squeeze(0).cpu().numpy()
    y_mean = mean_all[y_sv].values
    y_std = std_all[y_sv].values
    y_std_safe = np.where(np.abs(y_std) < 1e-6, 1.0, y_std)
    predictions_cov = predictions_np * y_std_safe + y_mean
    true_targets_cov = true_targets_np * y_std_safe + y_mean

    log_cols_inv = [c for c in ["B35_H2S", "B35_SO2"] if c in y_sv]
    if log_cols_inv:
        pred_df_tmp = pd.DataFrame(predictions_cov, columns=y_sv)
        true_df_tmp = pd.DataFrame(true_targets_cov, columns=y_sv)
        pred_df_tmp = data_utils.inverse_log_transform(pred_df_tmp, log_cols_inv)
        true_df_tmp = data_utils.inverse_log_transform(true_df_tmp, log_cols_inv)
        predictions_cov = pred_df_tmp.values
        true_targets_cov = true_df_tmp.values

    if not np.isfinite(predictions_cov).all() or not np.isfinite(true_targets_cov).all():
        print(f"  [WARN] {filename}: predictions or targets contain NaN/Inf.")
        return None

    metrics_results = []
    for i, name in enumerate(y_sv):
        y_true_col = true_targets_cov[:, i]
        y_pred_col = predictions_cov[:, i]
        if not np.isfinite(y_true_col).all() or not np.isfinite(y_pred_col).all():
            metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
        else:
            metrics = calculate_metrics(y_true_col, y_pred_col)
        metrics["Variable"] = name
        metrics_results.append(metrics)

    return {
        "filename": filename,
        "predictions": predictions_cov,
        "true_values": true_targets_cov,
        "metrics": metrics_results,
        "y_sv": y_sv,
        "num_steps": num_actual_preds,
    }


def parse_scenario_conditions(filename):
    """Parse scenario conditions from names like air2_180_t2_150_air2_change_10.csv."""
    name = filename.replace(".csv", "")
    m = re.match(r"air2_(-?\d+)_t2_(-?\d+)_(\w+)_change_(-?\d+)", name)
    if m:
        return {
            "air2": int(m.group(1)),
            "t2": int(m.group(2)),
            "change_var": m.group(3),
            "change_val": int(m.group(4)),
        }
    return {"air2": "?", "t2": "?", "change_var": "?", "change_val": "?"}


def print_conditions_table(csv_files):
    """Print a compact scenario table."""
    print(f"\n{'No.':<5} {'air2':>6} {'t2':>6} {'change_var':<12} {'change_val':>10}  filename")
    print("-" * 75)
    for i, f in enumerate(csv_files, 1):
        c = parse_scenario_conditions(f)
        print(f"{i:<5} {c['air2']:>6} {c['t2']:>6} {c['change_var']:<12} {c['change_val']:>10}  {f}")
    print("-" * 75)


def plot_combined_h2s_so2(dist_results, y_sv, dist_key, output_dir, exp_name):
    """Plot B35_H2S and B35_SO2 for all step-change scenarios, four scenarios per figure."""
    key_vars = [v for v in ["B35_H2S", "B35_SO2"] if v in y_sv]
    if not key_vars or not dist_results:
        return

    n_vars = len(key_vars)
    chunk_size = 4
    dist_label = dist_key.replace("_", " ").title()
    chunks = [dist_results[i : i + chunk_size] for i in range(0, len(dist_results), chunk_size)]

    for fig_idx, chunk in enumerate(chunks):
        n_rows = len(chunk)
        fig, axes = plt.subplots(n_rows, n_vars, figsize=(8 * n_vars, 4 * n_rows), squeeze=False)

        for row_idx, result in enumerate(chunk):
            cond = parse_scenario_conditions(result["filename"])
            cond_str = (
                f"air2={cond['air2']}  t2={cond['t2']}  "
                f"d{cond['change_var']}={cond['change_val']:+d}"
            )
            for col_idx, var in enumerate(key_vars):
                ax = axes[row_idx, col_idx]
                var_idx = y_sv.index(var)
                m = result["metrics"][var_idx]

                l1 = ax.plot(
                    result["true_values"][:, var_idx],
                    label="True (Aspen)",
                    color=TRUE_COLOR,
                    linewidth=1.5,
                )
                l2 = ax.plot(
                    result["predictions"][:, var_idx],
                    label="Predicted",
                    color=PRED_COLOR,
                    linestyle="--",
                    linewidth=1.5,
                )

                ax.set_title(
                    f'[{cond_str}]\n{var}   R2={m["R2"]:.4f}  RMSE={m["RMSE"]:.4f}  MAE={m["MAE"]:.4f}',
                    fontsize=9,
                    fontweight="bold",
                )
                ax.set_xlabel("Time Step")
                ax.set_ylabel(var)
                ax.legend(l1 + l2, [line.get_label() for line in l1 + l2], loc="best", fontsize=8)
                style_prediction_axis(ax)

        plt.tight_layout()

        save_path = os.path.join(output_dir, f"H2S_SO2_combined_part{fig_idx + 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved combined plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Step Change Prediction")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of SS1 decoder warm-up steps before collecting predictions.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Step Change Prediction using Trained Transformer Model")
    print("=" * 70)

    print(f"\n[1] Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prefix = config["exp_name"]
    inference_strategy = config["training"].get("inference_strategy", "sliding_window")
    print(f"    Model: {prefix}")
    print(f"    Inference strategy: {inference_strategy}")

    cfg_data = config["data"]
    cfg_win = config["window"]
    W = cfg_win["train_window_mins"] // cfg_win["sampling_interval_min"]
    H = cfg_win["prediction_length"]

    print("\n[2] Loading training z-score statistics...")
    zscore_mean_path = os.path.join("./results", prefix, "zscore_mean.csv")
    zscore_std_path = os.path.join("./results", prefix, "zscore_std.csv")

    if os.path.exists(zscore_mean_path) and os.path.exists(zscore_std_path):
        mean_all = pd.read_csv(zscore_mean_path, index_col=0).squeeze()
        std_all = pd.read_csv(zscore_std_path, index_col=0).squeeze()
        print(f"    Loaded saved z-score stats: {zscore_mean_path}")
    else:
        print(f"    [WARN] Missing {zscore_mean_path}; recalculating stats from training files.")
        train_dfs = []
        for fname in cfg_data["training_files"]:
            fpath = os.path.join(cfg_data["path"], fname)
            if os.path.exists(fpath):
                df_t = pd.read_csv(fpath)
                interval = cfg_win.get("sampling_interval_min", 1)
                if interval > 1:
                    df_t = df_t.rolling(window=interval, min_periods=interval).median(numeric_only=True)
                    df_t = df_t.iloc[interval - 1 :: interval].reset_index(drop=True)
                df_t.dropna(inplace=True)
                train_dfs.append(df_t)
        df_all = pd.concat(train_dfs, ignore_index=True)
        log_cols = [c for c in ["B35_H2S", "B35_SO2"] if c in df_all.columns]
        if log_cols:
            df_all = data_utils.apply_log_transform(df_all, log_cols)
        mean_all, std_all = data_utils.calculate_zscore_stats(df_all)

    de_mv, y_sv, _, en_mv_and_sv = variable_selection(cfg_data["variables_num"])
    config["data"]["num_en_input"] = len(en_mv_and_sv)
    config["data"]["num_de_input"] = len(de_mv)
    config["data"]["num_output"] = len(y_sv)

    print(f"    Encoder input variables: {len(en_mv_and_sv)}")
    print(f"    Decoder input variables (MV): {de_mv}")
    print(f"    Prediction targets: {len(y_sv)}")
    print(f"    Encoder window W: {W}")

    print("\n[3] Loading model...")
    device = "cuda" if torch.cuda.is_available() and config["training"]["device"] == "cuda" else "cpu"

    model = get_model(config)
    model_path = os.path.join("./saved_models/", f"{prefix}.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"    Model path: {model_path}")
    print(f"    Device: {device}")

    output_base_dir = os.path.join("./results", prefix, "step_change_predictions")
    print("\n[4] Processing step-change files...")
    print(f"    Output directory: {output_base_dir}")

    all_results = {}

    for dist_key, dist_dir in DISTRIBUTION_DIRS.items():
        data_dir = os.path.join(STEP_CHANGE_BASE_DIR, dist_dir)
        output_dir = os.path.join(output_base_dir, dist_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing: {dist_key.upper()} ({dist_dir})")
        print(f"{'=' * 60}")

        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") and "_converted" not in f])
        print(f"Found {len(csv_files)} CSV files.")
        print_conditions_table(csv_files)

        dist_results = []

        for csv_file in tqdm(csv_files, desc=f"Predict {dist_key}"):
            filepath = os.path.join(data_dir, csv_file)
            result = process_single_file(
                filepath,
                model,
                config,
                mean_all,
                std_all,
                device,
                de_mv,
                y_sv,
                en_mv_and_sv,
                W,
                H,
                inference_strategy,
                warmup_steps=args.warmup_steps,
            )

            if result is None:
                continue

            dist_results.append(result)

            file_output_dir = os.path.join(output_dir, csv_file.replace(".csv", ""))
            os.makedirs(file_output_dir, exist_ok=True)

            df_true = pd.DataFrame(result["true_values"], columns=y_sv)
            df_pred = pd.DataFrame(result["predictions"], columns=[f"{col}_pred" for col in y_sv])
            df_results = pd.concat([df_true, df_pred], axis=1)
            df_results.to_csv(os.path.join(file_output_dir, "predictions.csv"), index=False)

            df_metrics = pd.DataFrame(result["metrics"])
            df_metrics = df_metrics[["Variable", "MAE", "RMSE", "R2", "MAPE"]]
            df_metrics.to_csv(os.path.join(file_output_dir, "metrics.csv"), index=False)

            n_vars = len(y_sv)
            n_cols = 2
            n_rows = (n_vars + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
            axes = np.asarray(axes).flatten()

            for idx, var in enumerate(y_sv):
                ax = axes[idx]
                var_metrics = result["metrics"][idx]
                l1 = ax.plot(result["true_values"][:, idx], label="True (Aspen)", color=TRUE_COLOR, linewidth=1.2)
                l2 = ax.plot(
                    result["predictions"][:, idx],
                    label="Predicted",
                    color=PRED_COLOR,
                    linestyle="--",
                    linewidth=1.2,
                )
                ax.set_title(
                    f'{var}\nR2={var_metrics["R2"]:.4f}, RMSE={var_metrics["RMSE"]:.4f}',
                    fontsize=10,
                    fontweight="bold",
                )
                ax.set_xlabel("Time Step")
                ax.set_ylabel(var)
                ax.legend(l1 + l2, [line.get_label() for line in l1 + l2], loc="best", fontsize=8)
                style_prediction_axis(ax)

            for idx in range(n_vars, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(file_output_dir, "all_variables.png"), dpi=150, bbox_inches="tight")
            plt.close()

            for var in ["B35_H2S", "B35_SO2"]:
                if var not in y_sv:
                    continue

                idx = y_sv.index(var)
                var_metrics = result["metrics"][idx]

                fig, ax = plt.subplots(figsize=(14, 5))
                l1 = ax.plot(result["true_values"][:, idx], label="True (Aspen)", color=TRUE_COLOR, linewidth=1.5)
                l2 = ax.plot(
                    result["predictions"][:, idx],
                    label="Predicted",
                    color=PRED_COLOR,
                    linestyle="--",
                    linewidth=1.5,
                )
                title = f'{csv_file.replace(".csv", "")} - {var}\n'
                title += (
                    f'MAE={var_metrics["MAE"]:.6f}, '
                    f'RMSE={var_metrics["RMSE"]:.6f}, '
                    f'R2={var_metrics["R2"]:.4f}'
                )
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel("Time Step (minutes)")
                ax.set_ylabel(var)
                ax.legend(l1 + l2, [line.get_label() for line in l1 + l2], loc="best")
                style_prediction_axis(ax)
                fig.tight_layout()
                plt.savefig(os.path.join(file_output_dir, f"{var}.png"), dpi=150)
                plt.close()

        all_results[dist_key] = dist_results

        if dist_results:
            print(f"\n--- {dist_key.upper()} Summary Metrics ---")

            summary_data = {var: {"MAE": [], "RMSE": [], "R2": [], "MAPE": []} for var in y_sv}
            for result in dist_results:
                for m in result["metrics"]:
                    var = m["Variable"]
                    summary_data[var]["MAE"].append(m["MAE"])
                    summary_data[var]["RMSE"].append(m["RMSE"])
                    summary_data[var]["R2"].append(m["R2"])
                    summary_data[var]["MAPE"].append(m["MAPE"])

            summary_rows = []
            for var in y_sv:
                row = {
                    "Variable": var,
                    "MAE_mean": np.mean(summary_data[var]["MAE"]),
                    "MAE_std": np.std(summary_data[var]["MAE"]),
                    "RMSE_mean": np.mean(summary_data[var]["RMSE"]),
                    "RMSE_std": np.std(summary_data[var]["RMSE"]),
                    "R2_mean": np.mean(summary_data[var]["R2"]),
                    "R2_std": np.std(summary_data[var]["R2"]),
                    "MAPE_mean": np.mean(summary_data[var]["MAPE"]),
                    "MAPE_std": np.std(summary_data[var]["MAPE"]),
                }
                summary_rows.append(row)

                if var in ["B35_H2S", "B35_SO2"]:
                    print(f"  {var}:")
                    print(f"    MAE:  {row['MAE_mean']:.6f} +/- {row['MAE_std']:.6f}")
                    print(f"    RMSE: {row['RMSE_mean']:.6f} +/- {row['RMSE_std']:.6f}")
                    print(f"    R2:   {row['R2_mean']:.4f} +/- {row['R2_std']:.4f}")

            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
            print(f"  Saved summary metrics: {output_dir}/summary_metrics.csv")

            plot_combined_h2s_so2(dist_results, y_sv, dist_key, output_dir, prefix)

    print(f"\n{'=' * 70}")
    print("Comparison: In-Training vs Out-of-Training Distribution")
    print(f"{'=' * 70}")

    for var in ["B35_H2S", "B35_SO2"]:
        if var not in y_sv:
            continue
        idx = y_sv.index(var)

        in_r2 = [r["metrics"][idx]["R2"] for r in all_results.get("in_training", [])]
        out_r2 = [r["metrics"][idx]["R2"] for r in all_results.get("out_of_training", [])]

        if in_r2 and out_r2:
            print(f"\n{var}:")
            print(f"  In-training R2:     {np.mean(in_r2):.4f} +/- {np.std(in_r2):.4f}")
            print(f"  Out-of-training R2: {np.mean(out_r2):.4f} +/- {np.std(out_r2):.4f}")

    print(f"\n{'=' * 70}")
    print(f"Prediction complete. Results saved to: {output_base_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
