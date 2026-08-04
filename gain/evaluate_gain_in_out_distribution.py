"""Evaluate trained Gain/No-Gain Transformers on held-out in/OOD steady states.

The input files must be completed Aspen steady-state exports containing all
63-variable model columns. Generator input CSVs are not sufficient because a
Transformer gain rollout needs a complete encoder history at each probe point.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gain.train_pgin_stepwise_gain import (
    SimpleTabularMLP,
    dataframe_to_z_tensor,
    load_steady_state_source,
)
from src import utils as data_utils
from src.models import get_model
from src.variable_selection import variable_selection


DEFAULT_CONFIGS = [
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml",
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed43.yaml",
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed44.yaml",
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml",
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed43.yaml",
    "configs/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed44.yaml",
]

TAB_INPUT_COLS = [
    "air2_SP",
    "HEATER2_output_T_SP",
    "acidgas_Fm",
    "acidgas_P",
    "acidgas_T",
]
FULL_TARGET_COLS = ["B35_H2S", "B35_SO2"]
CORE_AIR2_RANGE = (140.0, 300.0)
CORE_T2_RANGE = (140.0, 240.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare trained Gain/No-Gain Transformer gain consistency on "
            "held-out in-range and out-of-range Aspen steady-state exports."
        )
    )
    parser.add_argument(
        "--in-range",
        required=True,
        help="Held-out in-range completed Aspen XLSX/CSV path or glob.",
    )
    parser.add_argument(
        "--out-range",
        required=True,
        help="Held-out soft-OOD completed Aspen XLSX/CSV path or glob.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Model configs to evaluate. Defaults to the six seed42-44 configs.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=512,
        help="Fixed probes per distribution; 0 uses every usable row.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260729)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        default="results/gain_in_out_distribution_evaluation",
    )
    return parser.parse_args()


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def model_contract(config):
    de_mv, y_sv, _, en_mv_and_sv = variable_selection(
        config["data"]["variables_num"]
    )
    config["data"]["num_en_input"] = len(en_mv_and_sv)
    config["data"]["num_de_input"] = len(de_mv)
    config["data"]["num_output"] = len(y_sv)

    gain_target_qv = config["training"].get(
        "gain_target_qv", FULL_TARGET_COLS
    )
    gain_target_mv = config["training"].get(
        "gain_target_mv", ["air2_SP", "HEATER2_output_T_SP"]
    )
    target_cols = [
        col
        for col in gain_target_qv
        if col in y_sv and col in FULL_TARGET_COLS
    ]
    mv_cols = [
        col
        for col in gain_target_mv
        if col in de_mv and col in TAB_INPUT_COLS
    ]
    if not target_cols or not mv_cols:
        raise ValueError(
            "Config has no valid gain target/output pair for this evaluator."
        )

    keep_cols = []
    for col in en_mv_and_sv + y_sv + TAB_INPUT_COLS:
        if col not in keep_cols:
            keep_cols.append(col)
    return de_mv, y_sv, en_mv_and_sv, target_cols, mv_cols, keep_cols


def load_model_stats(exp_name):
    result_dir = os.path.join("results", exp_name)
    mean_path = os.path.join(result_dir, "zscore_mean.csv")
    std_path = os.path.join(result_dir, "zscore_std.csv")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"Missing Transformer z-score stats under {result_dir}"
        )
    mean = pd.read_csv(mean_path, index_col=0).squeeze("columns")
    std = (
        pd.read_csv(std_path, index_col=0)
        .squeeze("columns")
        .replace(0, 1)
    )
    return mean, std


def assert_same_scaling(configs, required_cols):
    reference_name = configs[0][1]["exp_name"]
    reference_mean, reference_std = load_model_stats(reference_name)
    for _, config in configs[1:]:
        exp_name = config["exp_name"]
        mean, std = load_model_stats(exp_name)
        if not np.allclose(
            reference_mean[required_cols].values,
            mean[required_cols].values,
            rtol=0,
            atol=1e-12,
        ) or not np.allclose(
            reference_std[required_cols].values,
            std[required_cols].values,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{exp_name} does not share the same Transformer scaling as "
                f"{reference_name}; Gain/No-Gain comparison would not use "
                "identical physical finite-difference steps."
            )


def assert_same_contract(reference, current, config_path):
    ref_de, ref_y, ref_en, ref_targets, ref_mvs, ref_keep = reference
    cur_de, cur_y, cur_en, cur_targets, cur_mvs, cur_keep = current
    if (
        ref_de != cur_de
        or ref_y != cur_y
        or ref_en != cur_en
        or ref_targets != cur_targets
        or ref_mvs != cur_mvs
        or ref_keep != cur_keep
    ):
        raise ValueError(
            f"Model contract differs from the first config: {config_path}"
        )


def choose_probe_indices(row_count, max_points, seed):
    if max_points < 0:
        raise ValueError("--max-points must be >= 0.")
    if max_points == 0 or row_count <= max_points:
        return np.arange(row_count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=max_points, replace=False))


def validate_distribution_file(data, distribution):
    inside_core = (
        data["air2_SP"].between(*CORE_AIR2_RANGE, inclusive="both")
        & data["HEATER2_output_T_SP"].between(
            *CORE_T2_RANGE, inclusive="both"
        )
    )
    expected_fraction = (
        float(inside_core.mean())
        if distribution == "in_range"
        else float((~inside_core).mean())
    )
    print(
        f"[Info] {distribution} range-label match: "
        f"{expected_fraction * 100:.2f}%"
    )
    if expected_fraction < 0.95:
        raise ValueError(
            f"{distribution} file has only {expected_fraction * 100:.2f}% "
            "rows matching its expected air2/T2 range."
        )


def load_teacher(device):
    mean_path = "results/Tabular_MLP_New/zscore_mean.csv"
    std_path = "results/Tabular_MLP_New/zscore_std.csv"
    model_path = "saved_models/Tabular_MLP_5in_2out_QV.pth"
    for path in (mean_path, std_path, model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required ANN teacher artifact missing: {path}")

    tab_mean = pd.read_csv(mean_path, index_col=0).squeeze("columns")
    tab_std = (
        pd.read_csv(std_path, index_col=0)
        .squeeze("columns")
        .replace(0, 1)
    )
    missing = [
        col
        for col in TAB_INPUT_COLS + FULL_TARGET_COLS
        if col not in tab_mean.index or col not in tab_std.index
    ]
    if missing:
        raise ValueError(f"ANN teacher z-score stats missing: {missing}")

    model = SimpleTabularMLP(
        input_dim=len(TAB_INPUT_COLS),
        output_dim=len(FULL_TARGET_COLS),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.requires_grad_(False)

    target_mean = torch.tensor(
        tab_mean[FULL_TARGET_COLS].values,
        dtype=torch.float32,
        device=device,
    )
    target_std = torch.tensor(
        tab_std[FULL_TARGET_COLS].values,
        dtype=torch.float32,
        device=device,
    )
    target_std = torch.where(
        torch.abs(target_std) < 1e-6,
        torch.ones_like(target_std),
        target_std,
    )
    return model, tab_mean, tab_std, target_mean, target_std


def teacher_gain_matrix(
    batch_df,
    teacher,
    tab_mean,
    tab_std,
    target_mean,
    target_std,
    target_cols,
    mv_cols,
    device,
):
    teacher_input = dataframe_to_z_tensor(
        batch_df, TAB_INPUT_COLS, tab_mean, tab_std, device
    )
    teacher_input.requires_grad_(True)
    teacher_output_z = teacher(teacher_input)
    teacher_output_p = teacher_output_z * target_std + target_mean

    gain = torch.zeros(
        len(batch_df),
        len(target_cols),
        len(mv_cols),
        dtype=torch.float32,
        device=device,
    )
    for target_idx, target_name in enumerate(target_cols):
        full_target_idx = FULL_TARGET_COLS.index(target_name)
        grad_outputs = torch.zeros_like(teacher_output_p)
        grad_outputs[:, full_target_idx] = 1.0
        grads, = torch.autograd.grad(
            outputs=teacher_output_p,
            inputs=teacher_input,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=target_idx < len(target_cols) - 1,
        )
        for mv_idx, mv_name in enumerate(mv_cols):
            input_idx = TAB_INPUT_COLS.index(mv_name)
            scale = float(tab_std.get(mv_name, 1.0))
            if abs(scale) < 1e-6:
                scale = 1.0
            gain[:, target_idx, mv_idx] = grads[:, input_idx] / scale
    return gain.detach()


def build_history(batch_df, mean, std, en_cols, window, device):
    z_df = data_utils.apply_zscore(batch_df, mean, std).fillna(0.0)
    row_z = torch.tensor(
        z_df[en_cols].values,
        dtype=torch.float32,
        device=device,
    )
    return row_z.unsqueeze(1).repeat(1, window, 1)


def decoder_physical_to_z(values, de_mv, mean, std):
    columns = []
    for col_idx, col in enumerate(de_mv):
        scale = float(std.get(col, 1.0))
        if abs(scale) < 1e-6:
            scale = 1.0
        columns.append((values[:, col_idx] - float(mean.get(col, 0.0))) / scale)
    return torch.stack(columns, dim=1)


def rollout_targets(
    model,
    start_history,
    base_de_p,
    step_de_p,
    de_mv,
    target_indices,
    mean,
    std,
    y_sv,
    rollout_steps,
    step_change_idx,
):
    base_de_z = decoder_physical_to_z(base_de_p, de_mv, mean, std)
    step_de_z = (
        base_de_z
        if step_de_p is None
        else decoder_physical_to_z(step_de_p, de_mv, mean, std)
    )
    y_mean = torch.tensor(
        mean[y_sv].values,
        dtype=torch.float32,
        device=start_history.device,
    )
    y_std = torch.tensor(
        std[y_sv].replace(0, 1).values,
        dtype=torch.float32,
        device=start_history.device,
    )

    history = start_history.clone()
    target_steps = []
    for step_idx in range(rollout_steps):
        current_de_z = (
            step_de_z if step_idx >= step_change_idx else base_de_z
        )
        decoder_input = current_de_z.unsqueeze(1)
        _, context = model.encoder(history)
        prediction_z = model.decoder(decoder_input, context)
        prediction_log_or_p = prediction_z.squeeze(1) * y_std + y_mean
        target_steps.append(prediction_log_or_p[:, target_indices])

        if step_idx < rollout_steps - 1:
            next_features = torch.cat(
                [decoder_input, prediction_z.detach()],
                dim=2,
            )
            history = torch.cat([history[:, 1:, :], next_features], dim=1)
    return torch.stack(target_steps, dim=0)


@torch.no_grad()
def dynamic_gain_matrix(
    model,
    batch_df,
    mean,
    std,
    de_mv,
    y_sv,
    en_cols,
    target_cols,
    mv_cols,
    config,
    device,
):
    window = int(
        config["window"]["train_window_mins"]
        / config["window"]["sampling_interval_min"]
    )
    output_steps = int(config["window"]["prediction_length"])
    warmup_steps = int(
        config["training"].get("stepwise_gain_warmup_steps", 60)
    )
    rollout_steps = int(
        config["training"].get(
            "stepwise_gain_steps", warmup_steps + output_steps
        )
    )
    step_change_step = int(
        config["training"].get(
            "dynamic_gain_step_change_step", warmup_steps + 1
        )
    )
    step_change_idx = step_change_step - 1
    configured_tail_start = config["training"].get(
        "stepwise_gain_tail_start_step",
        max(
            config["training"].get(
                "dynamic_gain_tail_start_step", step_change_step
            ),
            step_change_step,
        ),
    )
    tail_start_idx = (
        min(max(int(configured_tail_start), step_change_step) - 1, rollout_steps - 1)
    )
    finite_diff_delta_std = float(
        config["training"].get("finite_diff_delta_std", 0.5)
    )

    history = build_history(batch_df, mean, std, en_cols, window, device)
    base_de_p = torch.tensor(
        batch_df[de_mv].values,
        dtype=torch.float32,
        device=device,
    )
    target_indices = [y_sv.index(col) for col in target_cols]

    baseline = rollout_targets(
        model,
        history,
        base_de_p,
        None,
        de_mv,
        target_indices,
        mean,
        std,
        y_sv,
        rollout_steps,
        step_change_idx,
    )[tail_start_idx:].mean(dim=0)

    plus_gains = []
    minus_gains = []
    for mv_name in mv_cols:
        mv_idx = de_mv.index(mv_name)
        delta = finite_diff_delta_std * float(std[mv_name])
        if abs(delta) < 1e-12:
            raise ValueError(f"Finite-difference delta is zero for {mv_name}.")

        plus_de_p = base_de_p.clone()
        minus_de_p = base_de_p.clone()
        plus_de_p[:, mv_idx] += delta
        minus_de_p[:, mv_idx] -= delta

        plus = rollout_targets(
            model,
            history,
            base_de_p,
            plus_de_p,
            de_mv,
            target_indices,
            mean,
            std,
            y_sv,
            rollout_steps,
            step_change_idx,
        )[tail_start_idx:].mean(dim=0)
        minus = rollout_targets(
            model,
            history,
            base_de_p,
            minus_de_p,
            de_mv,
            target_indices,
            mean,
            std,
            y_sv,
            rollout_steps,
            step_change_idx,
        )[tail_start_idx:].mean(dim=0)

        plus_gains.append((plus - baseline) / delta)
        minus_gains.append((baseline - minus) / delta)

    plus_matrix = torch.stack(plus_gains, dim=2)
    minus_matrix = torch.stack(minus_gains, dim=2)
    return torch.stack([plus_matrix, minus_matrix], dim=0)


def empty_counts(target_cols, mv_cols):
    return {
        (target, mv): {
            "possible": 0,
            "valid": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
        for target in target_cols
        for mv in mv_cols
    }


def update_counts(
    counts,
    teacher_gain,
    dynamic_gain,
    target_cols,
    mv_cols,
    valid_threshold,
):
    teacher_sign = torch.sign(teacher_gain).unsqueeze(0)
    valid = (
        torch.abs(teacher_gain) >= valid_threshold
    ).unsqueeze(0).expand_as(dynamic_gain)
    teacher_positive = teacher_sign > 0
    dynamic_positive = dynamic_gain > 0

    for target_idx, target in enumerate(target_cols):
        for mv_idx, mv in enumerate(mv_cols):
            mask = valid[:, :, target_idx, mv_idx]
            truth = teacher_positive[:, :, target_idx, mv_idx]
            pred = dynamic_positive[:, :, target_idx, mv_idx]
            pair = counts[(target, mv)]
            pair["possible"] += int(mask.numel())
            pair["valid"] += int(mask.sum().item())
            pair["tp"] += int((mask & truth & pred).sum().item())
            pair["tn"] += int((mask & ~truth & ~pred).sum().item())
            pair["fp"] += int((mask & ~truth & pred).sum().item())
            pair["fn"] += int((mask & truth & ~pred).sum().item())


def safe_percent(numerator, denominator):
    if denominator == 0:
        return np.nan
    return 100.0 * numerator / denominator


def count_row(
    exp_name,
    distribution,
    target,
    mv,
    counts,
    point_count,
):
    valid = counts["valid"]
    positive = counts["tp"] + counts["fn"]
    correct = counts["tp"] + counts["tn"]
    kci = safe_percent(correct, valid)
    seed_match = re.search(r"_seed(\d+)", exp_name)
    return {
        "experiment": exp_name,
        "training_type": "no_gain" if "no_gain" in exp_name else "gain",
        "seed": int(seed_match.group(1)) if seed_match else np.nan,
        "distribution": distribution,
        "target": target,
        "mv": mv,
        "probe_points": point_count,
        "possible_comparisons": counts["possible"],
        "valid_comparisons": valid,
        "coverage_percent": safe_percent(valid, counts["possible"]),
        "teacher_positive_percent": safe_percent(positive, valid),
        "dynamic_positive_percent": safe_percent(
            counts["tp"] + counts["fp"], valid
        ),
        "kci_percent": kci,
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
    }


def evaluate_model_on_distribution(
    config,
    config_path,
    data,
    distribution,
    teacher_artifacts,
    contract,
    batch_size,
    device,
):
    de_mv, y_sv, en_cols, target_cols, mv_cols, _ = contract
    exp_name = config["exp_name"]
    mean, std = load_model_stats(exp_name)
    missing_stats = [
        col
        for col in set(en_cols + de_mv + y_sv)
        if col not in mean.index or col not in std.index
    ]
    if missing_stats:
        raise ValueError(
            f"{exp_name} Transformer stats missing columns: {missing_stats}"
        )

    model_path = os.path.join("saved_models", f"{exp_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer checkpoint not found: {model_path}")
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    teacher, tab_mean, tab_std, target_mean, target_std = teacher_artifacts
    counts = empty_counts(target_cols, mv_cols)
    threshold = float(
        config["training"].get("gain_valid_delta_threshold", 1e-5)
    )

    for start in range(0, len(data), batch_size):
        batch_df = data.iloc[start : start + batch_size].reset_index(drop=True)
        teacher_gain = teacher_gain_matrix(
            batch_df,
            teacher,
            tab_mean,
            tab_std,
            target_mean,
            target_std,
            target_cols,
            mv_cols,
            device,
        )
        dynamic_gain = dynamic_gain_matrix(
            model,
            batch_df,
            mean,
            std,
            de_mv,
            y_sv,
            en_cols,
            target_cols,
            mv_cols,
            config,
            device,
        )
        update_counts(
            counts,
            teacher_gain,
            dynamic_gain,
            target_cols,
            mv_cols,
            threshold,
        )

    rows = []
    for target in target_cols:
        for mv in mv_cols:
            rows.append(
                count_row(
                    exp_name,
                    distribution,
                    target,
                    mv,
                    counts[(target, mv)],
                    len(data),
                )
            )
    print(
        f"[Done] {exp_name} | {distribution} | "
        f"{len(data)} fixed probe points"
    )
    return rows


def build_comparison(detail_df):
    gain_df = detail_df[detail_df["training_type"] == "gain"].copy()
    no_gain_df = detail_df[
        detail_df["training_type"] == "no_gain"
    ].copy()
    keys = ["seed", "distribution", "target", "mv"]
    value_cols = [
        "kci_percent",
        "coverage_percent",
    ]
    comparison = gain_df[keys + value_cols].merge(
        no_gain_df[keys + value_cols],
        on=keys,
        how="inner",
        suffixes=("_gain", "_no_gain"),
        validate="one_to_one",
    )
    comparison["kci_percent_gain_minus_no_gain"] = (
        comparison["kci_percent_gain"]
        - comparison["kci_percent_no_gain"]
    )
    return comparison


def build_summary(detail_df):
    rows = []
    for (experiment, training_type, seed, distribution), group in detail_df.groupby(
        ["experiment", "training_type", "seed", "distribution"],
        dropna=False,
    ):
        possible = int(group["possible_comparisons"].sum())
        valid = int(group["valid_comparisons"].sum())
        correct = int((group["tp"] + group["tn"]).sum())
        rows.append(
            {
                "experiment": experiment,
                "training_type": training_type,
                "seed": seed,
                "distribution": distribution,
                "probe_points": int(group["probe_points"].iloc[0]),
                "possible_comparisons": possible,
                "valid_comparisons": valid,
                "coverage_percent": safe_percent(valid, possible),
                "micro_kci_percent": safe_percent(correct, valid),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Info] Device: {device}")

    configs = [(path, load_yaml(path)) for path in args.configs]
    reference_contract = model_contract(configs[0][1])
    for config_path, config in configs[1:]:
        assert_same_contract(
            reference_contract,
            model_contract(config),
            config_path,
        )
    required_scaling_cols = list(
        dict.fromkeys(
            reference_contract[0]
            + reference_contract[1]
            + reference_contract[2]
        )
    )
    assert_same_scaling(configs, required_scaling_cols)
    keep_cols = reference_contract[-1]

    in_data = load_steady_state_source(
        args.in_range,
        keep_cols,
        log_target_cols=FULL_TARGET_COLS,
    )
    out_data = load_steady_state_source(
        args.out_range,
        keep_cols,
        log_target_cols=FULL_TARGET_COLS,
    )
    validate_distribution_file(in_data, "in_range")
    validate_distribution_file(out_data, "out_range")
    in_indices = choose_probe_indices(
        len(in_data), args.max_points, args.sample_seed
    )
    out_indices = choose_probe_indices(
        len(out_data), args.max_points, args.sample_seed + 1
    )
    probe_sets = {
        "in_range": in_data.iloc[in_indices].reset_index(drop=True),
        "out_range": out_data.iloc[out_indices].reset_index(drop=True),
    }
    print(
        f"[Info] Fixed probes | in_range={len(probe_sets['in_range'])}, "
        f"out_range={len(probe_sets['out_range'])}"
    )

    teacher_artifacts = load_teacher(device)
    detail_rows = []
    for config_path, config in configs:
        for distribution, data in probe_sets.items():
            detail_rows.extend(
                evaluate_model_on_distribution(
                    config,
                    config_path,
                    data,
                    distribution,
                    teacher_artifacts,
                    reference_contract,
                    args.batch_size,
                    device,
                )
            )

    os.makedirs(args.output_dir, exist_ok=True)
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(args.output_dir, "gain_evaluation_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    summary_df = build_summary(detail_df)
    summary_path = os.path.join(args.output_dir, "gain_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    comparison_df = build_comparison(detail_df)
    comparison_path = os.path.join(
        args.output_dir, "gain_vs_no_gain_comparison.csv"
    )
    comparison_df.to_csv(comparison_path, index=False)

    print(f"[Save] Detailed metrics: {detail_path}")
    print(f"[Save] Model/distribution summary: {summary_path}")
    print(f"[Save] Gain vs No-Gain comparison: {comparison_path}")


if __name__ == "__main__":
    main()
