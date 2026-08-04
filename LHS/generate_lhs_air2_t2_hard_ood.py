import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc


SOURCE_CSV = "data/Claus_dynamic/step_change/in_training_distribution/air2_190_t2_155_t2_change_-5.csv"
OUTPUT_PERTURB_FILENAME = "LHS/LHS_air2_t2_hard_ood_perturbation.csv"

AIR2_TAG = "B33.SPo.SPo"
T2_TAG = "B20.SPo.SPo"
ACIDGAS_F_TAG = "B34.SPo.SPo"
ACIDGAS_T_TAG = "ACIDGAS.T.T"
ACIDGAS_P_TAG = "ACIDGAS.P.P"

# lhs_generated_dynamic_ss_data_19.xlsx / soft-OOD envelope:
# air2=[110, 340], T2=[125, 250].
# Leave a gap so every generated hard-OOD case is strictly outside it.
SOFT_AIR2_RANGE = (110.0, 340.0)
SOFT_T2_RANGE = (125.0, 250.0)
SAFE_INNER_AIR2_RANGE = (145.0, 295.0)
SAFE_INNER_T2_RANGE = (145.0, 235.0)
AIR2_LOW_RANGE = (80.0, 105.0)
AIR2_HIGH_RANGE = (345.0, 380.0)
T2_LOW_RANGE = (105.0, 120.0)
T2_HIGH_RANGE = (255.0, 270.0)

# Keep the remaining three manipulated variables in the same safe range used
# by the soft-OOD generator, so the hard extrapolation is attributable to
# air2 and/or T2.
SAFE_ACIDGAS_RANGES = {
    ACIDGAS_F_TAG: (125.0, 155.0),
    ACIDGAS_T_TAG: (82.8, 84.4),
    ACIDGAS_P_TAG: (1.66, 1.69),
}

PERTURB_DELTA = {
    AIR2_TAG: 2.5,
    T2_TAG: 2.5,
    ACIDGAS_F_TAG: 0.5,
    ACIDGAS_T_TAG: 0.2,
    ACIDGAS_P_TAG: 0.005,
}

MAPPING = {
    AIR2_TAG: "air2_SP",
    T2_TAG: "HEATER2_output_T_SP",
    ACIDGAS_F_TAG: "acidgas_Fm",
    ACIDGAS_T_TAG: "acidgas_T",
    ACIDGAS_P_TAG: "acidgas_P",
    'ACIDGAS.Fcn.CO2.("CO2")': "acidgas_CO2",
    'ACIDGAS.Fcn.H2O.("H2O")': "acidgas_H2O",
    'ACIDGAS.Fcn.H2S.("H2S")': "acidgas_H2S",
    "B18.SPo.SPo": "burner_input_T_SP",
    "B19.SPo.SPo": "burner_output_T_SP",
    "BURNER_PC.SPo.SPo": "burner_output_P_SP",
    "FURANCE_PC.SPo.SPo": "fur_outputP_SP",
    "CAT1_PC.SPo.SPo": "cat1_output_P_SP",
    "CAT2_PC.SPo.SPo": "cat2_output_P_SP",
    "SEP2_PC.SPo.SPo": "SEP2_P_SP",
    "SEP1_PC.SPo.SPo": "SEP1_P_SP",
    "SEP3_PC.SPo.SPo": "SEP3_P_SP",
}


def make_strata():
    """Return 400 single-hard-OOD bases and 100 joint-hard-OOD bases."""

    single_specs = [
        ("single_air2_low", 100, AIR2_LOW_RANGE, SAFE_INNER_T2_RANGE, 1000),
        ("single_air2_high", 100, AIR2_HIGH_RANGE, SAFE_INNER_T2_RANGE, 1000),
        ("single_t2_low", 100, SAFE_INNER_AIR2_RANGE, T2_LOW_RANGE, 1000),
        ("single_t2_high", 100, SAFE_INNER_AIR2_RANGE, T2_HIGH_RANGE, 1000),
    ]
    joint_specs = [
        ("joint_air2_low_t2_low", 25, AIR2_LOW_RANGE, T2_LOW_RANGE, 250),
        ("joint_air2_low_t2_high", 25, AIR2_LOW_RANGE, T2_HIGH_RANGE, 250),
        ("joint_air2_high_t2_low", 25, AIR2_HIGH_RANGE, T2_LOW_RANGE, 250),
        ("joint_air2_high_t2_high", 25, AIR2_HIGH_RANGE, T2_HIGH_RANGE, 250),
    ]
    return single_specs + joint_specs


def sample_base_rows(fixed_values, final_tags, start_base_id, spec):
    region_name, n_base, air2_range, t2_range, target_rows = spec
    ranges = {
        AIR2_TAG: air2_range,
        T2_TAG: t2_range,
        **SAFE_ACIDGAS_RANGES,
    }
    lhs_tags = list(ranges)
    sampler = qmc.LatinHypercube(d=len(lhs_tags))
    sample = sampler.random(n=n_base)
    scaled = qmc.scale(
        sample,
        [ranges[tag][0] for tag in lhs_tags],
        [ranges[tag][1] for tag in lhs_tags],
    )

    df_base = pd.DataFrame([fixed_values] * n_base)
    for idx, tag in enumerate(lhs_tags):
        df_base[tag] = scaled[:, idx]
    df_base = df_base[final_tags]
    df_base.index = range(start_base_id, start_base_id + n_base)
    return region_name, target_rows, ranges, df_base


def expand_and_trim_region(region_name, target_rows, ranges, df_base):
    rows = []
    for base_id, row in df_base.iterrows():
        base_case = row.copy()
        base_case["case_id"] = f"{region_name}_base_{base_id:05d}"
        base_case["base_id"] = base_id
        base_case["case_type"] = "base"
        base_case["perturbed_mv"] = ""
        base_case["direction"] = 0
        base_case["delta"] = 0.0
        rows.append(base_case)

        for tag, (low, high) in ranges.items():
            delta = PERTURB_DELTA[tag]
            for direction in (1, -1):
                case = row.copy()
                old_value = float(case[tag])
                new_value = min(max(old_value + direction * delta, low), high)
                actual_delta = new_value - old_value
                if abs(actual_delta) < 1e-12:
                    continue
                case[tag] = new_value
                sign_name = "plus" if direction > 0 else "minus"
                case["case_id"] = (
                    f"{region_name}_base_{base_id:05d}_{tag}_{sign_name}"
                )
                case["base_id"] = base_id
                case["case_type"] = "perturb"
                case["perturbed_mv"] = tag
                case["direction"] = direction
                case["delta"] = actual_delta
                rows.append(case)

    df_region = pd.DataFrame(rows)
    base_mask = df_region["case_type"] == "base"
    df_base_cases = df_region[base_mask]
    df_perturb_cases = df_region[~base_mask]
    needed_perturb_rows = target_rows - len(df_base_cases)
    if needed_perturb_rows > len(df_perturb_cases):
        raise ValueError(
            f"{region_name} generated only {len(df_region)} rows, "
            f"fewer than target_rows={target_rows}."
        )
    df_perturb_cases = df_perturb_cases.sample(
        n=needed_perturb_rows
    ).sort_index()
    return pd.concat([df_base_cases, df_perturb_cases]).sort_index()


def validate_hard_ood(df_output):
    if len(df_output) != 5000:
        raise ValueError(f"Expected exactly 5000 rows, got {len(df_output)}.")

    air2_hard = (
        (df_output[AIR2_TAG] < SOFT_AIR2_RANGE[0])
        | (df_output[AIR2_TAG] > SOFT_AIR2_RANGE[1])
    )
    t2_hard = (
        (df_output[T2_TAG] < SOFT_T2_RANGE[0])
        | (df_output[T2_TAG] > SOFT_T2_RANGE[1])
    )
    if not (air2_hard | t2_hard).all():
        raise ValueError(
            "At least one generated row was not strictly outside the "
            "soft-OOD air2/T2 envelope."
        )

    single_count = int((air2_hard ^ t2_hard).sum())
    joint_count = int((air2_hard & t2_hard).sum())
    if single_count != 4000 or joint_count != 1000:
        raise ValueError(
            "Unexpected hard-OOD composition: "
            f"single={single_count}, joint={joint_count}."
        )

    for tag, (low, high) in SAFE_ACIDGAS_RANGES.items():
        if not df_output[tag].between(low, high, inclusive="both").all():
            raise ValueError(f"{tag} exceeded its safe range [{low}, {high}].")
    return single_count, joint_count


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    df_raw = pd.read_csv(SOURCE_CSV)
    final_tags = list(MAPPING)
    fixed_values = {}
    for out_tag, source_col in MAPPING.items():
        if source_col not in df_raw.columns:
            raise ValueError(
                f"Missing source column {source_col} for output tag {out_tag}"
            )
        data = pd.to_numeric(df_raw[source_col], errors="coerce").dropna()
        if len(data) == 0:
            raise ValueError(f"Source column has no numeric values: {source_col}")
        fixed_values[out_tag] = float(data.median())

    base_frames = []
    output_frames = []
    next_base_id = 0
    for spec in make_strata():
        region_name, target_rows, ranges, df_base = sample_base_rows(
            fixed_values, final_tags, next_base_id, spec
        )
        next_base_id += len(df_base)
        base_frames.append(df_base)
        output_frames.append(
            expand_and_trim_region(
                region_name, target_rows, ranges, df_base
            )
        )

    df_base_all = pd.concat(base_frames).sort_index()
    df_output = pd.concat(output_frames).reset_index(drop=True)
    metadata_cols = [
        "case_id",
        "base_id",
        "case_type",
        "perturbed_mv",
        "direction",
        "delta",
    ]
    df_output = df_output[final_tags + metadata_cols]
    single_count, joint_count = validate_hard_ood(df_output)
    df_output.to_csv(OUTPUT_PERTURB_FILENAME, index=False)

    plot_dir = os.path.join(base_dir, "LHS_Air2_T2_Hard_OOD_Check_Plots")
    os.makedirs(plot_dir, exist_ok=True)
    for tag in final_tags:
        plt.figure(figsize=(8, 5))
        plt.hist(df_base_all[tag], bins=50, color="skyblue", edgecolor="black")
        plt.title(
            f"{tag}\n"
            f"Range: [{df_base_all[tag].min():.6g}, "
            f"{df_base_all[tag].max():.6g}]"
        )
        plt.xlabel(tag)
        plt.ylabel("Base-point count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", tag)
        plt.savefig(
            os.path.join(plot_dir, f"Check_{safe_name}.png"), dpi=200
        )
        plt.close()

    print("Done.")
    print(f"Hard-OOD rows: {len(df_output)} -> {OUTPUT_PERTURB_FILENAME}")
    print(f"Single-variable hard-OOD rows: {single_count}")
    print(f"Joint air2 x T2 hard-OOD rows: {joint_count}")
    print(f"Base operating points: {len(df_base_all)}")
    print(f"Check plots: {plot_dir}")


if __name__ == "__main__":
    main()
