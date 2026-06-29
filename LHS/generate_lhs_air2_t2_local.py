import os
import re
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc


SOURCE_CSV = "data/Claus_dynamic/step_change/in_training_distribution/air2_190_t2_155_t2_change_-5.csv"
OUTPUT_PERTURB_FILENAME = "LHS/LHS_air2_190_t2_155_air2_t2_only_perturbation.csv"


def main():
    today_str = date.today().strftime("%Y%m%d")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    n_base = 480
    target_rows = 5000
    

    # Local region around the selected step-change condition.
    # air2_190_t2_155_air2_change_-5.csv covers air2 190 -> 185.
    # air2_190_t2_155_t2_change_-5.csv covers T2 155 -> 150.
    variable_ranges = {
        "B33.SPo.SPo": (140.0, 300.0),  # air2_SP
        "B20.SPo.SPo": (140.0, 240.0),  # HEATER2_output_T_SP
        "B34.SPo.SPo": (120.5, 160.5),  # acidgas_Fm, small local change around 140.5 kmol/hr
        "ACIDGAS.T.T": (82.5, 84.5),  # acidgas_T, small local change around 83.6 C
        "ACIDGAS.P.P": (1.65, 1.7),  # acidgas_P, small local change around 1.6722 bar
    }

    # Perturbation size for local gain cases. Keep this small enough to stay local,
    # but not so small that Aspen convergence tolerance dominates the response.
    perturb_delta = {
        "B33.SPo.SPo": 2.5,
        "B20.SPo.SPo": 2.5,
        "B34.SPo.SPo": 0.5,
        "ACIDGAS.T.T": 0.2,
        "ACIDGAS.P.P": 0.005,
    }

    mapping = {
        "B33.SPo.SPo": "air2_SP",
        "B20.SPo.SPo": "HEATER2_output_T_SP",
        "B34.SPo.SPo": "acidgas_Fm",
        "ACIDGAS.T.T": "acidgas_T",
        "ACIDGAS.P.P": "acidgas_P",
        'ACIDGAS.Fcn.CO2.("CO2")': "acidgas_CO2",
        'ACIDGAS.Fcn.H2O.("H2O")': "acidgas_H2O",
        'ACIDGAS.Fcn.H2S.("H2S")': "acidgas_H2S",
        # "B17.SPo.SPo": "air_SP",
        # "B35.SPo.SPo": "COG_SP",
        "B18.SPo.SPo": "burner_input_T_SP",
        "B19.SPo.SPo": "burner_output_T_SP",
        "BURNER_PC.SPo.SPo": "burner_output_P_SP",
        "FURANCE_PC.SPo.SPo": "fur_outputP_SP",
        "CAT1_PC.SPo.SPo": "cat1_output_P_SP",
        "CAT2_PC.SPo.SPo": "cat2_output_P_SP",
        "SEP2_PC.SPo.SPo": "SEP2_P_SP",
        "B21.SPo.SPo": "HEATER1_output_T_SP",
        "SEP1_PC.SPo.SPo": "SEP1_P_SP",
        "SEP3_PC.SPo.SPo": "SEP3_P_SP",
    }

    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    df_raw = pd.read_csv(SOURCE_CSV)
    final_tags = list(mapping.keys())

    fixed_values = {}
    for out_tag, source_col in mapping.items():
        if source_col not in df_raw.columns:
            raise ValueError(f"Missing source column {source_col} for output tag {out_tag}")
        data = pd.to_numeric(df_raw[source_col], errors="coerce").dropna()
        if len(data) == 0:
            raise ValueError(f"Source column has no numeric values: {source_col}")
        fixed_values[out_tag] = float(data.median())

    lhs_tags = []
    fixed_range_tags = []
    for tag, (low, high) in variable_ranges.items():
        if low < high:
            lhs_tags.append(tag)
        elif low == high:
            fixed_range_tags.append(tag)
        else:
            raise ValueError(f"Invalid range for {tag}: lower bound {low} > upper bound {high}")

    if not lhs_tags:
        raise ValueError("At least one variable range must have lower bound < upper bound for LHS.")

    df_base = pd.DataFrame([fixed_values] * n_base)
    sampler = qmc.LatinHypercube(d=len(lhs_tags))
    sample = sampler.random(n=n_base)
    bounds_low = [variable_ranges[tag][0] for tag in lhs_tags]
    bounds_high = [variable_ranges[tag][1] for tag in lhs_tags]
    scaled = qmc.scale(sample, bounds_low, bounds_high)

    for tag in fixed_range_tags:
        df_base[tag] = variable_ranges[tag][0]
    for idx, tag in enumerate(lhs_tags):
        df_base[tag] = scaled[:, idx]
    df_base = df_base[final_tags]

    perturb_rows = []
    for base_id, row in df_base.iterrows():
        base_case = row.copy()
        base_case["case_id"] = f"base_{base_id:05d}"
        base_case["base_id"] = base_id
        base_case["case_type"] = "base"
        base_case["perturbed_mv"] = ""
        base_case["direction"] = 0
        base_case["delta"] = 0.0
        perturb_rows.append(base_case)

        for tag in lhs_tags:
            low, high = variable_ranges[tag]
            delta = perturb_delta[tag]
            for direction in (1, -1):
                case = row.copy()
                old_value = float(case[tag])
                new_value = min(max(old_value + direction * delta, low), high)
                actual_delta = new_value - old_value
                if abs(actual_delta) < 1e-12:
                    continue
                case[tag] = new_value
                sign_name = "plus" if direction > 0 else "minus"
                case["case_id"] = f"base_{base_id:05d}_{tag}_{sign_name}"
                case["base_id"] = base_id
                case["case_type"] = "perturb"
                case["perturbed_mv"] = tag
                case["direction"] = direction
                case["delta"] = actual_delta
                perturb_rows.append(case)

    df_perturb = pd.DataFrame(perturb_rows)
    metadata_cols = ["case_id", "base_id", "case_type", "perturbed_mv", "direction", "delta"]
    df_perturb = df_perturb[final_tags + metadata_cols]
    generated_rows = len(df_perturb)
    if target_rows is not None and generated_rows != target_rows:
        base_mask = df_perturb["case_type"] == "base"
        df_base_cases = df_perturb[base_mask]
        df_perturb_cases = df_perturb[~base_mask]
        needed_perturb_rows = target_rows - len(df_base_cases)
        if needed_perturb_rows < 0:
            raise ValueError(
                f"target_rows={target_rows} is smaller than base rows={len(df_base_cases)}"
            )
        if needed_perturb_rows > len(df_perturb_cases):
            raise ValueError(
                f"Only generated {generated_rows} rows, fewer than target_rows={target_rows}. "
                "Increase n_base or widen the variable ranges."
            )
        df_perturb_cases = df_perturb_cases.sample(n=needed_perturb_rows).sort_index()
        df_perturb = (
            pd.concat([df_base_cases, df_perturb_cases])
            .sort_index()
            .reset_index(drop=True)
        )
    df_perturb.to_csv(OUTPUT_PERTURB_FILENAME, index=False)

    plot_dir = os.path.join(base_dir, f"LHS_Air2_T2_Local_Check_Plots_{today_str}")
    os.makedirs(plot_dir, exist_ok=True)
    for tag in final_tags:
        plt.figure(figsize=(8, 5))
        plt.hist(df_base[tag], bins=50, color="skyblue", edgecolor="black")
        status = "LHS" if tag in lhs_tags else "Fixed"
        plt.title(f"{tag} ({status})\nRange: [{df_base[tag].min():.6g}, {df_base[tag].max():.6g}]")
        plt.xlabel(tag)
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', tag)
        plt.savefig(os.path.join(plot_dir, f"Check_{safe_name}.png"), dpi=200)
        plt.close()

    print("Done.")
    if generated_rows != len(df_perturb):
        print(f"Generated rows before target trim: {generated_rows}")
    print(f"Perturbation rows: {len(df_perturb)} -> {OUTPUT_PERTURB_FILENAME}")
    print(f"Varied tags: {', '.join(lhs_tags)}")
    print(f"Check plots: {plot_dir}")


if __name__ == "__main__":
    main()
