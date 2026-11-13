"""Generate paired Aspen input conditions for independent ANN gain validation.

The CSV contains input conditions only, not solved Aspen concentrations. Each
base has one signed single-MV step of 0.5 dynamic-training standard deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/Claus_dynamic/step_change/in_training_distribution/air2_190_t2_155_t2_change_-5.csv"
DYNAMIC_STD = ROOT / "results/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42/zscore_std.csv"
AIR = "B33.SPo.SPo"
T2 = "B20.SPo.SPo"
FLOW = "B34.SPo.SPo"
TEMP = "ACIDGAS.T.T"
PRESSURE = "ACIDGAS.P.P"
MAPPING = {
    AIR: "air2_SP",
    T2: "HEATER2_output_T_SP",
    FLOW: "acidgas_Fm",
    TEMP: "acidgas_T",
    PRESSURE: "acidgas_P",
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
LHS_TAGS = [AIR, T2, FLOW, TEMP, PRESSURE]
ARMS = [(AIR, 1), (AIR, -1), (T2, 1), (T2, -1)]
STEP_STD_MULTIPLIER = 0.5
GLOBAL_MV_RANGES = {AIR: (110.0, 340.0), T2: (100.0, 280.0)}

# Core 50%, Soft-OOD 50%; within Soft-OOD, 80% single and 20% joint.
# The one-pair difference among joint strata is required by 1250 * 0.05.
STRATA = [
    ("core", "core", 1250, (140., 300.), (140., 240.)),
    ("soft_ood", "single_air2_low", 250, (110., 140.), (142.5, 237.5)),
    ("soft_ood", "single_air2_high", 250, (300., 340.), (142.5, 237.5)),
    ("soft_ood", "single_t2_low", 250, (142.5, 297.5), (100., 140.)),
    ("soft_ood", "single_t2_high", 250, (142.5, 297.5), (240., 280.)),
    ("soft_ood", "joint_air2_low_t2_low", 63, (110., 140.), (100., 140.)),
    ("soft_ood", "joint_air2_low_t2_high", 62, (110., 140.), (240., 280.)),
    ("soft_ood", "joint_air2_high_t2_low", 62, (300., 340.), (100., 140.)),
    ("soft_ood", "joint_air2_high_t2_high", 63, (300., 340.), (240., 280.)),
]
TARGETS = {"core": [313, 312, 313, 312], "soft_ood": [312, 313, 312, 313]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steps_from_scaler() -> dict[str, float]:
    std = pd.read_csv(DYNAMIC_STD, index_col=0).squeeze("columns")
    steps = {tag: STEP_STD_MULTIPLIER * float(std[MAPPING[tag]]) for tag in (AIR, T2)}
    if not all(np.isfinite(value) and value > 0 for value in steps.values()):
        raise ValueError("Dynamic scaler must provide finite positive MV standard deviations")
    return steps


def point_region(row: dict) -> str:
    return "core" if 140.0 < row[AIR] < 300.0 and 140.0 < row[T2] < 240.0 else "soft_ood"


def allocate_arms(rng: np.random.Generator) -> dict[str, list[int]]:
    """Balance the four MV/direction arms within each stratum and region."""
    remaining = {region: np.array(counts, dtype=int) for region, counts in TARGETS.items()}
    allocated = {}
    for region, name, count, _, _ in STRATA:
        n, extra = divmod(count, len(ARMS))
        counts = np.full(len(ARMS), n, dtype=int)
        remaining[region] -= counts
        # Distribute remainders where the region still needs the most cases.
        tie_break = rng.random(len(ARMS))
        order = sorted(range(len(ARMS)), key=lambda i: (-remaining[region][i], tie_break[i]))
        for i in order[:extra]:
            counts[i] += 1
            remaining[region][i] -= 1
        if np.any(remaining[region] < 0):
            raise ValueError(f"Cannot balance arms in stratum {name}")
        allocated[name] = counts.tolist()
    if any(np.any(counts != 0) for counts in remaining.values()):
        raise ValueError("Arm counts did not reach regional targets")
    return allocated


def fixed_values() -> dict[str, float]:
    raw = pd.read_csv(SOURCE)
    values = {}
    for tag, column in MAPPING.items():
        if column not in raw:
            raise ValueError(f"Missing source column: {column}")
        numeric = pd.to_numeric(raw[column], errors="coerce").dropna()
        if numeric.empty or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"No finite source values for {column}")
        values[tag] = float(numeric.median())
    return values


def generate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng()
    allocation = allocate_arms(rng)
    values = fixed_values()
    steps = steps_from_scaler()
    rows, manifest = [], []
    pair_id = 0
    strata_report = {}
    for region, name, _, air_range, t2_range in STRATA:
        bounds = {AIR: air_range, T2: t2_range,
                  FLOW: (120.5, 160.5) if region == "core" else (110.5, 160.5),
                  TEMP: (82.6, 84.6), PRESSURE: (1.66, 1.69)}
        strata_report[name] = {"base_region": region, "base_bounds": bounds, "arms": {}}
        for arm_index, (mv, direction) in enumerate(ARMS):
            n = allocation[name][arm_index]
            lower = np.array([bounds[tag][0] for tag in LHS_TAGS], dtype=float)
            upper = np.array([bounds[tag][1] for tag in LHS_TAGS], dtype=float)
            axis = LHS_TAGS.index(mv)
            # Preserve the original base stratum but permit crossing its boundary.
            # Restrict only the base interval needed to keep the full signed
            # step within the overall air2/T2 envelope, with no step clipping.
            global_low, global_high = GLOBAL_MV_RANGES[mv]
            if direction > 0:
                upper[axis] = min(upper[axis], global_high - steps[mv])
            else:
                lower[axis] = max(lower[axis], global_low + steps[mv])
            if np.any(lower >= upper):
                raise ValueError(f"Step too large for {name}: {mv}")
            samples = qmc.scale(qmc.LatinHypercube(d=5, rng=rng).random(n), lower, upper)
            arm_name = f"{MAPPING[mv]}_{'plus' if direction > 0 else 'minus'}"
            strata_report[name]["arms"][arm_name] = {
                "count": n,
                "effective_base_bounds": {tag: [float(lower[i]), float(upper[i])] for i, tag in enumerate(LHS_TAGS)},
            }
            for sample in samples:
                pair_id += 1
                base = values.copy()
                base.update(zip(LHS_TAGS, map(float, sample)))
                changed = base.copy()
                changed[mv] += direction * steps[mv]
                endpoint_in_base_stratum = all(bounds[tag][0] <= changed[tag] <= bounds[tag][1] for tag in LHS_TAGS)
                endpoint_region = point_region(changed)
                crosses_core_boundary = point_region(base) != endpoint_region
                for kind, row in (("base", base), ("perturb", changed)):
                    rows.append(row)
                    manifest.append({
                        "input_row_1based": len(rows), "case_id": f"pair_{pair_id:05d}_{kind}",
                        "base_id": pair_id, "case_type": kind, "region": region, "stratum": name,
                        "perturbed_mv": mv, "mv_name": MAPPING[mv], "direction": direction,
                        "delta": direction * steps[mv] if kind == "perturb" else 0.0,
                        "row_region": point_region(row), "endpoint_region": endpoint_region,
                        "crosses_core_boundary": crosses_core_boundary,
                        "endpoint_in_base_stratum": endpoint_in_base_stratum,
                    })
    inputs = pd.DataFrame(rows, columns=MAPPING.keys())
    meta = pd.DataFrame(manifest)
    validate(inputs, meta, steps)
    design = {
        "randomness": "Fresh system-entropy random generator; no fixed seed",
        "data_role": "Aspen input conditions; not solved Aspen outputs",
        "pairs": pair_id, "rows": len(inputs),
        "step_std_multiplier": STEP_STD_MULTIPLIER,
        "step_magnitudes": {MAPPING[tag]: magnitude for tag, magnitude in steps.items()},
        "dynamic_scaler_std": str(DYNAMIC_STD.relative_to(ROOT)), "dynamic_scaler_sha256": sha256(DYNAMIC_STD),
        "source_csv": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "fixed_values": {tag: values[tag] for tag in MAPPING if tag not in LHS_TAGS},
        "arm_definition": "one specified MV per base, one signed step; two rows per pair",
        "sampling": "5-dimensional random LHS separately within stratum and MV-direction arm",
        "boundary_rule": "base is sampled in the original stratum; endpoint may cross strata or core/soft boundary but remains within air2 110-340 and T2 100-280; direction-specific base bounds avoid clipping",
        "coverage_note": "Core base acidgas_Fm 120.5-160.5; Soft-OOD base 110.5-160.5; not the full enclosing five-dimensional rectangle. row_region and endpoint_region classify only air2/T2; an endpoint labelled core may have acidgas_Fm below the original core sampler minimum. The 50/50 and stratum weights are design choices, not volume weights. Base bounds near outer edges depend on direction; see arms. KCI rollout itself does not impose these endpoint limits.",
        "crosses_core_boundary_pairs": int(meta.loc[meta.case_type.eq("perturb"), "crosses_core_boundary"].sum()),
        "endpoint_outside_base_stratum_pairs": int((~meta.loc[meta.case_type.eq("perturb"), "endpoint_in_base_stratum"]).sum()),
        "core_endpoints_below_original_core_acidgas_flow_min": int(((meta.case_type.eq("perturb")) & (meta.row_region.eq("core")) & (inputs[FLOW] < 120.5)).sum()),
        "independence_note": "Fresh random input points; exact or near overlap with teacher training data has not been established by generation alone.",
        "strata": strata_report,
    }
    return inputs, meta, design


def validate(inputs: pd.DataFrame, meta: pd.DataFrame, steps: dict[str, float] | None = None) -> None:
    steps = steps_from_scaler() if steps is None else steps
    if len(inputs) != 5000 or len(meta) != 5000 or len(meta.base_id.unique()) != 2500:
        raise ValueError("Expected 5000 input rows and 2500 unique pairs")
    if list(inputs.columns) != list(MAPPING) or not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError("Unexpected Aspen columns or non-finite values")
    if meta.case_id.duplicated().any() or not np.array_equal(meta.input_row_1based, np.arange(1, 5001)):
        raise ValueError("Case IDs or row mapping are invalid")
    if meta.case_type.tolist() != [v for _ in range(2500) for v in ("base", "perturb")]:
        raise ValueError("Rows must be adjacent base/perturb pairs")
    if not np.array_equal(meta.base_id.to_numpy()[::2], meta.base_id.to_numpy()[1::2]):
        raise ValueError("Adjacent rows must have the same pair ID")
    base, changed = inputs.iloc[::2].reset_index(drop=True), inputs.iloc[1::2].reset_index(drop=True)
    pair_meta = meta.iloc[1::2].reset_index(drop=True)
    differences = changed.to_numpy() - base.to_numpy()
    expected = np.zeros_like(differences)
    for i, row in pair_meta.iterrows():
        expected[i, list(inputs.columns).index(row.perturbed_mv)] = row.direction * steps[row.perturbed_mv]
    if not np.allclose(differences, expected, rtol=0, atol=1e-12):
        raise ValueError("Pair changes more than its designated MV or has a truncated step")
    expected_deltas = np.array([row.direction * steps[row.perturbed_mv] for _, row in pair_meta.iterrows()])
    if not np.allclose(pair_meta.delta, expected_deltas, rtol=0, atol=1e-12):
        raise ValueError("Manifest delta does not match the step")
    if pair_meta.groupby(["mv_name", "direction"]).size().to_dict() != {
        ("air2_SP", -1): 625, ("air2_SP", 1): 625,
        ("HEATER2_output_T_SP", -1): 625, ("HEATER2_output_T_SP", 1): 625,
    }:
        raise ValueError("MV/direction arms are not balanced")
    for region, name, count, air_range, t2_range in STRATA:
        mask = meta.stratum.eq(name).to_numpy()
        if mask.sum() != 2 * count or not meta.loc[mask, "region"].eq(region).all():
            raise ValueError(f"Wrong size or region for {name}")
        selected = inputs.loc[mask & meta.case_type.eq("base").to_numpy()]
        flow_range = (120.5, 160.5) if region == "core" else (110.5, 160.5)
        for tag, (low, high) in {AIR: air_range, T2: t2_range, FLOW: flow_range,
                                 TEMP: (82.6, 84.6), PRESSURE: (1.66, 1.69)}.items():
            if not selected[tag].between(low, high).all():
                raise ValueError(f"{name} falls outside bounds for {tag}")
    for tag, (low, high) in GLOBAL_MV_RANGES.items():
        if not inputs[tag].between(low, high).all():
            raise ValueError(f"Aspen input falls outside overall bounds for {tag}")
    for i, row in meta.iterrows():
        actual = point_region(inputs.iloc[i].to_dict())
        if row.row_region != actual:
            raise ValueError(f"Incorrect actual row region at input row {i + 1}")
    strata_bounds = {name: (air_range, t2_range) for _, name, _, air_range, t2_range in STRATA}
    for pair_id, group in meta.groupby("base_id", sort=False):
        a = group[group.case_type.eq("base")].iloc[0]
        b = group[group.case_type.eq("perturb")].iloc[0]
        for key in ("region", "stratum", "perturbed_mv", "mv_name", "direction", "endpoint_region", "crosses_core_boundary", "endpoint_in_base_stratum"):
            if a[key] != b[key]:
                raise ValueError(f"Inconsistent pair metadata for {pair_id}: {key}")
        if a.perturbed_mv not in steps or MAPPING[a.perturbed_mv] != a.mv_name or a.delta != 0:
            raise ValueError(f"Incorrect MV metadata for pair {pair_id}")
        if a.row_region != a.region or b.endpoint_region != b.row_region:
            raise ValueError(f"Incorrect base or endpoint region for pair {pair_id}")
        if bool(b.crosses_core_boundary) != (a.row_region != b.row_region):
            raise ValueError(f"Incorrect cross-region flag for pair {pair_id}")
        endpoint = inputs.iloc[int(b.input_row_1based) - 1]
        air_range, t2_range = strata_bounds[b.stratum]
        inside = air_range[0] <= endpoint[AIR] <= air_range[1] and t2_range[0] <= endpoint[T2] <= t2_range[1]
        if bool(b.endpoint_in_base_stratum) != inside:
            raise ValueError(f"Incorrect original-stratum flag for pair {pair_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory; existing output files are never overwritten")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output = (args.output_dir or ROOT / f"data/sampling/ann_gain_validation_5000_large_step_{stamp}").resolve()
    input_path = output / "aspen_inputs.csv"
    manifest_path = output / "row_manifest.csv"
    report_path = output / "generation_report.json"
    if any(p.exists() for p in (input_path, manifest_path, report_path)):
        raise FileExistsError(f"Output exists; choose another --output-dir: {output}")
    inputs, meta, report = generate()
    output.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(input_path, index=False, float_format="%.17g")
    meta.to_csv(manifest_path, index=False)
    # Check the actual serialized files, including their row alignment.
    validate(pd.read_csv(input_path), pd.read_csv(manifest_path))
    report["verification"] = "passed: saved files reloaded; pair, arm, stratum, range, and signed step checks"
    report["output_sha256"] = {input_path.name: sha256(input_path), manifest_path.name: sha256(manifest_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(inputs)} Aspen input rows / {len(meta) // 2} pairs: {output}")
    print("Base core / Soft-OOD: 1250 / 1250 pairs; each MV and direction: 625 pairs")
    print(f"Step sizes: air2={report['step_magnitudes']['air2_SP']:.12g}; T2={report['step_magnitudes']['HEATER2_output_T_SP']:.12g}")
    print(f"Pairs crossing core/Soft-OOD boundary: {report['crosses_core_boundary_pairs']}")
    print("Aspen concentrations are not included; run the Aspen steady-state solver separately.")


if __name__ == "__main__":
    main()
