"""Render three completed regional evaluations with a single-seed selector."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.plotting.render_retrain_validation_html import PAGE, read_csv
from scripts.plotting.retrain_parameter_ui import render_parameter_page

DEFAULT_ROOT = ROOT / 'results/visualization'
SEEDS = (42, 43, 44)
REGIONS = {'internal': ('in_range', ('R5-3', 'R5-4', 'R5-5')),
           'external': ('soft_ood', ('R5-27', 'R5-28', 'R5-29'))}
TARGETS = ('B35_H2S', 'B35_SO2')
MVS = ('air2_SP', 'HEATER2_output_T_SP')
EVALUATIONS = ('H1', 'H18', 'TS')


def keyed(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in out:
            raise ValueError(f'Duplicate row: {key}')
        out[key] = row
    return out


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-14)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_seed(directory: Path, seed: int, manifest: list[dict]) -> tuple[dict, dict]:
    result = json.loads((directory/'evaluation_validation.json').read_text(encoding='utf-8-sig'))
    if not (result.get('passed') is True and result.get('seed') == seed
            and result.get('model_count') == 21 and result.get('per_region_points') == 21
            and result.get('source_hashes_unchanged') is True):
        raise ValueError(f'Seed {seed} evaluation is incomplete or not validated.')
    protocol = json.loads((directory/'run_protocol.json').read_text(encoding='utf-8-sig'))
    jobs = [j for j in manifest if j['seed'] == seed]
    models = {j['alias']: j['family'] for j in jobs}
    if len(jobs) != 21 or len(models) != 21 or protocol['jobs'] != jobs:
        raise ValueError(f'Seed {seed} manifest differs from evaluation protocol.')
    if (protocol['internal_cases'] != list(REGIONS['internal'][1])
            or protocol['external_cases'] != list(REGIONS['external'][1])):
        raise ValueError(f'Seed {seed} uses unexpected dynamic cases.')
    coordinates = keyed(read_csv(directory/'regional_plot_coordinates.csv'), ('model','partition'))
    metrics = keyed(read_csv(directory/'validation_metrics.csv'), ('model','partition','case','target','evaluation'))
    relative = keyed(read_csv(directory/'relative_rmse_components.csv'), ('model','partition','case','target','evaluation'))
    components = keyed(read_csv(directory/'kci_components.csv'), ('model','distribution','target','mv'))
    expected_coords = {(m, p) for m in models for p in REGIONS}
    expected_metrics = {(m, p, c, t, e) for m in models for p, (_, cases) in REGIONS.items()
                        for c in cases for t in TARGETS for e in EVALUATIONS}
    expected_kci = {(m, dist, target, mv) for m in models for dist, _ in REGIONS.values()
                    for target in TARGETS for mv in MVS}
    if set(coordinates) != expected_coords or set(metrics) != expected_metrics or set(relative) != expected_metrics or set(components) != expected_kci:
        raise ValueError(f'Seed {seed} has missing, extra or duplicate coordinate/component rows.')
    protocol_sha = sha(directory/'run_protocol.json')
    for model in models:
        model_dir = directory/'model_data'/model
        completed = json.loads((model_dir/'complete.json').read_text(encoding='utf-8-sig'))
        expected_files = {'validation_metrics.csv','kci_components.csv','gain_in_range.npz','gain_soft_ood.npz',
                          *[f'{case}_prediction.npz' for _,cases in REGIONS.values() for case in cases]}
        if completed['run_protocol_sha256'] != protocol_sha or not expected_files.issubset(completed['outputs']):
            raise ValueError(f'Seed {seed}: incomplete model output record: {model}')
        for name, digest in completed['outputs'].items():
            if sha(model_dir/name).lower() != digest.lower():
                raise ValueError(f'Seed {seed}: model output hash mismatch: {model}/{name}')
        model_metrics = keyed(read_csv(model_dir/'validation_metrics.csv'), ('model','partition','case','target','evaluation'))
        model_kci = keyed(read_csv(model_dir/'kci_components.csv'), ('model','distribution','target','mv'))
        if set(model_metrics) != {key for key in expected_metrics if key[0] == model} or set(model_kci) != {key for key in expected_kci if key[0] == model}:
            raise ValueError(f'Seed {seed}: per-model component coverage mismatch: {model}')
        for key, row in model_metrics.items():
            if not (int(row['n']) == int(metrics[key]['n']) and close(row['RMSE'],metrics[key]['RMSE'])):
                raise ValueError(f'Seed {seed}: aggregate metric differs from model file: {key}')
        for key, row in model_kci.items():
            for column in ('probe_points','possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                if int(row[column]) != int(components[key][column]):
                    raise ValueError(f'Seed {seed}: aggregate KCI differs from model file: {key}, {column}')
    points, details = [], []
    for model, partition in sorted(expected_coords):
        row = coordinates[(model, partition)]
        if row['family'] != models[model]:
            raise ValueError(f'Seed {seed}: wrong family for {model}')
        ratios = []
        for case in REGIONS[partition][1]:
            for target in TARGETS:
                for ev in EVALUATIONS:
                    key = (model, partition, case, target, ev)
                    a, b = metrics[key], metrics[('no_gain', partition, case, target, ev)]
                    r = relative[key]
                    value, baseline, ratio = float(a['RMSE']), float(b['RMSE']), float(r['relative_RMSE'])
                    if not all(math.isfinite(x) and x > 0 for x in (value, baseline, ratio)):
                        raise ValueError(f'Seed {seed}: nonpositive or nonfinite RMSE: {key}')
                    if not (int(a['n']) == int(b['n']) == int(r['baseline_n']) == int(r['n'])
                            and close(value, r['RMSE']) and close(baseline, r['baseline_RMSE'])
                            and close(ratio, value/baseline)):
                        raise ValueError(f'Seed {seed}: same-seed baseline mismatch: {key}')
                    ratios.append(ratio)
        y = float(row['AvgRelRMSE'])
        if not close(y, math.exp(sum(math.log(x) for x in ratios)/18)) or int(row['components']) != 18:
            raise ValueError(f'Seed {seed}: RMSE coordinate mismatch: {model}, {partition}')
        correct = valid = 0
        distribution = REGIONS[partition][0]
        for target in TARGETS:
            for mv in MVS:
                r = components[(model, distribution, target, mv)]
                tp, tn, fp, fn, zero = (int(r[x]) for x in ('tp','tn','fp','fn','zero_gain'))
                v, possible = int(r['valid_comparisons']), int(r['possible_comparisons'])
                probes = 4404 if distribution == 'in_range' else 2585
                if (int(r['probe_points']) != probes or possible != probes * 2 * 18
                        or min(tp,tn,fp,fn,zero) < 0 or v <= 0 or v > possible
                        or tp+tn+fp+fn+zero != v):
                    raise ValueError(f'Seed {seed}: invalid confusion counts.')
                correct += tp+tn; valid += v
                details.append(dict(model=model, partition=partition, mv=mv, target=target, correct=tp+tn, valid=v))
        x = float(row['ANN_reference_KCI_percent'])
        if not (int(row['correct']) == correct and int(row['valid']) == valid and close(x,100*correct/valid)):
            raise ValueError(f'Seed {seed}: KCI coordinate mismatch: {model}, {partition}')
        points.append(dict(model=model, family=models[model], partition=partition, x=x, y=y, correct=correct, valid=valid))
    for partition in REGIONS:
        if not close(coordinates[('no_gain',partition)]['AvgRelRMSE'],1):
            raise ValueError(f'Seed {seed}: No-Gain baseline must equal one.')
    return dict(points=points, details=details), protocol


def build_page(datasets):
    return render_parameter_page(PAGE, datasets, ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT/'retrain_lhs10_23_63_models_validation/regional_validation_by_seed.html')
    args = p.parse_args()
    manifest = json.loads((ROOT/'configs/pgin/retrain_lhs10_23/config_manifest.json').read_text(encoding='utf-8'))
    datasets, protocols = {}, {}
    for seed in SEEDS:
        directory = args.input_root/f'retrain_lhs10_23_21_models_validation_seed{seed}'
        datasets[str(seed)], protocols[seed] = load_seed(directory, seed, manifest)
    verified_sources = {}
    for seed in SEEDS:
        for name, digest in protocols[seed]['sources_sha256'].items():
            path = ROOT/name
            if path not in verified_sources:
                verified_sources[path] = sha(path)
            if verified_sources[path].lower() != digest.lower():
                raise ValueError(f'Seed {seed}: protected source changed since evaluation: {name}')
    # The same test cases, probe files, and ANN teacher must underlie every seed.
    common = ['saved_models/Tabular_MLP_5in_2out_QV.pth', 'results/steady_state/Tabular_MLP_New/zscore_mean.csv', 'results/steady_state/Tabular_MLP_New/zscore_std.csv',
              *[f'data/Claus_steady_state/lhs_valid_{dist}.xlsx' for dist, _ in REGIONS.values()],
              *[f'data/Claus_dynamic/Test_dataform_change_air2_R=5-{case.split("-")[1]}.{ "xlsx" if part == "internal" else "csv"}' for part, (_, cases) in REGIONS.items() for case in cases]]
    for key in common:
        hashes = [{k.replace('\\','/'):v for k,v in protocols[s]['sources_sha256'].items()}[key] for s in SEEDS]
        if len(set(hashes)) != 1:
            raise ValueError(f'Cross-seed shared source differs: {key}')
    denominators = {}
    for seed in SEEDS:
        for row in datasets[str(seed)]['details']:
            key = (row['partition'], row['target'], row['mv'])
            if key in denominators and denominators[key] != row['valid']:
                raise ValueError(f'KCI valid-count mismatch for {key}')
            denominators[key] = row['valid']
    csv_output = args.output.parent/'regional_coordinates_63_models.csv'
    for path in (args.output, csv_output):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(build_page(datasets),encoding='utf-8')
    with csv_output.open('w',encoding='utf-8',newline='') as stream:
        columns = ['seed','model','family','region','KCI_percent','AvgRelRMSE','correct','valid']
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for seed in SEEDS:
            for point in datasets[str(seed)]['points']:
                writer.writerow(dict(seed=seed,model=point['model'],family=point['family'],region=point['partition'],
                                     KCI_percent=point['x'],AvgRelRMSE=point['y'],correct=point['correct'],valid=point['valid']))
    print(f'Wrote {args.output}: three verified seeds, each with 21 models per region.')
    print(f'Wrote {csv_output}: 126 rows; no cross-seed aggregation.')


if __name__ == '__main__':
    main()
