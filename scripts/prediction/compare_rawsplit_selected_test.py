"""Select on rawsplit validation and evaluate all selected checkpoints on test.

All 15 selected runs are inferred afresh; historical test caches are not used.
The previous selection and evaluation algorithms are reused without modifying
their input or output paths on disk.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction import compare_dg95_selected_test as selection
from scripts.prediction import compare_retrained_mean_tradeoff_test as evaluation
from scripts.prediction.validate_hyperparameters import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'
VALID = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline_rawsplit'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23_full_80_10_10/config_manifest.json'
TRAIN = ROOT / 'results/pgin/retrain_lhs10_23_full_80_10_10'
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def select_models(plan_path):
    plan = load(plan_path)
    if set(plan['aliases']) != set(FAMILIES):
        raise ValueError('Invalid validation-only family selection')
    jobs = load(MANIFEST)
    selected = []
    for family in ('no_gain', *FAMILIES):
        alias = 'no_gain' if family == 'no_gain' else plan['aliases'][family]
        for seed in SEEDS:
            matches = [j for j in jobs if j['family'] == family and j['alias'] == alias and j['seed'] == seed]
            if len(matches) != 1:
                raise ValueError(f'Expected one matching model: {family} {seed} {alias}')
            j = matches[0]
            selected.append({'family': family, 'alias': alias, 'seed': seed,
                'role': ('fixed_reference' if seed == 42 else 'additional_reference_seed') if family == 'no_gain' else 'validation_selected',
                'validation_mean': None, 'model': f'{alias}_seed{seed}',
                **{k: j[k] for k in ('config', 'config_sha256', 'exp_name', 'checkpoint', 'results', 'cli_args')}})
    if len(selected) != 15 or len({r['model'] for r in selected}) != 15:
        raise ValueError('Expected 15 unique selected models')
    record = {'selection_rule': plan['selection_rule'], 'selection_plan': plan,
              'selection_plan_sha256': sha(plan_path),
              'validation_baseline': 'No-Gain seed 42 matched by case, target and evaluation',
              'selected_models': selected}
    path = OUT / 'selected_models.json'
    if path.exists() and load(path) != record:
        raise ValueError('Frozen model selection has changed')
    if not path.exists():
        write_json(path, record)
    return selected


def preflight(selected, plan_path):
    plan = load(plan_path)
    for name, digest in plan['validation_sources_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Validation input changed: {name}')
    inputs = [Path(__file__), MANIFEST, ROOT / 'scripts/prediction/compare_retrained_mean_tradeoff_test.py',
              ROOT / 'scripts/prediction/compare_dg95_selected_test.py',
              ROOT / 'scripts/evaluation/evaluate_gain_in_out_distribution.py',
              ROOT / 'scripts/prediction/validate_hyperparameters.py',
              ROOT / 'scripts/prediction/validate_gain_ss_hyperparameters.py',
              ROOT / 'scripts/prediction/predict.py', *list((ROOT / 'src').rglob('*.py')),
              *[evaluation.case_path(c) for c in evaluation.CASES], *evaluation.PROBES.values(),
              ROOT / 'saved_models/Tabular_MLP_5in_2out_QV.pth',
              ROOT / 'results/steady_state/Tabular_MLP_New/zscore_mean.csv',
              ROOT / 'results/steady_state/Tabular_MLP_New/zscore_std.csv']
    configs = {}
    for seed in SEEDS:
        batch_path = TRAIN / f'seed{seed}' / '_batch_run.json'
        batch = load(batch_path)
        if batch.get('status') != 'complete' or len(batch.get('completed', [])) != 21:
            raise ValueError(f'Incomplete training batch: seed {seed}')
        inputs.append(batch_path)
        for path, digest in batch['sources_sha256'].items():
            if sha(Path(path)).lower() != digest.lower():
                raise ValueError(f'Training input differs: {path}')
        for row in (r for r in selected if r['seed'] == seed):
            config_path = ROOT / row['config']
            cfg = yaml.safe_load(config_path.read_text(encoding='utf8'))
            if sha(config_path) != row['config_sha256'] or cfg['exp_name'] != row['exp_name'] or cfg['training']['seed'] != seed:
                raise ValueError(f'Invalid selected configuration: {row["model"]}')
            entries = [e for e in batch['completed'] if e['job']['exp_name'] == row['exp_name']]
            if len(entries) != 1:
                raise ValueError(f'No unique training entry: {row["model"]}')
            outputs = entries[0]['outputs_sha256']
            paths = [ROOT / row['checkpoint'], ROOT / row['results'] / 'zscore_mean.csv', ROOT / row['results'] / 'zscore_std.csv']
            for p in paths:
                if not p.is_file() or sha(p).lower() != outputs.get(str(p.relative_to(ROOT)), '').lower():
                    raise ValueError(f'Training artifact missing or changed: {p}')
            inputs += [config_path, *paths]
            configs[row['model']] = cfg
    contract = native.model_contract(configs['no_gain_seed42'])
    for name, cfg in configs.items():
        native.assert_same_contract(contract, native.model_contract(cfg), name)
    native.assert_same_scaling([(r['config'], configs[r['model']]) for r in selected], contract[0]+contract[1]+contract[2])
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    source_hashes.update(plan['validation_sources_sha256'])
    protocol = {'selection_sha256': sha(OUT/'selected_models.json'), 'selection_plan_sha256': sha(plan_path),
        'sources_sha256': source_hashes, 'selected_models': selected,
        'internal_test': evaluation.INTERNAL, 'external_test': evaluation.EXTERNAL,
        'steady_test': {k: str(v.relative_to(ROOT)) for k,v in evaluation.PROBES.items()},
        'preprocessing': '10-row median per case; saved training scaler; 18 historical steps',
        'baseline': 'No-Gain seed42 test RMSE, matched partition/case/target/evaluation',
        'KCI': 'ANN-reference micro pooled all18; |teacher gain| >= 1e-5; +/- 0.5 dynamic std; zero dynamic gain incorrect',
        'interpretation': 'Previously used test trajectories and probes, not a new blind holdout; recorded future decoder inputs',
        'seed_scope': 'Shared data and ANN; dynamic-training variability only',
        'reused_models': 0, 'newly_evaluated_models': 15, 'optimizer_updates': 0}
    path = OUT / 'run_protocol.json'
    if path.exists() and load(path) != protocol:
        raise ValueError('Inputs changed during test run')
    if not path.exists():
        write_json(path, protocol)
    return configs, contract, protocol


def main():
    os.chdir(ROOT)
    selection.VALID = VALID
    plan, candidates = selection.select_validation()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'validation_selection_plan.json'
    if path.exists() and load(path) != plan:
        raise ValueError('Frozen validation selection changed')
    if not path.exists():
        write_json(path, plan)
    with (OUT / 'validation_selection_candidates.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print('Validation selection:', plan['aliases'], flush=True)
    evaluation.OUT = OUT
    evaluation.selection_before_test = select_models
    evaluation.preflight = preflight
    evaluation.run(path)
    audit = OUT / 'independent_npz_audit.json'
    audit_cmd = [sys.executable, '-B', '-m', 'scripts.analysis.audit_retrained_mean_tradeoff_test_npz',
                 '--directory', str(OUT)]
    if not audit.exists():
        audit_cmd += ['--report', str(audit)]
    subprocess.run(audit_cmd, cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-B', '-m', 'scripts.analysis.generate_rawsplit_test_figures'],
                   cwd=ROOT, check=True)
    lines = [
        '# Raw-split validation-selected test comparison', '',
        'Selection was frozen from the raw-split 63-model validation results before running test inference.',
        'Per family, the three-seed means of the combined regional AvgRelRMSE and KCI are normalized',
        'within that family; the setting closest to lowest error and highest KCI is selected.', '',
        'Each selected setting has seeds 42, 43, and 44; No-Gain also has all three seeds.',
        'All 15 models were evaluated from their new checkpoints without reusing previous predictions.',
        'AvgRelRMSE uses the matching test case, output and horizon of the new No-Gain seed 42 as baseline.',
        'KCI is directional agreement with the shared ANN teacher on steady-state test probes.',
        'The dynamic test cases and steady-state probes were previously used; this is not a new blind test.',
        'Dynamic inference uses recorded future decoder inputs. Seeds quantify dynamic-training variation only.', '',
        'Selected validation aliases:', '',
        *[f'- {family}: {alias}' for family, alias in plan['aliases'].items()], '',
        'The interactive HTML and PNG/PDF summary are under `three_seed_report/`;',
        'the journal-ready figure and separate caption are under `jtice_figures/`.',
        'Use `independent_npz_audit.json` for the independent recomputation of output metrics.',
    ]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
