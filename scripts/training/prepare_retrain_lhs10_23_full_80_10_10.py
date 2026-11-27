"""Prepare isolated full-trajectory 80/10/10 training configurations."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'configs/pgin/retrain_lhs10_23'
NEW = ROOT / 'configs/pgin/retrain_lhs10_23_full_80_10_10'
EXPERIMENT = 'lhs10_23_full_80_10_10'
SEEDS = (42, 43, 44)
FAMILIES = {'no_gain': 1, 'gain': 5, 'ss_only': 5, 'gain_ss': 5, 'difficult_gradnorm': 5}
FILES = [f'Test_dataform_change_air2_R={case}.csv' for case in ('5', '5-1', '5-2', '5-6')]
DG_SUFFIX = '_difficult_group_gradnorm_no_ss'
MODULES = {
    'scripts.training.train_pgin_stepwise_gain': 'scripts.training.train_pgin_stepwise_gain_rawsplit',
    'scripts.training.train_pgin_stepwise_gain_ss_loss': 'scripts.training.train_pgin_stepwise_gain_ss_loss_rawsplit',
    'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm': 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm_rawsplit',
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare() -> None:
    original = json.loads((OLD / 'config_manifest.json').read_text(encoding='utf-8'))
    expanded = json.loads((OLD / 'config_manifest_dg_expanded.json').read_text(encoding='utf-8'))
    selected = [j for j in original if j['family'] != 'difficult_gradnorm' or j['alias'] == 'dg_ratio0p015'] + expanded
    expected = {(seed, family): count for seed in SEEDS for family, count in FAMILIES.items()}
    if len(selected) != 63 or Counter((j['seed'], j['family']) for j in selected) != expected:
        raise ValueError('The active source manifests must contain the expected 63 models')
    prepared = []
    for job in selected:
        seed = job['seed']
        source = ROOT / job['config']
        source_bytes = source.read_bytes()
        source_hash = sha(source_bytes)
        if source_hash.lower() != job['config_sha256'].lower():
            raise ValueError(f'Source config does not match manifest: {source}')
        cfg = yaml.safe_load(source_bytes)
        if (cfg['training']['seed'] != seed or cfg['data']['test_data_split'] != 0.1
                or cfg['data']['valid_data_split'] != 0.1
                or cfg['training']['scaler_fit_scope'] != 'train_only'):
            raise ValueError(f'Unexpected original configuration: {source}')
        name = f'{EXPERIMENT}_{job["alias"]}_seed{seed}'
        if job['family'] == 'difficult_gradnorm':
            name += DG_SUFFIX
        cfg['exp_name'] = name
        cfg['data']['training_files'] = FILES.copy()
        cfg['data']['test_data_split'] = 0.1
        # The existing split function applies this fraction to the first 90%.
        cfg['data']['valid_data_split'] = 1 / 9
        cfg['data']['split_before_downsample'] = True
        cfg['output']['results_dir'] = f'./results/pgin/retrain_lhs10_23_full_80_10_10/seed{seed}'
        config_path = NEW / f'seed{seed}' / f'{job["alias"]}.yaml'
        data = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode('utf-8')
        row = dict(job)
        if job['module'] not in MODULES:
            raise ValueError(f'Unexpected source trainer: {job["module"]}')
        row.update({
            'template_config': job['config'],
            'template_sha256': source_hash,
            'config': config_path.relative_to(ROOT).as_posix(),
            'config_sha256': sha(data),
            'exp_name': name,
            'results': f'results/pgin/retrain_lhs10_23_full_80_10_10/seed{seed}/{name}',
            'checkpoint': f'saved_models/{name}.pth',
            'cli_args': job.get('cli_args', []),
            'module': MODULES[job['module']],
        })
        prepared.append((config_path, data, row))
    if len({row['exp_name'] for _, _, row in prepared}) != 63:
        raise ValueError('Experiment names must be unique')
    manifest = NEW / 'config_manifest.json'
    if manifest.exists() or any(path.exists() for path, _, _ in prepared):
        raise FileExistsError('Prepared experiment already exists; refusing to overwrite it')
    for path, data, _ in prepared:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest.write_text(json.dumps([row for _, _, row in prepared], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Prepared {len(prepared)} configurations at {NEW}')


if __name__ == '__main__':
    prepare()
