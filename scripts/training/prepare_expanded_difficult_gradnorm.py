"""Prepare four new Difficult-GradNorm configurations for each training seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / 'configs/pgin/retrain_lhs10_23'
RATIOS = ('0.15', '1.5', '15', '150')
SEEDS = (42, 43, 44)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-scale', type=float, default=50.0)
    args = parser.parse_args()
    if not math.isfinite(args.max_scale) or args.max_scale < 1:
        raise ValueError('max-scale must be finite and at least 1')
    max_scale = format(args.max_scale, '.15g')
    original = json.loads((CONFIG_ROOT / 'config_manifest.json').read_text(encoding='utf-8-sig'))
    jobs = []
    for seed in SEEDS:
        reference = [x for x in original if x['seed'] == seed and x['alias'] == 'dg_ratio0p015']
        if len(reference) != 1:
            raise ValueError(f'Missing original 0.015 manifest entry for seed {seed}')
        reference = reference[0]
        reference_path = ROOT / reference['config']
        if digest(reference_path).lower() != reference['config_sha256'].lower():
            raise ValueError(f'Original template differs from manifest: {reference_path}')
        template = reference_path.read_text(encoding='utf-8')
        if template.count(reference['exp_name']) != 1:
            raise ValueError(f'Unexpected template experiment name count: {reference_path}')
        for ratio in RATIOS:
            alias = 'dg_ratio' + ratio.replace('.', 'p')
            exp_name = f'lhs10_23_{alias}_seed{seed}_difficult_group_gradnorm_no_ss'
            path = CONFIG_ROOT / f'seed{seed}' / f'{alias}.yaml'
            content = template.replace(reference['exp_name'], exp_name)
            if path.exists() and path.read_text(encoding='utf-8') != content:
                raise FileExistsError(f'Refusing to replace different config: {path}')
            if not path.exists():
                path.write_text(content, encoding='utf-8', newline='\n')
            jobs.append(dict(seed=seed, family='difficult_gradnorm', alias=alias,
                target_ratio=ratio, max_scale=max_scale, config=str(path.relative_to(ROOT)).replace('\\', '/'),
                config_sha256=digest(path), template_config=reference['config'],
                template_sha256=reference['config_sha256'], exp_name=exp_name,
                results=f'results/pgin/retrain_lhs10_23/seed{seed}/{exp_name}',
                checkpoint=f'saved_models/{exp_name}.pth',
                module='scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm',
                cli_args=['--no-ss','--target-ratio',ratio,'--max-scale',max_scale,
                          '--use-config-steady-state-source']))
    out = CONFIG_ROOT / 'config_manifest_dg_expanded.json'
    data = json.dumps(jobs, indent=2, ensure_ascii=False) + '\n'
    if out.exists() and out.read_text(encoding='utf-8') != data:
        for job in jobs:
            if (ROOT/job['checkpoint']).exists() or (ROOT/job['results']).exists():
                raise FileExistsError('Refusing to change manifest after training artifacts exist')
        for seed in SEEDS:
            if (ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}/_dg_expanded_batch_run.json').exists():
                raise FileExistsError('Refusing to change manifest after a batch has started')
        temporary = out.with_suffix('.json.tmp')
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(data)
        os.replace(temporary, out)
    elif not out.exists():
        out.write_text(data, encoding='utf-8', newline='\n')
    print(f'Prepared {len(jobs)} configs: {out}')


if __name__ == '__main__':
    main()
