"""Run the expanded Difficult-GradNorm ratios, optionally in two disjoint groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
RATIOS = ('0.15', '1.5', '15', '150')
GROUPS = {'low': ('0.15', '1.5'), 'high': ('15', '150')}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(sources: dict[str, str]) -> None:
    for path, expected in sources.items():
        p = Path(path)
        if not p.is_file() or sha(p).lower() != expected.lower():
            raise ValueError(f'Protected training source changed or is missing: {p}')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def prepare(seed: int, group: str | None) -> dict:
    entries = json.loads(MANIFEST.read_text(encoding='utf-8'))
    all_jobs = [x for x in entries if x['seed'] == seed]
    if len(entries) != 12 or len(all_jobs) != 4 or [x['target_ratio'] for x in all_jobs] != list(RATIOS):
        raise ValueError('Expanded DG manifest must specify four ratios for each of three seeds')
    ratios = GROUPS[group] if group else RATIOS
    jobs = [x for x in all_jobs if x['target_ratio'] in ratios]
    if [x['target_ratio'] for x in jobs] != list(ratios):
        raise ValueError('Unexpected ratio ordering in selected group')
    scales = {j['max_scale'] for j in entries}
    if len(scales) != 1 or not math.isfinite(float(next(iter(scales)))) or float(next(iter(scales))) < 1:
        raise ValueError('All expanded jobs must share one finite max-scale of at least 1')
    original_path = ROOT / f'results/pgin/retrain_lhs10_23/seed{seed}/_batch_run.json'
    original = json.loads(original_path.read_text(encoding='utf-8-sig'))
    if original.get('status') != 'complete' or original.get('seed') != seed:
        raise ValueError('Original seed training batch is not complete')
    sources = original['teacher_and_source_sha256']
    verify(sources)
    actual_lhs = {p.name for p in (ROOT/'data/Claus_steady_state').glob('lhs_generated_dynamic_ss_data*.xlsx')}
    expected_lhs = {f'lhs_generated_dynamic_ss_data_{i}.xlsx' for i in (10,11,15,16,17,18,19,20,21,22,23)}
    if actual_lhs != expected_lhs:
        raise ValueError('Steady-state source list differs from the shared ANN training sources')
    protected = {str(MANIFEST): sha(MANIFEST), str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                 str(original_path): sha(original_path)}
    batch_dir = original_path.parent
    record_path = batch_dir / (f'_dg_expanded_{group}_batch_run.json' if group else '_dg_expanded_batch_run.json')
    if record_path.exists():
        raise FileExistsError(f'Expanded training record exists; inspect it before any rerun: {record_path}')
    conflicting_records = ([batch_dir / '_dg_expanded_batch_run.json'] if group else
                           [batch_dir / f'_dg_expanded_{g}_batch_run.json' for g in GROUPS])
    if any(p.exists() for p in conflicting_records):
        raise FileExistsError('An incompatible full/group batch record already exists; inspect before continuing')
    for job in jobs:
        path = ROOT / job['config']
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        if (sha(path) != job['config_sha256'] or cfg['training']['seed'] != seed
                or cfg['exp_name'] != job['exp_name'] or cfg['training']['scaler_fit_scope'] != 'train_only'
                or job['module'] != 'scripts.training.train_pgin_stepwise_gain_difficult_group_gradnorm'
                or job['cli_args'] != ['--no-ss','--target-ratio',job['target_ratio'],'--max-scale',job['max_scale'],
                                       '--use-config-steady-state-source']):
            raise ValueError(f'Config or CLI mismatch: {job["alias"]}')
        for output in ('checkpoint', 'results'):
            if (ROOT/job[output]).exists():
                raise FileExistsError(f'Refusing to overwrite {job[output]}')
        protected[str(path)] = sha(path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing to start this training batch')
    print(f'Seed {seed}: {len(jobs)} ratios {", ".join(ratios)}; max-scale {next(iter(scales))}; '
          'shared ANN and protected sources verified.', flush=True)
    return {'seed': seed, 'group': group, 'jobs': jobs, 'record_path': record_path,
            'original_path': original_path, 'sources': sources, 'protected': protected}


def run_seed(prepared: dict) -> None:
    seed, jobs = prepared['seed'], prepared['jobs']
    sources, protected = prepared['sources'], prepared['protected']
    record_path, original_path = prepared['record_path'], prepared['original_path']
    verify(protected)
    record = {'status':'running','seed':seed,'group':prepared['group'],'started_utc':now(),'active':None,'completed':[],
              'shared_original_batch':str(original_path.relative_to(ROOT)), 'teacher_and_source_sha256':sources,
              'expanded_source_sha256':protected, 'manifest':str(MANIFEST.relative_to(ROOT))}
    with record_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    try:
        for i, job in enumerate(jobs, 1):
            verify(sources); verify(protected)
            for output in ('checkpoint', 'results'):
                if (ROOT/job[output]).exists():
                    raise FileExistsError(f'Output appeared after preflight: {job[output]}')
            record['active'] = job['exp_name']
            save(record_path, record)
            print(f'[{i}/{len(jobs)}] seed {seed}, ratio {job["target_ratio"]}', flush=True)
            subprocess.run([sys.executable, '-u', '-m', job['module'], '--config', str(ROOT/job['config']),
                            *job['cli_args']], cwd=ROOT, check=True)
            result = ROOT/job['results']; checkpoint = ROOT/job['checkpoint']
            artifacts = [checkpoint, result/'zscore_mean.csv', result/'zscore_std.csv']
            if any(not p.is_file() for p in artifacts):
                raise FileNotFoundError(f'Training finished without required artifacts: {job["exp_name"]}')
            verify(sources); verify(protected)
            record['completed'].append({'exp_name':job['exp_name'], 'target_ratio':job['target_ratio'],
                'checkpoint':job['checkpoint'], 'checkpoint_sha256':sha(checkpoint),
                'zscore_mean_sha256':sha(artifacts[1]), 'zscore_std_sha256':sha(artifacts[2])})
            record['active'] = None
            save(record_path, record)
        record['status'] = 'complete'
    except BaseException as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
        raise
    finally:
        record['finished_utc'] = now()
        save(record_path, record)
    print(f'Completed {len(jobs)} expanded Difficult-GradNorm experiments for seed {seed}.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), help='One seed; omit with --group to run all three.')
    parser.add_argument('--group', choices=tuple(GROUPS), help='low: 0.15/1.5; high: 15/150.')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and outputs without training.')
    args = parser.parse_args()
    if args.seed is None and args.group is None:
        parser.error('Specify --seed, --group, or both')
    os.chdir(ROOT)
    seeds = (args.seed,) if args.seed is not None else (42, 43, 44)
    prepared = [prepare(seed, args.group) for seed in seeds]
    if args.check_only:
        print('Check-only passed; no training or output started.', flush=True)
        return
    # Reserve each experiment atomically across all selected seeds. Different
    # groups may run simultaneously; overlapping full/partial runs may not.
    locks = []
    try:
        for batch in prepared:
            lock_dir = batch['record_path'].parent / '_dg_expanded_job_locks'
            lock_dir.mkdir(exist_ok=True)
            for job in batch['jobs']:
                lock_path = lock_dir / (job['alias'] + '.lock')
                with lock_path.open('x', encoding='utf-8') as stream:
                    stream.write(f'pid={os.getpid()}\n')
                locks.append(lock_path)
        # Close races between preflight and lock acquisition.
        prepared = [prepare(seed, args.group) for seed in seeds]
        for batch in prepared:
            run_seed(batch)
    finally:
        for lock_path in locks:
            lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
