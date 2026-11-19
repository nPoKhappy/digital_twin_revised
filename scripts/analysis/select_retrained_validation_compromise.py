"""Freeze the four validation-selected configurations before test evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
OUTPUT = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_staging'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    paths = [INPUT / n for n in ('regional_coordinates_63_models.csv', 'relative_rmse_components.csv',
             'scores_by_model_seed.csv', 'scores_by_family_model.csv', 'analysis_validation.json', 'provenance.json')]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [*paths, MANIFEST, Path(__file__)]}
    provenance = json.loads((INPUT/'provenance.json').read_text(encoding='utf-8-sig'))
    validated = json.loads((INPUT/'analysis_validation.json').read_text(encoding='utf-8-sig'))
    if not (validated.get('passed') and validated.get('model_count') == 63 and validated.get('seeds') == list(SEEDS)
            and validated.get('components') == 2268 and validated.get('regional_points') == 126
            and provenance.get('baseline_seed') == 42):
        raise ValueError('Fixed No-Gain seed 42 validation was not completed successfully.')
    for name, digest in provenance['outputs_sha256'].items():
        if sha(INPUT/name).lower() != digest.lower():
            raise ValueError(f'Original validation output changed: {name}')
    for relative, digest in provenance['sources_sha256'].items():
        path = ROOT / relative
        if sha(path).lower() != digest.lower():
            raise ValueError(f'Original validation source changed: {relative}')

    jobs = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
    jobs42 = [job for job in jobs if job['seed'] == 42]
    model_family = {j['alias']: j['family'] for j in jobs42}
    if len(jobs42) != 21 or len(model_family) != 21 or model_family.get('no_gain') != 'no_gain':
        raise ValueError('Expected 21 seed-42 models and one No-Gain.')
    for seed in SEEDS:
        current = [j for j in jobs if j['seed'] == seed]
        if len(current) != 21 or {j['alias']: j['family'] for j in current} != model_family:
            raise ValueError(f'Model family set differs for seed {seed}')
    for family in FAMILIES:
        if sum(f == family for f in model_family.values()) != 5:
            raise ValueError(f'Expected five candidates for {family}')

    coords = pd.read_csv(INPUT/'regional_coordinates_63_models.csv')
    expected = {(seed, model, region) for seed in SEEDS for model in model_family for region in ('internal', 'external')}
    keys = list(zip(coords.seed, coords.model, coords.region))
    if len(keys) != 126 or len(set(keys)) != 126 or set(keys) != expected:
        raise ValueError('Expected one coordinate per model, seed and region.')
    if not coords.family.eq(coords.model.map(model_family)).all():
        raise ValueError('Family mismatch in validation coordinates.')
    if not (np.isfinite(coords[['AvgRelRMSE', 'KCI_percent']]).all().all()
            and coords.AvgRelRMSE.gt(0).all() and coords.KCI_percent.between(0, 100).all()
            and np.allclose(coords.KCI_percent, 100*coords.correct/coords.valid, rtol=1e-12)):
        raise ValueError('Invalid validation coordinates or KCI counts.')
    pivot = coords.pivot(index=['model', 'family', 'seed'], columns='region', values=['AvgRelRMSE', 'KCI_percent'])
    pivot.columns = ['_'.join(x) for x in pivot.columns]
    pivot = pivot.reset_index()
    pivot['error_50_50'] = np.sqrt(pivot.AvgRelRMSE_internal * pivot.AvgRelRMSE_external)
    pivot['kci_50_50'] = (pivot.KCI_percent_internal + pivot.KCI_percent_external) / 2
    old = pd.read_csv(INPUT/'scores_by_model_seed.csv')
    check = pivot.merge(old, on=['model', 'family', 'seed'], validate='one_to_one')
    if len(check) != 63 or not np.allclose(check.error_50_50, check.overall_50_50, rtol=1e-12):
        raise ValueError('Fixed-baseline 50/50 error disagrees with validation scores.')
    all_candidates = []
    for family in FAMILIES:
        sub = pivot[pivot.family.eq(family)].groupby('model', as_index=False).agg(
            error_mean=('error_50_50', 'mean'), error_sd=('error_50_50', 'std'),
            kci_mean=('kci_50_50', 'mean'), kci_sd=('kci_50_50', 'std'), n=('seed', 'size'))
        if len(sub) != 5 or not sub.n.eq(3).all():
            raise ValueError(f'Wrong seed count in {family}')
        emin, emax = float(sub.error_mean.min()), float(sub.error_mean.max())
        kmin, kmax = float(sub.kci_mean.min()), float(sub.kci_mean.max())
        if emax == emin or kmax == kmin:
            raise ValueError(f'Zero normalization span in {family}')
        sub['error_normalized'] = (sub.error_mean - emin) / (emax - emin)
        sub['kci_shortfall_normalized'] = (kmax - sub.kci_mean) / (kmax - kmin)
        sub['distance'] = np.hypot(sub.error_normalized, sub.kci_shortfall_normalized)
        sub = sub.sort_values(['distance', 'error_mean', 'model'], kind='mergesort').reset_index(drop=True)
        for rank, row in enumerate(sub.to_dict('records'), 1):
            all_candidates.append({'family': family, 'alias': row['model'], 'rank': rank,
                'error_mean': float(row['error_mean']), 'error_sd': float(row['error_sd']),
                'kci_mean_percent': float(row['kci_mean']), 'kci_sd_percentage_points': float(row['kci_sd']),
                'error_normalized': float(row['error_normalized']), 'kci_shortfall_normalized': float(row['kci_shortfall_normalized']),
                'distance': float(row['distance']), 'n_seeds': int(row['n']),
                'error_family_min': emin, 'error_family_max': emax, 'kci_family_min_percent': kmin, 'kci_family_max_percent': kmax})
    selected = {family: next(x['alias'] for x in all_candidates if x['family'] == family and x['rank'] == 1) for family in FAMILIES}
    rule = ('Within each family, average the three per-seed sqrt(internal AvgRelRMSE * external AvgRelRMSE) values '
            'against fixed No-Gain seed 42; average the internal and external ANN-reference KCI equally for each seed, '
            'then average across seeds. Min-max normalize the five family candidates separately: '
            'error=(E-Emin)/(Emax-Emin), KCI shortfall=(Kmax-K)/(Kmax-Kmin). '
            'Choose the smallest Euclidean distance sqrt(error^2+KCI_shortfall^2), with equal axis weight; '
            'resolve exact ties by lower mean error and then alias. Validation only; No-Gain is the fixed reference.')
    selection = {'aliases': selected, 'selection_rule': rule,
        'selection_details': {'baseline': 'No-Gain seed 42', 'validation_only': True, 'seeds': list(SEEDS),
                              'rmse_and_kci_regions': 'internal and external, equal 50/50', 'family_candidate_count': 5,
                              'tie_break': ['distance', 'error_mean', 'alias'],
                              'normalization_note': 'Distances are relative to the five candidates within each family and are not comparable across families.'},
        'provenance': {'sources_sha256': sources, 'upstream_sources_checked': len(provenance['sources_sha256']),
                       'upstream_output_sha256_checked': provenance['outputs_sha256']},
        'scores': all_candidates}
    for path, digest in sources.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Selection input changed during computation: {path}')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT/'validation_selection_candidates.csv'
    plan_path = OUTPUT/'validation_selection_plan.json'
    if table_path.exists() or plan_path.exists():
        raise FileExistsError('Selection plan or candidate table already exists; refusing to overwrite.')
    with table_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_candidates[0])); writer.writeheader(); writer.writerows(all_candidates)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'aliases': selected, 'plan': str(plan_path), 'table': str(table_path)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
