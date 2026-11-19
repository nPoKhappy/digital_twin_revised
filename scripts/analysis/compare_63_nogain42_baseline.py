"""Recalculate validation ratios against the same seed-42 No-Gain baseline.

Reads completed experiments only; model outputs, metrics and KCI are not rerun.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/visualization'
OUT = BASE / 'retrain_lhs10_23_63_models_validation_nogain42_baseline'
OLD = BASE / 'retrain_lhs10_23_63_models_validation/regional_coordinates_63_models.csv'
MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest.json'
SEEDS = (42, 43, 44)
CASES = {'internal': ('R5-3', 'R5-4', 'R5-5'), 'external': ('R5-27', 'R5-28', 'R5-29')}
TARGETS = ('B35_H2S', 'B35_SO2')
EVALUATIONS = ('H1', 'H18', 'TS')
KEYS = ['partition', 'case', 'target', 'evaluation']


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gm(values) -> float:
    values = np.asarray(values, dtype=float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError('Geometric mean inputs must be positive and finite')
    return float(np.exp(np.log(values).mean()))


def main() -> None:
    names = ['regional_coordinates_63_models.csv', 'relative_rmse_components.csv',
             'scores_by_model_seed.csv', 'scores_by_family_model.csv',
             'baseline_raw_rmse.csv', 'analysis_validation.json', 'provenance.json', 'README.md']
    existing = [name for name in names if (OUT / name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite previous analysis: {existing}')
    jobs = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected_jobs = {s: [j for j in jobs if j['seed'] == s] for s in SEEDS}
    aliases = {j['alias']: j['family'] for j in expected_jobs[42]}
    if len(aliases) != 21 or 'no_gain' not in aliases:
        raise ValueError('Expected 21 model aliases including No-Gain')
    for s in SEEDS:
        if len(expected_jobs[s]) != 21 or {j['alias']: j['family'] for j in expected_jobs[s]} != aliases:
            raise ValueError(f'Model or family mismatch for seed {s}')

    cases = [c for group in CASES.values() for c in group]
    expected = {(region, case, target, ev) for region, region_cases in CASES.items()
                for case in region_cases for target in TARGETS for ev in EVALUATIONS}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), MANIFEST, OLD]}
    truth_ref = {}
    metrics_all = []
    for seed in SEEDS:
        directory = BASE / f'retrain_lhs10_23_21_models_validation_seed{seed}'
        protocol = directory / 'run_protocol.json'
        evaluation = directory / 'evaluation_validation.json'
        summary_path = directory / 'validation_metrics.csv'
        kci_path = directory / 'kci_components.csv'
        coordinates_path = directory / 'regional_plot_coordinates.csv'
        for path in (protocol, evaluation, summary_path, kci_path, coordinates_path):
            sources[str(path.relative_to(ROOT))] = sha(path)
        result = json.loads(evaluation.read_text(encoding='utf-8-sig'))
        if not (result.get('passed') and result.get('seed') == seed and result.get('model_count') == 21
                and result.get('per_region_points') == 21 and result.get('source_hashes_unchanged')):
            raise ValueError(f'Incomplete original evaluation for seed {seed}')
        data = pd.read_csv(summary_path)
        if len(data) != 21 * 36 or data.duplicated(['model'] + KEYS).any():
            raise ValueError(f'Missing or duplicate metrics for seed {seed}')
        if set(data.model) != set(aliases):
            raise ValueError(f'Model set mismatch for seed {seed}')
        coords = pd.read_csv(coordinates_path)
        all_kci = pd.read_csv(kci_path)
        if len(coords) != 42 or len(all_kci) != 168:
            raise ValueError(f'Missing original coordinates/KCI components for seed {seed}')
        for model in aliases:
            per = directory / 'model_data' / model
            complete_path = per / 'complete.json'
            complete = json.loads(complete_path.read_text(encoding='utf-8-sig'))
            if complete.get('run_protocol_sha256') != sha(protocol):
                raise ValueError(f'Protocol hash differs: {seed} {model}')
            sources[str(complete_path.relative_to(ROOT))] = sha(complete_path)
            needed = ['validation_metrics.csv', 'kci_components.csv', 'gain_in_range.npz', 'gain_soft_ood.npz']
            needed += [f'{case}_prediction.npz' for case in cases]
            for name in needed:
                path = per / name
                digest = sha(path)
                if complete.get('outputs', {}).get(name) != digest:
                    raise ValueError(f'Output hash differs: {seed} {model} {name}')
                sources[str(path.relative_to(ROOT))] = digest
            stored_metrics = pd.read_csv(per / 'validation_metrics.csv')
            matches = data[data.model.eq(model)].merge(stored_metrics, on=['model'] + KEYS,
                                                        suffixes=('_combined', '_model'), validate='one_to_one')
            if len(matches) != 36 or set(map(tuple, matches[KEYS].to_numpy())) != expected:
                raise ValueError(f'Wrong conditions: {seed} {model}')
            if not np.array_equal(matches.n_combined, matches.n_model):
                raise ValueError(f'Sample count differs: {seed} {model}')
            np.testing.assert_allclose(matches.RMSE_combined, matches.RMSE_model, rtol=1e-13, atol=1e-15)
            stored_kci = pd.read_csv(per / 'kci_components.csv')
            match_kci = all_kci[all_kci.model.eq(model)].merge(
                stored_kci, on=['model', 'distribution', 'target', 'mv'],
                suffixes=('_combined', '_model'), validate='one_to_one')
            if len(match_kci) != 8:
                raise ValueError(f'Wrong KCI coverage: {seed} {model}')
            for col in ('valid_comparisons', 'tp', 'tn', 'fp', 'fn', 'zero_gain'):
                if not np.array_equal(match_kci[f'{col}_combined'], match_kci[f'{col}_model']):
                    raise ValueError(f'KCI counts differ: {seed} {model} {col}')
            for case in cases:
                with np.load(per / f'{case}_prediction.npz') as archive:
                    for key in ('ts_true', 'horizon_true'):
                        value = archive[key]
                        if (case, key) not in truth_ref:
                            if seed != 42 or model != 'no_gain':
                                # Manifest order is not an appropriate data-integrity assumption.
                                reference = BASE / 'retrain_lhs10_23_21_models_validation_seed42' / 'model_data' / 'no_gain' / f'{case}_prediction.npz'
                                with np.load(reference) as ref:
                                    truth_ref[(case, key)] = ref[key].copy()
                            else:
                                truth_ref[(case, key)] = value.copy()
                        np.testing.assert_allclose(value, truth_ref[(case, key)], rtol=1e-12, atol=1e-14)
        data['seed'] = seed
        metrics_all.append(data)

    metrics = pd.concat(metrics_all, ignore_index=True)
    baseline = metrics[(metrics.seed == 42) & metrics.model.eq('no_gain')][KEYS + ['RMSE', 'n']].copy()
    baseline = baseline.rename(columns={'RMSE': 'NoGain_seed42_RMSE', 'n': 'NoGain_seed42_n'})
    if len(baseline) != 36 or baseline.duplicated(KEYS).any():
        raise ValueError('Incorrect seed-42 No-Gain benchmark')
    components = metrics.merge(baseline, on=KEYS, validate='many_to_one')
    if len(components) != 63 * 36 or not components.n.eq(components.NoGain_seed42_n).all():
        raise ValueError('Wrong paired sample counts or incomplete comparisons')
    components['relative_RMSE'] = components.RMSE / components.NoGain_seed42_RMSE
    if not np.isfinite(components.relative_RMSE).all() or not components.relative_RMSE.gt(0).all():
        raise ValueError('Nonfinite or nonpositive RMSE ratio')
    components['family'] = components.model.map(aliases)
    regions = components.groupby(['seed', 'model', 'family', 'partition'], as_index=False).agg(
        AvgRelRMSE=('relative_RMSE', gm), component_count=('relative_RMSE', 'size'))
    if len(regions) != 126 or not regions.component_count.eq(18).all():
        raise ValueError('Unexpected regional aggregation')

    previous = pd.read_csv(OLD)
    if len(previous) != 126 or previous.duplicated(['seed', 'model', 'region']).any():
        raise ValueError('Prior KCI coordinates incomplete')
    coords = regions.rename(columns={'partition': 'region'}).merge(
        previous[['seed', 'model', 'family', 'region', 'KCI_percent', 'correct', 'valid']],
        on=['seed', 'model', 'family', 'region'], validate='one_to_one')
    if len(coords) != 126 or not np.allclose(coords.KCI_percent, 100 * coords.correct / coords.valid, rtol=1e-12):
        raise ValueError('Original KCI counts inconsistent')
    for seed in SEEDS:
        path = BASE / f'retrain_lhs10_23_21_models_validation_seed{seed}' / 'regional_plot_coordinates.csv'
        original = pd.read_csv(path).rename(columns={'partition': 'region', 'ANN_reference_KCI_percent': 'KCI_percent'})
        both = coords[coords.seed.eq(seed)].merge(original, on=['model', 'family', 'region'],
                                                   suffixes=('_new', '_original'), validate='one_to_one')
        if len(both) != 42:
            raise ValueError(f'Original KCI coordinates missing: {seed}')
        for col in ('KCI_percent', 'correct', 'valid'):
            np.testing.assert_allclose(both[f'{col}_new'], both[f'{col}_original'], rtol=1e-12, atol=1e-12)
    scores = coords.pivot(index=['seed', 'model', 'family'], columns='region', values='AvgRelRMSE').reset_index()
    scores['overall_50_50'] = np.sqrt(scores.internal * scores.external)
    scores = scores.sort_values(['overall_50_50', 'family', 'model', 'seed']).reset_index(drop=True)
    scores.insert(0, 'rank_single_run', np.arange(1, len(scores) + 1))
    family = scores.groupby(['family', 'model'], as_index=False).agg(
        mean=('overall_50_50', 'mean'), sd=('overall_50_50', 'std'),
        minimum=('overall_50_50', 'min'), maximum=('overall_50_50', 'max'), n_seeds=('seed', 'nunique'))
    family = family.sort_values('mean').reset_index(drop=True)
    family.insert(0, 'rank_mean', np.arange(1, len(family) + 1))
    check = scores[(scores.seed == 42) & scores.model.eq('no_gain')]
    if len(check) != 1 or not np.allclose(check[['internal', 'external', 'overall_50_50']], 1):
        raise AssertionError('No-Gain seed 42 should be exactly one')

    OUT.mkdir(parents=True, exist_ok=True)
    coords[['seed', 'model', 'family', 'region', 'KCI_percent', 'AvgRelRMSE', 'correct', 'valid']].sort_values(
        ['seed', 'model', 'region']).to_csv(OUT / names[0], index=False, float_format='%.17g')
    components[['seed', 'model', 'family'] + KEYS + ['n', 'RMSE', 'NoGain_seed42_n', 'NoGain_seed42_RMSE', 'relative_RMSE']].to_csv(
        OUT / names[1], index=False, float_format='%.17g')
    scores.to_csv(OUT / names[2], index=False, float_format='%.17g')
    family.to_csv(OUT / names[3], index=False, float_format='%.17g')
    baseline.sort_values(KEYS).to_csv(OUT / names[4], index=False, float_format='%.17g')
    validation = {'passed': True, 'model_count': 63, 'seeds': list(SEEDS), 'components': len(components),
                  'regional_points': len(coords), 'components_per_model_region': 18,
                  'baseline': 'No-Gain seed 42, matching partition/case/target/evaluation',
                  'all_378_prediction_truth_arrays_checked': True, 'kci_unchanged': True,
                  'original_model_outputs_hash_checked': True, 'training_or_inference_rerun': False}
    (OUT / names[5]).write_text(json.dumps(validation, indent=2), encoding='utf-8')
    readme = '''# 統一 No-Gain seed 42 的驗證比較

對所有模型與 seed，每個案例／輸出／評估方式的分母均為同條件 **No-Gain seed 42** 原始 RMSE；分子使用已完成的 validation 原始 RMSE。各區 3 案例 × 2 輸出 × H1/H18/TS 的 18 項比率取幾何平均，內外各 50% 的綜合分數為 `sqrt(內插 × 外插)`。分數是相對誤差，不是單一原始 RMSE。

內插組 R5-3/4/5；外插組 R5-27/28/29。R5-5 約 33.19% 原始時間點在核心範圍外。KCI 與前次完全相同，使用另一份穩態驗證資料，是 ANN 參考增益符號一致率，不代表 Aspen 真實增益或閉迴路表現。

No-Gain seed 42 為 1；No-Gain seed 43、44 可偏離 1，因為它們使用同一個 seed 42 分母。單次最低只是 validation 的事後排序；多次訓練結果仍應一起報告，且不可用 test 進行選模。

- `regional_coordinates_63_models.csv`：126 個區域座標，KCI、correct、valid 維持原值。
- `relative_rmse_components.csv`：2,268 個逐案例相對 RMSE 分量與分子、分母、樣本數。
- `scores_by_model_seed.csv`：63 個模型每區與內外各 50% 綜合分數。
- `scores_by_family_model.csv`：21 個設定的三 seed 均值、樣本標準差與範圍。
- `baseline_raw_rmse.csv`：36 個 No-Gain seed 42 原始 RMSE 基準。
- `provenance.json` 和 `analysis_validation.json`：來源 SHA-256 和驗證摘要。

重算：`.\\.venv\\Scripts\\python.exe -m scripts.analysis.compare_63_nogain42_baseline`。為防止覆寫，本程式拒絕覆蓋已有結果。
'''
    (OUT / names[7]).write_text(readme, encoding='utf-8')
    outputs = {name: sha(OUT / name) for name in names if name != 'provenance.json'}
    provenance = {'baseline_seed': 42, 'sources_sha256': sources, 'outputs_sha256': outputs,
                  'aggregation': '18 component geometric mean per region, then sqrt(internal*external)',
                  'kci': 'unchanged from previous validation; ANN reference',
                  'scope': 'validation only; no model retraining, no predictions recomputed'}
    (OUT / names[6]).write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Completed:', OUT)
    print('No-Gain:', scores[scores.model.eq('no_gain')][['seed', 'internal', 'external', 'overall_50_50']].to_string(index=False))
    print('Best in each family:', scores.sort_values('overall_50_50').groupby('family', sort=False).head(1).to_string(index=False))


if __name__ == '__main__':
    main()
