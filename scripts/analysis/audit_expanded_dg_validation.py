"""Independently verify the expanded DG validation and combined coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / 'results/visualization'
HISTORIC = VIS / 'retrain_lhs10_23_63_models_validation_nogain42_baseline'
INCREMENTAL = VIS / 'retrain_lhs10_23_expanded_dg_validation_staging'
COMBINED = VIS / 'retrain_lhs10_23_63_models_validation_nogain42_expanded_staging'
REGIONS = {'internal': ('in_range', ('R5-3', 'R5-4', 'R5-5')),
           'external': ('soft_ood', ('R5-27', 'R5-28', 'R5-29'))}
TARGETS = ('B35_H2S', 'B35_SO2')
MVS = ('air2_SP', 'HEATER2_output_T_SP')
KEYS = ['seed', 'model', 'partition', 'case', 'target', 'evaluation']


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--historic', type=Path, default=HISTORIC)
    parser.add_argument('--incremental', type=Path, default=INCREMENTAL)
    parser.add_argument('--combined', type=Path, default=COMBINED)
    parser.add_argument('--incremental-only', action='store_true')
    args = parser.parse_args()
    historic, incremental, combined_dir = args.historic, args.incremental, args.combined
    original = load(historic/'provenance.json')
    for name, digest in original['sources_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Historic validation source changed: {name}')
    for name, digest in original['outputs_sha256'].items():
        if sha(historic/name) != digest:
            raise ValueError(f'Historic validation output changed: {name}')
    protocol = load(incremental/'run_protocol.json')
    if not load(incremental/'evaluation_validation.json')['passed'] or len(protocol['jobs']) != 12:
        raise ValueError('Expanded DG validation is incomplete')
    for name, digest in protocol['sources_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Incremental validation source changed: {name}')
    metrics = pd.read_csv(incremental/'validation_metrics.csv').set_index(KEYS, verify_integrity=True)
    counts = pd.read_csv(incremental/'kci_components.csv').set_index(
        ['seed', 'model', 'distribution', 'target', 'mv'], verify_integrity=True)
    if len(metrics) != 432 or len(counts) != 96:
        raise ValueError('Expected 432 incremental RMSEs and 96 KCI component rows')
    protocol_sha = sha(incremental/'run_protocol.json')
    max_rmse = max_kci = 0.0
    for job in protocol['jobs']:
        seed, alias = job['seed'], job['alias']
        folder = incremental/'model_data'/f'seed{seed}'/alias
        marker = load(folder/'complete.json')
        if marker['run_protocol_sha256'] != protocol_sha:
            raise ValueError(f'Protocol mismatch: {seed}/{alias}')
        for name, digest in marker['outputs'].items():
            if sha(folder/name) != digest:
                raise ValueError(f'Model output changed: {seed}/{alias}/{name}')
        for partition, (distribution, cases) in REGIONS.items():
            with np.load(folder/f'gain_{distribution}.npz') as data:
                teacher, dynamic = data['teacher_gain'], data['dynamic_gain']
                if dynamic.shape != (2, 18, len(teacher), 2, 2):
                    raise ValueError('Unexpected dynamic gain shape')
                teacher_ref = VIS/'retrain_lhs10_23_21_models_validation_seed42/model_data/no_gain'/f'gain_{distribution}.npz'
                with np.load(teacher_ref) as original_gain:
                    old_teacher = original_gain['teacher_gain']
                    np.testing.assert_allclose(teacher, old_teacher, rtol=1e-5, atol=1e-10)
                    if not np.array_equal(np.abs(teacher) >= 1e-5, np.abs(old_teacher) >= 1e-5):
                        raise ValueError(f'ANN reference threshold membership differs: {seed}/{alias}/{distribution}')
                    valid_reference = np.abs(old_teacher) >= 1e-5
                    if not np.array_equal(np.sign(teacher[valid_reference]), np.sign(old_teacher[valid_reference])):
                        raise ValueError(f'ANN reference direction differs: {seed}/{alias}/{distribution}')
                for i, target in enumerate(TARGETS):
                    for j, mv in enumerate(MVS):
                        truth = np.broadcast_to(teacher[:, i, j], dynamic[:, :, :, i, j].shape)
                        pred = dynamic[:, :, :, i, j]
                        valid = np.abs(truth) >= 1e-5
                        rebuilt = {'possible_comparisons': valid.size, 'valid_comparisons': valid.sum(),
                                   'tp': ((truth > 0) & (pred > 0) & valid).sum(),
                                   'tn': ((truth < 0) & (pred < 0) & valid).sum(),
                                   'fp': ((truth < 0) & (pred > 0) & valid).sum(),
                                   'fn': ((truth > 0) & (pred < 0) & valid).sum(),
                                   'zero_gain': ((pred == 0) & valid).sum()}
                        reported = counts.loc[(seed, alias, distribution, target, mv)]
                        if any(int(reported[key]) != int(value) for key, value in rebuilt.items()):
                            raise ValueError(f'KCI recount mismatch: {seed}/{alias}/{distribution}/{target}/{mv}')
                        max_kci = max(max_kci, abs(float(reported.kci_percent) -
                            100*(int(rebuilt['tp'])+int(rebuilt['tn']))/int(rebuilt['valid_comparisons'])))
            for case in cases:
                baseline = VIS/'retrain_lhs10_23_21_models_validation_seed42/model_data/no_gain'/f'{case}_prediction.npz'
                with np.load(folder/f'{case}_prediction.npz') as data, np.load(baseline) as reference:
                    for key in ('ts_true', 'horizon_true'):
                        np.testing.assert_allclose(data[key], reference[key], rtol=1e-12, atol=1e-14)
                    for i, target in enumerate(TARGETS):
                        for ev in ('TS', 'H1', 'H18'):
                            if ev == 'TS':
                                truth, pred = data['ts_true'][:, i], data['ts_pred'][:, i]
                            else:
                                h = 0 if ev == 'H1' else 17
                                truth, pred = data['horizon_true'][:, h, i], data['horizon_pred'][:, h, i]
                            err = np.linalg.norm(truth-pred)/np.sqrt(len(truth))
                            row = metrics.loc[(seed, alias, partition, case, target, ev)]
                            if int(row.n) != len(truth):
                                raise ValueError('RMSE sample count mismatch')
                            max_rmse = max(max_rmse, abs(float(row.RMSE)-err))
    if max_rmse > 1e-12 or max_kci > 1e-9:
        raise ValueError('Incremental numeric mismatch')
    if args.incremental_only:
        result = dict(passed=True, expanded_models=12, incremental_rmse=432,
                      incremental_kci_components=96, historic_sources_checked=len(original['sources_sha256']),
                      max_rmse_abs_diff=max_rmse, max_kci_percentage_point_abs_diff=max_kci,
                      incremental_protocol_sha256=protocol_sha, audit_script_sha256=sha(Path(__file__)))
        text = json.dumps(result, indent=2, allow_nan=False)+'\n'
        if args.report:
            with args.report.open('x', encoding='utf8') as stream:
                stream.write(text)
        print(text)
        return

    prior = pd.read_csv(historic/'relative_rmse_components.csv')
    combined = pd.read_csv(combined_dir/'relative_rmse_components.csv')
    points = pd.read_csv(combined_dir/'regional_coordinates_63_models.csv')
    if len(combined) != 2268 or len(points) != 126 or combined.duplicated(KEYS).any():
        raise ValueError('Combined 63-model component coverage differs')
    prior_points = pd.read_csv(historic/'regional_coordinates_63_models.csv')
    kept_points = points[(points.family != 'difficult_gradnorm') | (points.model == 'dg_ratio0p015')]
    old_kept = prior_points[(prior_points.family != 'difficult_gradnorm') | (prior_points.model == 'dg_ratio0p015')]
    point_compare = kept_points.merge(old_kept, on=['seed', 'model', 'family', 'region'],
                                      suffixes=('_new', '_old'), validate='one_to_one')
    if len(point_compare) != 102:
        raise ValueError('Expected 102 unchanged coordinates from 51 historical models')
    for col in ('AvgRelRMSE', 'KCI_percent', 'correct', 'valid'):
        np.testing.assert_allclose(point_compare[f'{col}_new'], point_compare[f'{col}_old'], rtol=1e-12, atol=1e-12)
    reused = combined[(combined.family != 'difficult_gradnorm') | (combined.model == 'dg_ratio0p015')]
    old_reused = prior[(prior.family != 'difficult_gradnorm') | (prior.model == 'dg_ratio0p015')]
    matched = reused.merge(old_reused, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    if len(matched) != 51*36:
        raise ValueError('Reused 51 models do not match historic component coverage')
    for col in ('RMSE', 'NoGain_seed42_RMSE', 'relative_RMSE', 'n', 'NoGain_seed42_n'):
        np.testing.assert_allclose(matched[f'{col}_new'], matched[f'{col}_old'], rtol=1e-12, atol=1e-14)
    new = combined[(combined.family == 'difficult_gradnorm') & (combined.model != 'dg_ratio0p015')]
    check = new.merge(metrics.reset_index(), on=KEYS, suffixes=('_combined', '_raw'), validate='one_to_one')
    if len(check) != 12*36:
        raise ValueError('New models do not match incremental metrics')
    np.testing.assert_allclose(check.RMSE_combined, check.RMSE_raw, rtol=1e-12, atol=1e-14)
    base = prior[(prior.seed == 42) & (prior.model == 'no_gain')][
        ['partition', 'case', 'target', 'evaluation', 'RMSE', 'n']]
    pair = combined.merge(base, on=['partition', 'case', 'target', 'evaluation'],
                          suffixes=('', '_baseline'), validate='many_to_one')
    if len(pair) != 2268 or not pair.n.eq(pair.n_baseline).all():
        raise ValueError('Pairing to No-Gain 42 differs')
    np.testing.assert_allclose(pair.NoGain_seed42_RMSE, pair.RMSE_baseline, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(pair.relative_RMSE, pair.RMSE/pair.RMSE_baseline, rtol=1e-12, atol=1e-12)
    rebuilt = pair.groupby(['seed', 'model', 'family', 'partition'], as_index=False).agg(
        recomputed=('relative_RMSE', lambda x: float(np.exp(np.log(x).mean()))), n=('relative_RMSE', 'size'))
    merged = points.merge(rebuilt.rename(columns={'partition':'region'}),
                          on=['seed', 'model', 'family', 'region'], validate='one_to_one')
    if len(merged) != 126 or not merged.n.eq(18).all():
        raise ValueError('Regional RMSE component counts differ')
    np.testing.assert_allclose(merged.AvgRelRMSE, merged.recomputed, rtol=1e-12, atol=1e-12)
    kci = pd.read_csv(combined_dir/'kci_components.csv')
    if len(kci) != 504:
        raise ValueError('Expected 504 KCI component rows')
    sums = kci.assign(correct=kci.tp+kci.tn).groupby(['seed','model','distribution'], as_index=False).agg(
        total_correct=('correct','sum'), total_valid=('valid_comparisons','sum'))
    sums['region'] = sums.distribution.map({'in_range':'internal','soft_ood':'external'})
    checked = points.merge(sums, on=['seed','model','region'], validate='one_to_one')
    if len(checked)!=126 or not checked.correct.eq(checked.total_correct).all() or not checked.valid.eq(checked.total_valid).all():
        raise ValueError('Regional KCI counts mismatch')
    np.testing.assert_allclose(checked.KCI_percent, 100*checked.total_correct/checked.total_valid, rtol=1e-12)
    result = dict(passed=True, expanded_models=12, reused_models=51, incremental_rmse=432,
                  incremental_kci_components=96, combined_components=2268, regional_points=126,
                  historic_source_hashes_checked=len(original['sources_sha256']),
                  max_rmse_abs_diff=max_rmse, max_kci_percentage_point_abs_diff=max_kci,
                  incremental_protocol_sha256=protocol_sha, audit_script_sha256=sha(Path(__file__)))
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x', encoding='utf8') as stream:
            stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
