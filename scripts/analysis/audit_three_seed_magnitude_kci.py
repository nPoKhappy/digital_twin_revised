"""Independently audit quartile KCI results from the saved test arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
STAGE = ROOT / 'results/visualization/gain_magnitude_3seeds_staging'
SCALES = ROOT / 'results/steady_state/Tabular_MLP_New/zscore_std.csv'
TARGETS = ('B35_H2S', 'B35_SO2')
MVS = ('air2_SP', 'HEATER2_output_T_SP')
DISTS = ('in_range', 'soft_ood')


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=STAGE)
    args = parser.parse_args()
    out = args.directory.resolve()
    per_path = out/'conditional_kci_by_gain_magnitude_per_seed.csv'
    avg_path = out/'conditional_kci_by_gain_magnitude.csv'
    edges_path = out/'gain_magnitude_quartile_edges.csv'
    per = pd.read_csv(per_path)
    avg = pd.read_csv(avg_path)
    edges = pd.read_csv(edges_path)
    selected = json.loads((SOURCE/'selected_models.json').read_text(encoding='utf8'))['selected_models']
    if len(selected) != 15 or len(per) != 480 or len(avg) != 160 or len(edges) != 8:
        raise AssertionError('Unexpected output coverage')
    per = per.set_index(['model', 'distribution', 'target', 'mv', 'quartile'], verify_integrity=True)
    avg = avg.set_index(['family', 'alias', 'distribution', 'target', 'mv', 'quartile'], verify_integrity=True)
    edges = edges.set_index(['distribution', 'target', 'mv'], verify_integrity=True)
    official = pd.read_csv(SOURCE/'kci_components.csv').set_index(['model', 'distribution', 'target', 'mv'], verify_integrity=True)
    scales = pd.read_csv(SCALES, index_col=0).iloc[:, 0]
    if not all(float(scales[k]) > 0 for k in (*MVS, *TARGETS)):
        raise AssertionError('Nonpositive normalization scale')
    protocol_hash = sha(SOURCE/'run_protocol.json')
    refs, data = {}, {}
    max_diff = 0.0
    checked_official = 0
    for model in selected:
        name, family, alias, seed = (model[k] for k in ('model', 'family', 'alias', 'seed'))
        dest = SOURCE/'model_data'/name
        complete = json.loads((dest/'complete.json').read_text(encoding='utf8'))
        if complete['protocol_sha256'] != protocol_hash:
            raise AssertionError(f'Protocol mismatch: {name}')
        for distribution in DISTS:
            file = dest/f'gain_{distribution}.npz'
            if sha(file) != complete['outputs'][file.name]:
                raise AssertionError(f'NPZ hash mismatch: {file}')
            with np.load(file) as f:
                teacher, dynamic = f['teacher_gain'].copy(), f['dynamic_gain'].copy()
            n = 4385 if distribution == 'in_range' else 2692
            if teacher.shape != (n,2,2) or dynamic.shape != (2,18,n,2,2):
                raise AssertionError('Invalid shape')
            if distribution in refs and not np.array_equal(teacher, refs[distribution]):
                raise AssertionError('Teacher differs across models or seeds')
            refs[distribution] = teacher
            for ti,target in enumerate(TARGETS):
                for mi,mv in enumerate(MVS):
                    t = teacher[:,ti,mi]
                    g = dynamic[:,:,:,ti,mi]
                    valid = np.abs(t) >= 1e-5
                    raw = np.abs(t[valid]).astype('float64')
                    normalized = raw * float(scales[mv]) / float(scales[target])
                    q = np.quantile(normalized,[.25,.5,.75])
                    raw_q = np.quantile(raw,[.25,.5,.75])
                    meta = edges.loc[(distribution,target,mv)]
                    if int(meta.valid_probe_count) != len(raw):
                        raise AssertionError('Wrong valid probe count')
                    np.testing.assert_allclose(raw_q, [meta.q25_abs_raw_gain,meta.q50_abs_raw_gain,meta.q75_abs_raw_gain], rtol=1e-12, atol=1e-15)
                    np.testing.assert_allclose(q, [meta.q25_abs_normalized_gain,meta.q50_abs_normalized_gain,meta.q75_abs_normalized_gain], rtol=1e-12, atol=1e-15)
                    # Explicit half-open comparisons are independent of searchsorted.
                    masks = [normalized<q[0], (normalized>=q[0]) & (normalized<q[1]),
                             (normalized>=q[1]) & (normalized<q[2]), normalized>=q[2]]
                    success = ((t[None,None,:]>0)&(g>0)) | ((t[None,None,:]<0)&(g<0))
                    success = success[:,:,valid]
                    sum_total = sum_correct = 0
                    for i, mask in enumerate(masks,1):
                        n_probe = int(mask.sum())
                        correct, total = int(success[:,:,mask].sum()), 36*n_probe
                        row = per.loc[(name,distribution,target,mv,f'Q{i}')]
                        if (int(meta[f'Q{i}_probe_count']) != n_probe or int(row.probe_count) != n_probe
                                or int(row.total) != total or int(row.correct) != correct
                                or row.family != family or row.alias != alias or int(row.seed) != seed):
                            raise AssertionError(f'Quartile count/metadata mismatch {name} {distribution} {target} {mv} Q{i}')
                        value = 100*correct/total
                        max_diff = max(max_diff, abs(value-float(row.kci_percent)))
                        data[(family,alias,distribution,target,mv,f'Q{i}',seed)] = value
                        sum_correct += correct
                        sum_total += total
                    row = official.loc[(name,distribution,target,mv)]
                    if int(row.valid_comparisons) != sum_total or int(row.tp+row.tn) != sum_correct:
                        raise AssertionError('Quartile sums disagree with official KCI')
                    checked_official += 1
    max_sd = 0.0
    for index,row in avg.iterrows():
        values = [data[(*index,s)] for s in (42,43,44)]
        max_diff = max(max_diff, abs(statistics.mean(values)-float(row.mean_kci_percent)))
        max_sd = max(max_sd, abs(statistics.stdev(values)-float(row.sd_kci_percentage_points)))
        if int(row.n_seeds) != 3:
            raise AssertionError('Wrong seed count')
    provenance = json.loads((out/'conditional_kci_by_gain_magnitude_provenance.json').read_text(encoding='utf8'))
    for path, expected in provenance['input_sha256'].items():
        if sha(ROOT/path) != expected:
            raise AssertionError(f'Input hash differs: {path}')
    for path, expected in provenance['output_sha256'].items():
        if sha(out/path) != expected:
            raise AssertionError(f'Output hash differs: {path}')
    if max_diff > 1e-11 or max_sd > 1e-11 or checked_official != 120:
        raise AssertionError('Numeric mismatch')
    audit = dict(passed=True, models=15, per_seed_groups=len(per), summary_groups=len(avg),
                 quartile_edge_groups=len(edges), official_pair_counts_checked=checked_official,
                 common_teacher_arrays_verified=True, half_open_quartile_recount=True,
                 mean_sample_sd_recalculated_from_npz=True,
                 max_kci_difference_percentage_points=max_diff,
                 max_sd_difference_percentage_points=max_sd,
                 per_seed_csv_sha256=sha(per_path), mean_sd_csv_sha256=sha(avg_path),
                 quartile_edges_sha256=sha(edges_path), audit_script_sha256=sha(Path(__file__)))
    (out/'independent_three_seed_magnitude_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
