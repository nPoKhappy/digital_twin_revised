"""Independently verify three-seed conditional KCI from saved gain arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
STAGE = ROOT / 'results/visualization/gain_pair_difficulty_3seeds_staging'
OLD = ROOT / 'results/visualization/gain_pair_difficulty_seed42/conditional_kci_by_teacher_sign.csv'
TARGETS = ('B35_H2S', 'B35_SO2')
MVS = ('air2_SP', 'HEATER2_output_T_SP')
DISTS = ('in_range', 'soft_ood')


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=STAGE)
    args = parser.parse_args()
    directory = args.directory.resolve()
    per_path = directory / 'conditional_kci_by_teacher_sign_per_seed.csv'
    avg_path = directory / 'conditional_kci_by_teacher_sign.csv'
    per = pd.read_csv(per_path)
    averages = pd.read_csv(avg_path)
    selected = json.loads((SOURCE/'selected_models.json').read_text(encoding='utf8'))['selected_models']
    if len(selected) != 15 or len(per) != 240 or len(averages) != 80:
        raise AssertionError('Wrong model or group count')
    model_keys = ['model', 'distribution', 'target', 'mv', 'teacher_sign_label']
    agg_keys = ['family', 'alias', 'distribution', 'target', 'mv', 'teacher_sign_label']
    per = per.set_index(model_keys, verify_integrity=True)
    averages = averages.set_index(agg_keys, verify_integrity=True)
    old_frame = pd.read_csv(OLD)
    # The target is intentionally overwritten by the 3-seed summary after
    # the initial audit, so the previous seed-42 table may no longer exist.
    old = old_frame.set_index(model_keys, verify_integrity=True) if set(model_keys).issubset(old_frame.columns) else None
    official = pd.read_csv(SOURCE/'kci_components.csv').set_index(
        ['model', 'distribution', 'target', 'mv'], verify_integrity=True)
    actual_protocol = digest(SOURCE/'run_protocol.json')
    results = {}
    reference = {}
    max_difference = 0.0
    official_checked = old_checked = 0
    for model in selected:
        name, family, alias, seed = (model[k] for k in ('model', 'family', 'alias', 'seed'))
        dest = SOURCE/'model_data'/name
        complete = json.loads((dest/'complete.json').read_text(encoding='utf8'))
        if complete['protocol_sha256'] != actual_protocol:
            raise AssertionError('Protocol hash mismatch: ' + name)
        for dist in DISTS:
            p = dest/f'gain_{dist}.npz'
            if complete['outputs'][p.name] != digest(p):
                raise AssertionError('NPZ hash mismatch: ' + str(p))
            with np.load(p) as npz:
                teacher = npz['teacher_gain'].copy()
                dynamic = npz['dynamic_gain'].copy()
            n = 4385 if dist == 'in_range' else 2692
            if teacher.shape != (n, 2, 2) or dynamic.shape != (2, 18, n, 2, 2):
                raise AssertionError('Array shape mismatch')
            if dist not in reference:
                reference[dist] = teacher
            if not np.array_equal(reference[dist], teacher):
                raise AssertionError('Teacher or probe order differs')
            for tj, target in enumerate(TARGETS):
                for mj, mv in enumerate(MVS):
                    t, g = teacher[:, tj, mj], dynamic[:, :, :, tj, mj]
                    valid = np.abs(t) >= 1e-5
                    # Explicit sign equality: zero dynamic is always incorrect.
                    successes = ((t[None, None, :] > 0) & (g > 0)) | ((t[None, None, :] < 0) & (g < 0))
                    direct_total = direct_correct = 0
                    for sign in ('Negative', 'Positive'):
                        mask = valid & ((t < 0) if sign == 'Negative' else (t > 0))
                        n_probe = int(mask.sum())
                        n_total = n_probe * 36
                        n_correct = int(successes[:, :, mask].sum())
                        computed = 100.0 * n_correct / n_total if n_total else math.nan
                        row = per.loc[(name, dist, target, mv, sign)]
                        if (row.family != family or row.alias != alias or int(row.seed) != seed
                                or int(row.probe_count) != n_probe or int(row.total) != n_total
                                or int(row.correct) != n_correct):
                            raise AssertionError('Conditional count/metadata mismatch')
                        if n_total:
                            max_difference = max(max_difference, abs(float(row.kci_percent)-computed))
                        elif not pd.isna(row.kci_percent):
                            raise AssertionError('Empty group wrongly assigned a KCI')
                        results[(family, alias, dist, target, mv, sign, seed)] = computed
                        direct_total += n_total
                        direct_correct += n_correct
                        if seed == 42 and old is not None:
                            prior = old.loc[(name, dist, target, mv, sign)]
                            if int(prior.correct) != n_correct or int(prior.total) != n_total:
                                raise AssertionError('Seed 42 differs from prior sign figure')
                            if n_total:
                                max_difference = max(max_difference, abs(float(prior.kci_percent)-computed))
                            elif not pd.isna(prior.kci_percent):
                                raise AssertionError('Old empty group differs')
                            old_checked += 1
                    o = official.loc[(name, dist, target, mv)]
                    if int(o.valid_comparisons) != direct_total or int(o.tp + o.tn) != direct_correct:
                        raise AssertionError('Mismatch against saved official KCI')
                    official_checked += 1
    for family, alias, dist, target, mv, sign in averages.index:
        row = averages.loc[(family, alias, dist, target, mv, sign)]
        vals = [results[(family, alias, dist, target, mv, sign, seed)] for seed in (42, 43, 44)]
        per_group = per.reset_index()
        selected_group = per_group[(per_group.family == family) & (per_group.alias == alias)
            & (per_group.distribution == dist) & (per_group.target == target)
            & (per_group.mv == mv) & (per_group.teacher_sign_label == sign)]
        if len(selected_group) != 3 or selected_group.probe_count.nunique() != 1:
            raise AssertionError('Probe counts differ among seeds')
        if int(row.probe_count) != int(selected_group.probe_count.iloc[0]):
            raise AssertionError('Aggregate probe count mismatch')
        valid_vals = [x for x in vals if not math.isnan(x)]
        if len(valid_vals) == 0:
            if int(row.n_seeds) != 0 or int(row.probe_count) != 0 or not pd.isna(row.mean_kci_percent) or not pd.isna(row.sd_kci_percentage_points):
                raise AssertionError('Empty group should stay blank')
        elif len(valid_vals) == 3:
            if int(row.n_seeds) != 3:
                raise AssertionError('Seed count mismatch')
            average, sd = statistics.mean(valid_vals), statistics.stdev(valid_vals)
            max_difference = max(max_difference, abs(float(row.mean_kci_percent)-average), abs(float(row.sd_kci_percentage_points)-sd))
        else:
            raise AssertionError('Unexpected missing seed')
    if official_checked != 120 or (old is not None and old_checked != 80) or max_difference > 1e-10:
        raise AssertionError('Incomplete or inaccurate verification')
    out = {'passed': True, 'models': 15, 'conditional_seed_groups': 240, 'mean_sd_groups': 80,
           'official_pair_counts_checked': official_checked, 'old_seed42_sign_groups_checked': old_checked,
           'old_seed42_reference_available': old is not None,
           'mean_and_sample_sd_recalculated_from_raw_npz': True, 'missing_groups': int(averages.n_seeds.eq(0).sum()),
           'max_absolute_percentage_point_difference': max_difference,
           'per_seed_csv_sha256': digest(per_path), 'mean_sd_csv_sha256': digest(avg_path),
           'audit_script_sha256': digest(Path(__file__))}
    (directory/'independent_three_seed_kci_audit.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
