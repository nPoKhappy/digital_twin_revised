"""Plot per-sign ANN-reference KCI: means and sample SD across three seeds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from scripts.analysis.plot_retrained_teacher_sign_seed42 import (
    COLORS, FAMILIES, MARKERS, MVS, PAIRS, TARGETS, THRESHOLD, model_label,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42/updated_validation_best'
OUT = ROOT / 'results/visualization/gain_pair_updated_sign_staging'
SEEDS = (42, 43, 44)
DISTRIBUTIONS = (('in_range', 'In-range', 4385), ('soft_ood', 'Soft-OOD', 2692))


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    validation = json.loads((SOURCE/'evaluation_validation.json').read_text(encoding='utf8'))
    if not validation.get('passed') or validation.get('model_count') != 15:
        raise ValueError('15-model test evaluation has not passed')
    protocol = json.loads((SOURCE/'run_protocol.json').read_text(encoding='utf8'))
    selection_path = SOURCE/'selected_models.json'
    plan_path = SOURCE/'validation_selection_plan.json'
    if protocol['selection_sha256'] != sha(selection_path) or protocol['selection_plan_sha256'] != sha(plan_path):
        raise ValueError('Test selection or validation plan has changed')
    for relative, expected in protocol['sources_sha256'].items():
        path = ROOT/relative
        if not path.is_file() or sha(path).lower() != expected.lower():
            raise ValueError(f'Protected test source differs: {relative}')
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    expected = {'gain':'gain25', 'ss_only':'ss2p5_only', 'gain_ss':'gain25_ss25',
                'difficult_gradnorm':'dg_ratio1p5'}
    if plan['aliases'] != expected:
        raise ValueError('Unexpected current validation-selected model settings')
    selection = json.loads(selection_path.read_text(encoding='utf8'))['selected_models']
    official = pd.read_csv(SOURCE/'kci_components.csv').set_index(
        ['model', 'distribution', 'target', 'mv'], verify_integrity=True)
    models = []
    for family in FAMILIES:
        selected = [r for r in selection if r['family'] == family]
        if len(selected) != 3 or {r['seed'] for r in selected} != set(SEEDS) or len({r['alias'] for r in selected}) != 1:
            raise ValueError(f'Expected one configuration with all three seeds: {family}')
        models.extend(sorted(selected, key=lambda row: row['seed']))
    protocol_sha = sha(SOURCE/'run_protocol.json')
    inputs = [Path(__file__), ROOT/'scripts/analysis/plot_retrained_teacher_sign_seed42.py',
              SOURCE/'selected_models.json', SOURCE/'run_protocol.json',
              SOURCE/'evaluation_validation.json', SOURCE/'kci_components.csv', SOURCE/'validation_selection_plan.json']
    rows = []
    reference = {}
    for row in models:
        dest = SOURCE/'model_data'/row['model']
        complete = json.loads((dest/'complete.json').read_text(encoding='utf8'))
        if complete['protocol_sha256'] != protocol_sha:
            raise ValueError(f'Protocol mismatch: {row["model"]}')
        inputs.append(dest/'complete.json')
        for distribution, _, n in DISTRIBUTIONS:
            path = dest/f'gain_{distribution}.npz'
            if sha(path) != complete['outputs'][path.name]:
                raise ValueError(f'Prediction hash mismatch: {path}')
            inputs.append(path)
            with np.load(path) as z:
                teacher, dynamic = z['teacher_gain'].copy(), z['dynamic_gain'].copy()
            if teacher.shape != (n, 2, 2) or dynamic.shape != (2, 18, n, 2, 2):
                raise ValueError(f'Unexpected gain shape: {path}')
            if not np.isfinite(teacher).all() or not np.isfinite(dynamic).all():
                raise ValueError(f'Nonfinite gain: {path}')
            if distribution in reference:
                if not np.array_equal(teacher, reference[distribution]):
                    raise ValueError(f'Teacher or probe order differs: {path}')
            else:
                reference[distribution] = teacher
            for target, mv, pair in PAIRS:
                j, k = TARGETS.index(target), MVS.index(mv)
                t = teacher[:, j, k]
                g = dynamic[:, :, :, j, k]
                valid = np.abs(t) >= THRESHOLD
                per_probe = (((g > 0) == (t[None, None, :] > 0)) & (g != 0)).sum(axis=(0, 1))
                o = official.loc[(row['model'], distribution, target, mv)]
                if int(o.valid_comparisons) != int(valid.sum())*36 or int(o.tp+o.tn) != int(per_probe[valid].sum()):
                    raise ValueError(f'Official KCI mismatch: {row["model"]}/{distribution}/{pair}')
                for sign in ('Negative', 'Positive'):
                    mask = valid & ((t < 0) if sign == 'Negative' else (t > 0))
                    n_probe = int(mask.sum())
                    count, total = int(per_probe[mask].sum()), 36*n_probe
                    rows.append(dict(model=row['model'], family=row['family'], alias=row['alias'], seed=row['seed'],
                                     distribution=distribution, pair=pair, target=target, mv=mv,
                                     teacher_sign_label=sign, probe_count=n_probe, correct=count, total=total,
                                     kci_percent=100*count/total if total else np.nan))
    data = pd.DataFrame(rows)
    keys = ['family', 'alias', 'distribution', 'pair', 'target', 'mv', 'teacher_sign_label']
    if len(data) != 240 or data.duplicated(['model', 'distribution', 'target', 'mv', 'teacher_sign_label']).any():
        raise ValueError('Unexpected group coverage')
    stats = data.groupby(keys, as_index=False, dropna=False).agg(
        n_seeds=('kci_percent', 'count'), probe_count=('probe_count', 'first'),
        mean_kci_percent=('kci_percent', 'mean'), sd_kci_percentage_points=('kci_percent', 'std'))
    if len(stats) != 80 or not stats.n_seeds.isin([0, 3]).all():
        raise ValueError('Three-seed coverage or empty group mismatch')
    if data.groupby(keys).probe_count.nunique().ne(1).any():
        raise ValueError('Grouped probe counts differ across seeds')
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    OUT.mkdir(parents=True, exist_ok=False)
    data.to_csv(OUT/'conditional_kci_by_teacher_sign_per_seed.csv', index=False, float_format='%.17g')
    stats.to_csv(OUT/'conditional_kci_by_teacher_sign.csv', index=False, float_format='%.17g')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for ri, (distribution, region, _) in enumerate(DISTRIBUTIONS):
        for ci, (target, mv, pair) in enumerate(PAIRS):
            ax = axes[ri, ci]
            for m, family in enumerate(FAMILIES):
                row = next(r for r in models if r['family'] == family)
                s = stats[(stats.family == family) & (stats.distribution == distribution) &
                          (stats.target == target) & (stats.mv == mv)].set_index('teacher_sign_label').loc[['Negative', 'Positive']]
                x, y = np.arange(2, dtype=float), s.mean_kci_percent.to_numpy()
                ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], markersize=5,
                        linestyle=['--', '-', '-.', ':', '-'][m], linewidth=1.3,
                        label=model_label(family, row['alias']))
                for xx, (_, point) in zip(x, s.iterrows()):
                    if point.n_seeds == 3:
                        ax.vlines(xx, point.mean_kci_percent-point.sd_kci_percentage_points,
                                  point.mean_kci_percent+point.sd_kci_percentage_points,
                                  color=COLORS[m], linewidth=1.1)
                        ax.hlines([point.mean_kci_percent-point.sd_kci_percentage_points,
                                   point.mean_kci_percent+point.sd_kci_percentage_points],
                                  xx-.025, xx+.025, color=COLORS[m], linewidth=1)
            for si, sign in enumerate(('Negative', 'Positive')):
                group = stats[(stats.distribution == distribution) & (stats.target == target) &
                              (stats.mv == mv) & (stats.teacher_sign_label == sign)]
                if group.n_seeds.eq(0).all():
                    ax.text(si, 4, 'n = 0', ha='center', color='#666666', fontsize=8)
            ax.set_xticks([0, 1], ['Negative', 'Positive'])
            ax.set_xlim(-.12, 1.12)
            finite = stats[stats.n_seeds.eq(3)]
            ymin = min(-3, float((finite.mean_kci_percent-finite.sd_kci_percentage_points).min())-2)
            ymax = max(104, float((finite.mean_kci_percent+finite.sd_kci_percentage_points).max())+2)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis='y', color='#dddddd', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
            if ri == 0:
                ax.set_title(pair, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(region + '\nAll18 KCI (%)')
            if ri == 1:
                ax.set_xlabel('ANN gain sign')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=5, frameon=False, fontsize=9)
    fig.suptitle('All18 KCI conditioned on ANN gain sign · seeds 42, 43, 44', fontsize=14, fontweight='bold', y=.985)
    fig.text(.5, .018, 'Points: three-seed mean. Bars: ±1 sample SD across model training seeds (n = 3).', ha='center', fontsize=9)
    fig.subplots_adjust(left=.067, right=.985, top=.84, bottom=.105, wspace=.18, hspace=.32)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT/f'conditional_kci_by_teacher_sign.{suffix}', dpi=300)
    plt.close(fig)
    with Image.open(OUT/'conditional_kci_by_teacher_sign.png') as im:
        im.convert('L').save(OUT/'conditional_kci_by_teacher_sign_grayscale.png')
    (OUT/'conditional_kci_by_teacher_sign_README.md').write_text(
        '# Three-seed ANN gain sign KCI\n\n'
        'Models: No-Gain, Gain 25, SS-only 2.5, Gain+SS 25/25, DG rho=1.5. '
        'Points are the mean KCI across training seeds 42, 43, 44; bars are one sample SD (ddof=1), not bootstrap intervals. '
        'The three seeds share the ANN and validation/test operating points.\n\n'
        'Each valid point has absolute ANN gain at least 1e-5 and contributes 36 comparisons (two step directions by 18 prediction steps). '
        'Dynamic zero gain counts as incorrect. Air2 to SO2 has no negative ANN-gain probes in either region; these groups are missing, not zero. '
        'The plotted statistics retain all valid points. KCI measures agreement to the ANN reference, not direct physical gain validation. '
        'The test probes have been used previously; they are not a fresh blind holdout.\n\n'
        'This figure uses the updated validation selection. Some historical validation source files were deleted, '
        'so the full historical provenance is unavailable; current test NPZ files and their remaining source hashes have been verified. '
        'The folder name includes seed42 but this figure uses three seeds. '
        'Existing teacher landscape and legacy seed42-only data in the target folder are from earlier analysis.\n\n'
        'Rebuild: python -m scripts.analysis.plot_updated_test_teacher_sign_three_seeds\n', encoding='utf8')
    if any(sha(ROOT/path) != digest for path, digest in input_hashes.items()):
        raise ValueError('An input changed while computing')
    record = dict(passed=True, models=[r['model'] for r in models], seeds=list(SEEDS),
                  selection='Current validation-selected families including DG rho=1.5; no training or new inference',
                  full_historical_provenance_available=bool(validation.get('full_historical_provenance_available')),
                  uncertainty='arithmetic mean ± 1 sample SD across 3 model seeds, ddof=1; not bootstrap CI',
                  threshold=THRESHOLD, group_count=len(stats), per_seed_rows=len(data),
                  empty_groups=int(stats.n_seeds.eq(0).sum()), input_sha256=input_hashes,
                  output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'conditional_kci_by_teacher_sign_provenance.json'})
    (OUT/'conditional_kci_by_teacher_sign_provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'output':str(OUT), 'groups':len(stats), 'per_seed_rows':len(data),
                      'empty_groups':record['empty_groups']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
