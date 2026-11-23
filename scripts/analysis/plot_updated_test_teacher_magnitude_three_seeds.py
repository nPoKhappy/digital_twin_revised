"""Plot ANN-gain magnitude-stratified KCI using three training seeds."""
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
OUT = ROOT / 'results/visualization/gain_magnitude_updated_test_staging'
SEEDS = (42, 43, 44)
DISTRIBUTIONS = (('in_range', 'In-range', 4385), ('soft_ood', 'Soft-OOD', 2692))
QUARTILES = ('Q1', 'Q2', 'Q3', 'Q4')
TEACHER_SCALE = ROOT/'results/steady_state/Tabular_MLP_New/zscore_std.csv'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    validation = json.loads((SOURCE/'evaluation_validation.json').read_text(encoding='utf8'))
    if not validation.get('passed') or validation.get('model_count') != 15:
        raise ValueError('15-model test evaluation has not passed')
    protocol = json.loads((SOURCE/'run_protocol.json').read_text(encoding='utf8'))
    if sha(SOURCE/'selected_models.json') != protocol['selection_sha256'] or sha(SOURCE/'validation_selection_plan.json') != protocol['selection_plan_sha256']:
        raise ValueError('Selection hash chain differs from test protocol')
    for path, digest in protocol['sources_sha256'].items():
        if not (ROOT/path).is_file() or sha(ROOT/path) != digest:
            raise ValueError(f'Changed or missing protected test source: {path}')
    audit = json.loads((SOURCE/'independent_npz_audit.json').read_text(encoding='utf8'))
    if not audit.get('passed'):
        raise ValueError('Independent NPZ verification has not passed')
    selection = json.loads((SOURCE/'selected_models.json').read_text(encoding='utf8'))['selected_models']
    plan = json.loads((SOURCE/'validation_selection_plan.json').read_text(encoding='utf8'))
    if set(plan['aliases']) != set(FAMILIES) - {'no_gain'}:
        raise ValueError('Unexpected validation family coverage')
    official = pd.read_csv(SOURCE/'kci_components.csv').set_index(
        ['model', 'distribution', 'target', 'mv'], verify_integrity=True)
    scales = pd.read_csv(TEACHER_SCALE, index_col=0).iloc[:, 0]
    if not all(np.isfinite(float(scales[name])) and float(scales[name]) > 0 for name in (*TARGETS, *MVS)):
        raise ValueError('Teacher normalization scales must be finite and positive')
    models = []
    for family in FAMILIES:
        selected = [r for r in selection if r['family'] == family]
        if len(selected) != 3 or {r['seed'] for r in selected} != set(SEEDS) or len({r['alias'] for r in selected}) != 1:
            raise ValueError(f'Expected one configuration with all three seeds: {family}')
        if family != 'no_gain' and selected[0]['alias'] != plan['aliases'][family]:
            raise ValueError(f'Selection differs from validation plan: {family}')
        models.extend(sorted(selected, key=lambda row: row['seed']))
    protocol_sha = sha(SOURCE/'run_protocol.json')
    inputs = [Path(__file__), ROOT/'scripts/analysis/plot_retrained_teacher_sign_seed42.py',
              SOURCE/'selected_models.json', SOURCE/'run_protocol.json',
              SOURCE/'evaluation_validation.json', SOURCE/'kci_components.csv', SOURCE/'validation_selection_plan.json',
              SOURCE/'independent_npz_audit.json', TEACHER_SCALE]
    rows, edge_rows, reference, bins = [], [], {}, {}
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
                key = (distribution, target, mv)
                if key not in bins:
                    values = np.abs(t[valid]).astype(np.float64)
                    normalized = values * float(scales[mv]) / float(scales[target])
                    raw_edges = np.quantile(values, [0.25, 0.50, 0.75])
                    edges = np.quantile(normalized, [0.25, 0.50, 0.75])
                    if len(values) < 4 or np.any(np.diff(edges) <= 0):
                        raise ValueError(f'Insufficient data or duplicate quartile edges: {key}')
                    assignments = np.full(len(t), -1, dtype=np.int8)
                    assignments[valid] = np.searchsorted(edges, normalized, side='right')
                    bins[key] = assignments
                    edge_rows.append(dict(distribution=distribution, target=target, mv=mv, pair=pair,
                        valid_probe_count=int(valid.sum()), q25_abs_raw_gain=raw_edges[0],
                        q50_abs_raw_gain=raw_edges[1], q75_abs_raw_gain=raw_edges[2],
                        q25_abs_normalized_gain=edges[0], q50_abs_normalized_gain=edges[1], q75_abs_normalized_gain=edges[2],
                        **{f'{q}_probe_count':int((assignments == i).sum()) for i,q in enumerate(QUARTILES)}))
                for i, quartile in enumerate(QUARTILES):
                    mask = bins[key] == i
                    n_probe = int(mask.sum())
                    count, total = int(per_probe[mask].sum()), 36*n_probe
                    rows.append(dict(model=row['model'], family=row['family'], alias=row['alias'], seed=row['seed'],
                        distribution=distribution, pair=pair, target=target, mv=mv, quartile=quartile,
                        probe_count=n_probe, correct=count, total=total,
                        kci_percent=100*count/total if total else np.nan))
    data = pd.DataFrame(rows)
    keys = ['family', 'alias', 'distribution', 'pair', 'target', 'mv', 'quartile']
    if len(data) != 480 or data.duplicated(['model', 'distribution', 'target', 'mv', 'quartile']).any():
        raise ValueError('Unexpected group coverage')
    stats = data.groupby(keys, as_index=False, dropna=False).agg(
        n_seeds=('kci_percent', 'count'), probe_count=('probe_count', 'first'),
        mean_kci_percent=('kci_percent', 'mean'), sd_kci_percentage_points=('kci_percent', 'std'))
    if len(stats) != 160 or not stats.n_seeds.eq(3).all():
        raise ValueError('Unexpected three-seed coverage')
    if data.groupby(keys).probe_count.nunique().ne(1).any():
        raise ValueError('Grouped probe counts differ across seeds')
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    OUT.mkdir(parents=True, exist_ok=False)
    data.to_csv(OUT/'conditional_kci_by_gain_magnitude_per_seed.csv', index=False, float_format='%.17g')
    stats.to_csv(OUT/'conditional_kci_by_gain_magnitude.csv', index=False, float_format='%.17g')
    pd.DataFrame(edge_rows).to_csv(OUT/'gain_magnitude_quartile_edges.csv', index=False, float_format='%.17g')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    finite = stats[stats.n_seeds.eq(3)]
    ymin = min(-3, float((finite.mean_kci_percent-finite.sd_kci_percentage_points).min())-2)
    ymax = max(104, float((finite.mean_kci_percent+finite.sd_kci_percentage_points).max())+2)
    for ri, (distribution, region, _) in enumerate(DISTRIBUTIONS):
        for ci, (target, mv, pair) in enumerate(PAIRS):
            ax = axes[ri, ci]
            for m, family in enumerate(FAMILIES):
                row = next(r for r in models if r['family'] == family)
                s = stats[(stats.family == family) & (stats.distribution == distribution) &
                    (stats.target == target) & (stats.mv == mv)].set_index('quartile').loc[list(QUARTILES)]
                x, y = np.arange(4, dtype=float), s.mean_kci_percent.to_numpy()
                ax.errorbar(x, y, yerr=s.sd_kci_percentage_points.to_numpy(), color=COLORS[m], marker=MARKERS[m],
                    markersize=5, linestyle=['--', '-', '-.', ':', '-'][m], linewidth=1.3, capsize=2.5,
                    label=model_label(family, row['alias']))
            ax.set_xticks(range(4), list(QUARTILES))
            ax.set_xlim(-.15, 3.15)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis='y', color='#dddddd', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
            if ri == 0:
                ax.set_title(pair, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(region + '\nAll18 KCI (%)')
            if ri == 1:
                ax.set_xlabel('ANN |gain| quartile')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=5, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.067, right=.985, top=.84, bottom=.105, wspace=.18, hspace=.32)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT/f'conditional_kci_by_gain_magnitude.{suffix}', dpi=300)
    plt.close(fig)
    with Image.open(OUT/'conditional_kci_by_gain_magnitude.png') as im:
        im.convert('L').save(OUT/'conditional_kci_by_gain_magnitude_grayscale.png')
    (OUT/'conditional_kci_by_gain_magnitude_README.md').write_text(f'''# 三個訓練 seed 的 ANN 增益大小分組 KCI

圖上各點為 seed 42、43、44 的 KCI 算術平均，誤差棒為三個 seed 的樣本標準差（mean ± 1 SD，n=3，ddof=1），不是 bootstrap 信賴區間。三個動態模型共用相同 ANN 與相同 test 穩態探點；標準差描述動態訓練 seed 的變異。新版 validation 選模：{plan['aliases']}，另含 No-Gain。

每個區域、輸出與 MV 配對分別先篩選 |ANN 增益| ≥ 1e-5 的有效操作點，再依 |ANN 增益| 的 25%、50%、75% 分位數分為 Q1–Q4。沿用舊圖方法，以已儲存 teacher scaler 的 σ_MV / σ_target 標準化增益絕對值，使用 searchsorted side=right；若與邊界相等，分配至較高分組。界線對所有模型與 seed 相同；原始增益絕對值與正的標準化倍率在同一配對中排序相同。每個操作點合併正、負擾動及各 18 預測步共 36 次比較，動態增益為零計為方向不一致；每 seed 對各組全部有效點計算 KCI。逐 seed 數值見 `conditional_kci_by_gain_magnitude_per_seed.csv`，均值和 SD 見 `conditional_kci_by_gain_magnitude.csv`，分位數及各組點數見 `gain_magnitude_quartile_edges.csv`。

mean ± SD 可延伸至 0% 以下或 100% 以上，但實際 KCI 都在 0–100%。test 穩態探點先前已使用，並非全新盲測。KCI 是對 ANN 參考方向的一致率，不能單憑它驗證 Aspen 真實增益方向。前次 validation 歷史備份五個來源已刪除，選模與 test protocol 記錄此追溯限制；仍存的來源及預測陣列已核對。

重建：`.\\.venv\\Scripts\\python.exe -m scripts.analysis.plot_updated_test_teacher_magnitude_three_seeds`，預設輸出 staging 資料夾供覆核。
''', encoding='utf8')
    if any(sha(ROOT/path) != digest for path, digest in input_hashes.items()):
        raise ValueError('An input changed while computing')
    record = dict(passed=True, models=[r['model'] for r in models], seeds=list(SEEDS),
        selection=f'Updated validation-selected settings {plan["aliases"]}; 12 reused and 3 newly evaluated models; no new inference for this plot',
        source_limit='Prior validation historical backup five files unavailable; see source test selection and protocol for exact scope',
        uncertainty='arithmetic mean ± 1 sample SD across 3 model seeds, ddof=1; not bootstrap CI',
        quartiles='valid |ANN gain| normalized by teacher std[mv]/std[target] within each region and output-MV pair; 25/50/75 percentiles; searchsorted side=right; shared across models and seeds',
        threshold=THRESHOLD, group_count=len(stats), per_seed_rows=len(data),
        input_sha256=input_hashes,
        output_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'conditional_kci_by_gain_magnitude_provenance.json'})
    (OUT/'conditional_kci_by_gain_magnitude_provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'output':str(OUT), 'groups':len(stats), 'per_seed_rows':len(data)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
