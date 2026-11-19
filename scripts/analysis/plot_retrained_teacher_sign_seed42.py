"""Plot conditional ANN-reference all18 KCI for the selected seed-42 models."""
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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
OUT = ROOT / 'results/visualization/gain_pair_difficulty_seed42_staging'
PAIRS = [('B35_H2S', 'air2_SP', 'Air2 → H₂S'), ('B35_H2S', 'HEATER2_output_T_SP', 'T2 → H₂S'),
         ('B35_SO2', 'air2_SP', 'Air2 → SO₂'), ('B35_SO2', 'HEATER2_output_T_SP', 'T2 → SO₂')]
TARGETS = ['B35_H2S', 'B35_SO2']
MVS = ['air2_SP', 'HEATER2_output_T_SP']
FAMILIES = ['no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm']
COLORS = ['#4d4d4d', '#0072B2', '#009E73', '#D55E00', '#CC79A7']
MARKERS = ['s', 'o', '^', 'D', 'v']
N_BOOT = 2000
THRESHOLD = 1e-5


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_label(family: str, alias: str) -> str:
    if family == 'no_gain':
        return 'No-Gain'
    if family == 'gain':
        return f"Gain = {alias.removeprefix('gain').replace('p', '.')}"
    if family == 'ss_only':
        return f"SS-only = {alias.removeprefix('ss').removesuffix('_only').replace('p', '.')}"
    if family == 'gain_ss':
        a, b = alias.split('_ss')
        return f"Gain + SS = {a.removeprefix('gain').replace('p', '.')} / {b.replace('p', '.')}"
    if family == 'difficult_gradnorm':
        return f"Difficult-GradNorm, ρ = {alias.removeprefix('dg_ratio').replace('p', '.')}"
    raise ValueError(f'Unexpected family {family}')


def main() -> None:
    if not json.loads((SOURCE/'evaluation_validation.json').read_text(encoding='utf8')).get('passed'):
        raise ValueError('Test evaluation has not passed')
    selection = json.loads((SOURCE/'selected_models.json').read_text(encoding='utf8'))['selected_models']
    models = [next(r for r in selection if r['family'] == family and r['seed'] == 42) for family in FAMILIES]
    if len(models) != 5 or len({r['model'] for r in models}) != 5:
        raise ValueError('Expected five unique seed-42 models')
    official = pd.read_csv(SOURCE/'kci_components.csv').set_index(['model', 'distribution', 'target', 'mv'], verify_integrity=True)
    protocol_sha = sha(SOURCE/'run_protocol.json')
    inputs = [Path(__file__), SOURCE/'selected_models.json', SOURCE/'run_protocol.json',
              SOURCE/'evaluation_validation.json', SOURCE/'kci_components.csv']
    input_hashes = {}
    arrays = {}
    for row in models:
        dest = SOURCE/'model_data'/row['model']
        complete = json.loads((dest/'complete.json').read_text(encoding='utf8'))
        if complete['protocol_sha256'] != protocol_sha:
            raise ValueError('Model protocol hash mismatch: '+row['model'])
        inputs.append(dest/'complete.json')
        for distribution, n in [('in_range', 4385), ('soft_ood', 2692)]:
            path = dest/f'gain_{distribution}.npz'
            if sha(path) != complete['outputs'][path.name]:
                raise ValueError(f'Input hash mismatch: {path}')
            inputs.append(path)
            with np.load(path) as z:
                teacher, dynamic = z['teacher_gain'].copy(), z['dynamic_gain'].copy()
            if teacher.shape != (n, 2, 2) or dynamic.shape != (2, 18, n, 2, 2):
                raise ValueError(f'Unexpected array shape: {path}')
            if not np.isfinite(teacher).all() or not np.isfinite(dynamic).all():
                raise ValueError(f'Nonfinite gain: {path}')
            arrays[(row['model'], distribution)] = (teacher, dynamic)
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    rows, probe_rows = [], []
    for model_index, row in enumerate(models):
        name = row['model']
        rng = np.random.default_rng(142 + model_index)
        for distribution, n in [('in_range', 4385), ('soft_ood', 2692)]:
            teacher, dynamic = arrays[(name, distribution)]
            ref_teacher, _ = arrays[(models[0]['model'], distribution)]
            if not np.array_equal(teacher, ref_teacher):
                raise ValueError(f'Teacher or probe order differs: {name}/{distribution}')
            for target, mv, pair in PAIRS:
                j, k = TARGETS.index(target), MVS.index(mv)
                t = teacher[:, j, k]
                g = dynamic[:, :, :, j, k]
                valid = np.abs(t) >= THRESHOLD
                # Zero dynamic gain is in the denominator and is never correct.
                correct_per_probe = ((g > 0) == (t[None, None, :] > 0)) & (g != 0)
                correct_per_probe = correct_per_probe.sum(axis=(0, 1))
                o = official.loc[(name, distribution, target, mv)]
                if int(o.valid_comparisons) != int(valid.sum()) * 36 or int(o.tp + o.tn) != int(correct_per_probe[valid].sum()):
                    raise ValueError(f'Official KCI count mismatch: {name}/{distribution}/{pair}')
                for sign in ('Negative', 'Positive'):
                    mask = valid & ((t < 0) if sign == 'Negative' else (t > 0))
                    ix = np.flatnonzero(mask)
                    counts = correct_per_probe[ix]
                    total = int(len(ix)*36)
                    count = int(counts.sum())
                    if len(ix):
                        scores = counts.astype(float) * (100.0/36.0)
                        means = []
                        for start in range(0, N_BOOT, 100):
                            sample = rng.integers(0, len(ix), size=(min(100, N_BOOT-start), len(ix)))
                            means.append(scores[sample].mean(axis=1))
                        low, high = map(float, np.quantile(np.concatenate(means), [0.025, 0.975]))
                        kci = 100.0 * count / total
                    else:
                        low = high = kci = float('nan')
                    rows.append(dict(model=name, family=row['family'], alias=row['alias'], seed=42,
                        distribution=distribution, pair=pair, target=target, mv=mv, teacher_sign_label=sign,
                        probe_count=len(ix), correct=count, total=total, kci_percent=kci,
                        ci95_low=low, ci95_high=high, bootstrap_resamples=N_BOOT,
                        bootstrap_rng_seed=142+model_index))
                    probe_rows.extend(dict(model=name, distribution=distribution, target=target, mv=mv,
                        teacher_sign_label=sign, probe_index=int(i), correct=int(c), total=36)
                        for i, c in zip(ix, counts))
    data, probe = pd.DataFrame(rows), pd.DataFrame(probe_rows)
    if len(data) != 80 or data.duplicated(['model', 'distribution', 'target', 'mv', 'teacher_sign_label']).any():
        raise ValueError('Invalid conditional group coverage')
    for _, s in data.iterrows():
        if s.probe_count and not (0 <= s.ci95_low <= s.ci95_high <= 100):
            raise ValueError('Invalid CI')
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT/'conditional_kci_by_teacher_sign.csv', index=False, float_format='%.17g')
    probe.to_csv(OUT/'gain_pair_probe_accuracy_seed42.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for ri, (distribution, region) in enumerate([('in_range', 'In-range'), ('soft_ood', 'Soft-OOD')]):
        for ci, (target, mv, pair) in enumerate(PAIRS):
            ax = axes[ri, ci]
            for m, row in enumerate(models):
                subset = data[(data.model == row['model']) & (data.distribution == distribution) &
                              (data.target == target) & (data.mv == mv)].set_index('teacher_sign_label').loc[['Negative', 'Positive']]
                x = np.arange(2, dtype=float)
                y = subset.kci_percent.to_numpy()
                ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], markersize=5,
                        linestyle=['--', '-', '-.', ':', '-'][m], linewidth=1.3,
                        label=model_label(row['family'], row['alias']))
            for si, sign in enumerate(('Negative', 'Positive')):
                group = data[(data.distribution == distribution) & (data.target == target) & (data.mv == mv) & (data.teacher_sign_label == sign)]
                if group.probe_count.eq(0).all():
                    ax.text(si, 5, 'n = 0', ha='center', color='#666666', fontsize=8)
            ax.set_xticks([0,1], ['Negative','Positive'])
            ax.set_xlim(-0.12, 1.12)
            ax.set_ylim(-3, 104)
            ax.set_yticks([0,20,40,60,80,100])
            ax.grid(axis='y', color='#dddddd', linewidth=.7)
            ax.spines[['top','right']].set_visible(False)
            if ri == 0:
                ax.set_title(pair, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(region + '\nAll18 KCI (%)')
            if ri == 1:
                ax.set_xlabel('ANN gain sign')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.942), ncol=5, frameon=False, fontsize=9)
    fig.suptitle('All18 KCI conditioned on ANN gain sign · seed 42', fontsize=14, fontweight='bold', y=.985)
    fig.subplots_adjust(left=.067, right=.985, top=.84, bottom=.105, wspace=.18, hspace=.32)
    fig.savefig(OUT/'conditional_kci_by_teacher_sign.png', dpi=300)
    fig.savefig(OUT/'conditional_kci_by_teacher_sign.pdf')
    plt.close(fig)
    with Image.open(OUT/'conditional_kci_by_teacher_sign.png') as im:
        im.convert('L').save(OUT/'conditional_kci_by_teacher_sign_grayscale.png')
    (OUT/'conditional_kci_by_teacher_sign_README.md').write_text('''# seed 42：ANN 增益正負號分組 KCI

圖與 CSV 對應新 validation 選定的 No-Gain、Gain 25、SS-only 2.5、Gain+SS 25/25、Difficult-GradNorm ρ=0.015，均為 seed 42。使用已保存的 test 預測陣列，沒有重訓或重新推論。

每個有效操作點的 ANN 增益絕對值至少為 1e-5，依正負分組；同一點的正、負擾動與 18 個預測步共 36 次比較保持一組。動態增益為零計入有效分母、視為不一致。圖中的點是使用全部有效操作點計算的 pooled all18 KCI，圖上不顯示誤差棒。CSV 保留以組內操作點抽樣 2,000 次算出的 95% percentile bootstrap 區間，供需要時查閱；這不是三個訓練 seed 的變異。

Air2 → SO₂ 在兩個區域都沒有 ANN 負增益的有效探點，故該分組標示 n = 0、KCI 和區間為空，不畫成 0%。CSV 有每組操作點數、正確數、比較數和區間；`gain_pair_probe_accuracy_seed42.csv` 可重算每點數據。門檻篩選前的穩態探點：in-range 4,385，soft-OOD 2,692；通過門檻的實際分組數見 CSV。

KCI 只表示與 ANN 參考增益方向的一致率，並不單獨證明 Aspen 物理增益方向正確。原 test 穩態探點先前已使用，並非全新的盲測。

此資料夾原有的 `conditional_kci_by_gain_magnitude.*`、`teacher_gain_landscape.*`、`teacher_gain_summary.csv` 與 `gain_magnitude_quartile_edges.csv` 是**舊模型與舊 ANN 的分析**，不屬於本次五模型結果，不可混用。重現本圖：

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.analysis.plot_retrained_teacher_sign_seed42
```

上述程式寫入 `results/visualization/gain_pair_difficulty_seed42_staging`，避免未經檢查便覆蓋正式圖。
''', encoding='utf8')
    if any(sha(ROOT/path) != digest for path,digest in input_hashes.items()):
        raise ValueError('Input changed during calculation')
    record = dict(passed=True, models=[r['model'] for r in models], model_seed=42,
        selection='Validation-selected hyperparameters; no training or new inference', figure_error_bars=False,
        distribution_probe_counts={'in_range':4385,'soft_ood':2692}, sign_groups=80,
        pairs_per_model=8, comparison_per_valid_probe=36, teacher_absolute_threshold=THRESHOLD,
        teacher_and_probe_order_identical=True, official_pair_counts_checked=40,
        bootstrap={'unit':'operating point within each sign group; all 36 comparisons remain grouped',
                   'resamples':N_BOOT,'ci':'percentile 2.5/97.5', 'rng':'numpy default_rng; integers sampled with replacement in batches of 100; each model has one generator',
                   'iteration_order':'in_range then soft_ood; Air2-H2S, T2-H2S, Air2-SO2, T2-SO2; Negative then Positive; empty groups consume no random numbers',
                   'rng_seed_by_model':{r['model']:142+i for i,r in enumerate(models)}},
        nonempty_group_count=int(data.probe_count.gt(0).sum()), empty_group_count=int(data.probe_count.eq(0).sum()),
        inputs_sha256=input_hashes,
        outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'conditional_kci_by_teacher_sign_provenance.json'})
    (OUT/'conditional_kci_by_teacher_sign_provenance.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'output':str(OUT),'rows':len(data),'empty_groups':record['empty_group_count'],'probe_rows':len(probe)}, indent=2))


if __name__ == '__main__':
    main()
