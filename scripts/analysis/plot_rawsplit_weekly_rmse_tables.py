"""Make four weekly-report-style tables from audited rawsplit test results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/evaluation/rawsplit_test_rmse_vs_nogain42'
TEST = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'
OUT = SOURCE / 'weekly_tables'
SUMMARY = SOURCE / 'rmse_three_seed_summary_H1_H18_TS.csv'
FAMILIES = ('no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
CASES = {'internal': ['R5', 'R5-1', 'R5-2', 'R5-6'],
         'external': [f'R5-{i}' for i in range(19, 27)]}
HORIZONS = ('H1', 'H18', 'TS')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_data():
    prov = json.loads((SOURCE / 'provenance.json').read_text(encoding='utf8'))
    audit = json.loads((TEST / 'independent_npz_audit.json').read_text(encoding='utf8'))
    if not prov.get('passed') or not audit.get('passed') or audit.get('models') != 15:
        raise ValueError('The selected-test and comparison audits must pass')
    if sha(SUMMARY) != prov['output_sha256'][SUMMARY.name]:
        raise ValueError('The three-seed summary does not match its provenance')
    for name, digest in audit['source_sha256'].items():
        if sha(TEST / name) != digest:
            raise ValueError(f'Audited selected-test source changed: {name}')
    selected_path = TEST / 'selected_models.json'
    selected = json.loads(selected_path.read_text(encoding='utf8'))['selected_models']
    if {(j['family'], j['seed']) for j in selected} != {(f, s) for f in FAMILIES for s in (42, 43, 44)}:
        raise ValueError('Expected five families, each with three seeds')
    labels = {}
    for family in FAMILIES:
        family_jobs = [j for j in selected if j['family'] == family]
        if len({j['alias'] for j in family_jobs}) != 1:
            raise ValueError(f'Setting differs across seeds: {family}')
        weights = []
        for job in family_jobs:
            config = ROOT / job['config']
            if sha(config) != job['config_sha256']:
                raise ValueError(f'Config hash differs: {config}')
            cfg = yaml.safe_load(config.read_text(encoding='utf8'))['training']
            if family == 'no_gain': label = 'No-Gain RMSE\nMean ± SD'
            elif family == 'gain': label = f"Gain {float(cfg['gain_loss_weight']):g}\nΔ (%)"
            elif family == 'ss_only': label = f"SS-only {float(cfg['ss_loss_weight']):g}\nΔ (%)"
            elif family == 'gain_ss': label = f"Gain + SS {float(cfg['gain_loss_weight']):g}/{float(cfg['ss_loss_weight']):g}\nΔ (%)"
            else: label = f"DG ρ = {float(cfg['dg_target_ratio']):g}\nΔ (%)"
            weights.append(label)
        if len(set(weights)) != 1: raise ValueError(f'Labels differ across seeds: {family}')
        labels[family] = weights[0]
    frame = pd.read_csv(SUMMARY)
    if len(frame) != 360 or frame.duplicated(['partition', 'case', 'target', 'evaluation', 'family']).any():
        raise ValueError('Expected 360 unique three-seed summary values')
    if not frame['seeds'].eq(3).all() or not frame.baseline_rmse.gt(0).all():
        raise ValueError('Invalid seed count or No-Gain 42 baseline')
    if not np.isfinite(frame[['mean_rmse', 'sd_rmse', 'baseline_rmse', 'mean_delta_percent', 'sd_delta_percentage_points']]).all().all():
        raise ValueError('Nonfinite RMSE value')
    np.testing.assert_allclose(frame.mean_delta_percent, 100 * (frame.mean_rmse / frame.baseline_rmse - 1), rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(frame.sd_delta_percentage_points, 100 * frame.sd_rmse / frame.baseline_rmse, rtol=1e-9, atol=1e-8)
    for region, cases in CASES.items():
        for target in ('B35_H2S', 'B35_SO2'):
            sub=frame[(frame.partition == region) & (frame.target == target)]
            expected={(case, h, f) for case in cases for h in HORIZONS for f in FAMILIES}
            if set(zip(sub['case'], sub.evaluation, sub.family)) != expected:
                raise ValueError(f'Incorrect coverage for {region}/{target}')
    return frame, labels


def draw(frame: pd.DataFrame, labels: dict, region: str, target: str) -> None:
    cases=CASES[region]
    subset=frame[(frame.partition == region) & (frame.target == target)]
    # The same 16:9 canvas is convenient for adding each table to a weekly PPT.
    fig=plt.figure(figsize=(13.33, 7.5), facecolor='white')
    ax=fig.add_axes([0.015, 0.035, 0.97, 0.93]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    edges=np.array([0.0, .062, .133, .194, .305, .425, .536, .648, .773, 1.0])
    # The rightmost DG column is wider; each value remains centered.
    centers=(edges[:-1]+edges[1:])/2
    headers=['Interpolation\nOutput' if region=='internal' else 'Extrapolation\nOutput',
             'Case', 'Horizon', 'No-Gain 42 RMSE\n($\\times 10^{-3}$ kmol/kmol)', 'No-Gain RMSE Mean ± SD\n($\\times 10^{-3}$ kmol/kmol)',
             labels['gain'], labels['ss_only'], labels['gain_ss'], labels['difficult_gradnorm']]
    for x, label in zip(centers, headers):
        ax.text(x, .955, label, ha='center', va='center', fontsize=9.1, fontweight='bold', linespacing=1.05)
    ax.plot([0, 1], [.997, .997], color='#303030', lw=.8)
    ax.plot([0, 1], [.916, .916], color='#303030', lw=.8)
    nrows=3*len(cases); top=.900; bottom=.040; dy=(top-bottom)/nrows
    fs=9.1 if region=='internal' else 7.5
    ax.text(centers[0], (top+bottom)/2, 'H$_2$S' if target=='B35_H2S' else 'SO$_2$',
            ha='center', va='center', fontsize=fs+1)
    table_rows=[]
    for ci,case in enumerate(cases):
        ax.text(centers[1], top - (3*ci+1.5)*dy, case, ha='center', va='center', fontsize=fs)
        if ci:
            y=top-3*ci*dy
            ax.plot([edges[1],1], [y,y], color='#dddddd', lw=.35)
        for hi,horizon in enumerate(HORIZONS):
            y=top-(3*ci+hi+.5)*dy
            sub=subset[(subset['case']==case)&(subset.evaluation==horizon)].set_index('family').loc[list(FAMILIES)]
            baselines=sub.baseline_rmse.to_numpy()
            if not np.allclose(baselines, baselines[0], rtol=1e-12, atol=0): raise ValueError('Baseline mismatch')
            minimum=float(sub.mean_rmse.min())
            is_best=np.isclose(sub.mean_rmse.to_numpy(),minimum,rtol=1e-12,atol=1e-18)
            ax.text(centers[2],y,horizon,ha='center',va='center',fontsize=fs)
            ax.text(centers[3],y,f'{baselines[0]*1000:.4f}',ha='center',va='center',fontsize=fs)
            ng=sub.loc['no_gain']; text=f"{ng.mean_rmse*1000:.4f} ± {ng.sd_rmse*1000:.4f}"
            ax.text(centers[4],y,text,ha='center',va='center',fontsize=fs,fontweight='bold' if is_best[0] else 'normal')
            for idx,family in enumerate(FAMILIES[1:],5):
                row=sub.loc[family]; val=float(row.mean_delta_percent); sd=float(row.sd_delta_percentage_points)
                color='#b00020' if val > 1e-9 else '#007a28' if val < -1e-9 else '#222222'
                text=f'{val:+.1f} ± {sd:.1f}'
                ax.text(centers[idx],y,text,ha='center',va='center',fontsize=fs,color=color,fontweight='bold' if is_best[idx-4] else 'normal')
            for fi,family in enumerate(FAMILIES):
                row=sub.loc[family]
                table_rows.append({'partition':region,'target':target,'case':case,'evaluation':horizon,'family':family,
                    'baseline_rmse':baselines[0],'mean_rmse':row.mean_rmse,'sd_rmse':row.sd_rmse,
                    'mean_delta_percent':row.mean_delta_percent,'sd_delta_percentage_points':row.sd_delta_percentage_points,
                    'lowest_mean_rmse':bool(is_best[fi])})
    ax.plot([0,1],[bottom,bottom],color='#303030',lw=.8)
    stem=f'weekly_rmse_{region}_{"h2s" if target=="B35_H2S" else "so2"}'
    fig.savefig(OUT/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(OUT/f'{stem}.pdf',facecolor='white')
    plt.close(fig)
    caption=('Test RMSE comparison with the same-case, output, and horizon No-Gain seed 42. '
        'No-Gain 42 RMSE and No-Gain mean ± sample SD are in 10⁻³ kmol/kmol. '
        'Other columns show mean ± sample SD across seeds 42, 43, and 44 of Δ = 100 × (model RMSE / No-Gain 42 RMSE − 1); '
        'negative and positive values indicate lower and higher RMSE, respectively. '
        'Bold marks the lowest three-seed mean raw RMSE among the five model families for each case and horizon. '
        'H1 and H18 are rolling prediction horizons and TS is continuous time-series prediction. '
        'These previously used test cases are not a new blind test.\n')
    (OUT/f'{stem}_caption.txt').write_text(caption,encoding='utf8')
    return table_rows


def main() -> None:
    frame,labels=load_data()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42})
    values=[]
    for region in ('internal','external'):
        for target in ('B35_H2S','B35_SO2'):
            values.extend(draw(frame,labels,region,target))
    values=pd.DataFrame(values)
    if len(values)!=360 or values.duplicated(['partition','case','target','evaluation','family']).any():
        raise ValueError('Wrong table cell coverage')
    values.to_csv(OUT/'weekly_table_values.csv',index=False,float_format='%.17g')
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}
    meta={'passed':True,'table_count':4,'cell_rows':len(values),'baseline':'No-Gain seed 42',
          'seeds':[42,43,44], 'summary_sha256':sha(SUMMARY), 'comparison_provenance_sha256':sha(SOURCE/'provenance.json'),
          'selected_models_sha256':sha(TEST/'selected_models.json'), 'npz_audit_sha256':sha(TEST/'independent_npz_audit.json'),
          'script_sha256':sha(Path(__file__)), 'output_sha256':outputs}
    (OUT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'Created {meta["table_count"]} tables and {len(values)} audited values at {OUT}')


if __name__=='__main__':
    main()
