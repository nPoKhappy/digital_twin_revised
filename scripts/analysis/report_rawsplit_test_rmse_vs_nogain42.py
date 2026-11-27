"""Build a per-case test RMSE comparison against rawsplit No-Gain seed 42."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42_rawsplit'
OUT = ROOT / 'results/evaluation/rawsplit_test_rmse_vs_nogain42'
KEYS = ['partition', 'case', 'target', 'evaluation']
FAMILIES = ('no_gain', 'gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
SEEDS = (42, 43, 44)


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def label(family: str, cfg: dict) -> str:
    t = cfg['training']
    if family == 'no_gain': return 'No-Gain'
    if family == 'gain': return f"Gain = {float(t['gain_loss_weight']):g}"
    if family == 'ss_only': return f"SS-only = {float(t['ss_loss_weight']):g}"
    if family == 'gain_ss': return f"Gain + SS = {float(t['gain_loss_weight']):g} / {float(t['ss_loss_weight']):g}"
    if family == 'difficult_gradnorm': return f"Difficult-GradNorm, ρ = {float(t['dg_target_ratio']):g}"
    raise ValueError(f'Unexpected family {family}')


def build() -> None:
    audit = json.loads((SOURCE / 'independent_npz_audit.json').read_text(encoding='utf8'))
    if not audit.get('passed') or audit.get('models') != 15 or audit.get('rmse_rows') != 1080:
        raise ValueError('The selected-test independent audit is incomplete')
    for name, expected in audit['source_sha256'].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f'Selected-test audited output changed: {name}')
    protocol = json.loads((SOURCE / 'run_protocol.json').read_text(encoding='utf8'))
    for name, expected in protocol['sources_sha256'].items():
        if sha(ROOT / name).lower() != expected.lower():
            raise ValueError(f'Selected-test source changed: {name}')
    selections = json.loads((SOURCE / 'selected_models.json').read_text(encoding='utf8'))['selected_models']
    if len(selections) != 15 or {(x['family'], int(x['seed'])) for x in selections} != {(f, s) for f in FAMILIES for s in SEEDS}:
        raise ValueError('Expected one selected model per family and seed')
    labels = {}
    for family in FAMILIES:
        entries = [x for x in selections if x['family'] == family]
        if len({x['alias'] for x in entries}) != 1:
            raise ValueError(f'Selected setting differs across seeds: {family}')
        cfg_labels = []
        for entry in entries:
            cfg_path = ROOT / entry['config']
            if sha(cfg_path) != entry['config_sha256']:
                raise ValueError(f'Selected configuration changed: {cfg_path}')
            cfg_labels.append(label(family, yaml.safe_load(cfg_path.read_text(encoding='utf8'))))
        if len(set(cfg_labels)) != 1:
            raise ValueError(f'Inconsistent configuration labels: {family}')
        labels[family] = cfg_labels[0]

    standard = pd.read_csv(SOURCE / 'test_metrics.csv')
    horizons = pd.read_csv(SOURCE / 'horizon_metrics.csv')
    if len(standard) != 1080 or len(horizons) != 6480:
        raise ValueError('Unexpected test or horizon metrics coverage')
    if set(horizons.horizon) != set(range(1, 19)):
        raise ValueError('Expected every prediction horizon H1 through H18')
    horizons['evaluation'] = 'H' + horizons.horizon.astype(str)
    horizon_rows = horizons.drop(columns=['horizon'])
    m = pd.concat([horizon_rows, standard[standard.evaluation.eq('TS')]], ignore_index=True)
    # Existing H1/H18 test_metrics and corresponding horizon metrics must agree.
    duplicate_check = standard[standard.evaluation.isin(['H1', 'H18'])].merge(
        horizon_rows, on=['model'] + KEYS, suffixes=('_standard', '_horizon'), validate='one_to_one')
    if len(duplicate_check) != 720 or not duplicate_check.n_standard.eq(duplicate_check.n_horizon).all():
        raise ValueError('H1/H18 comparisons are incomplete or differ in sample count')
    np.testing.assert_allclose(duplicate_check.RMSE_standard, duplicate_check.RMSE_horizon, rtol=1e-12, atol=1e-15)
    existing = pd.read_csv(SOURCE / 'relative_rmse_components.csv')
    model_map = pd.DataFrame(selections)[['model', 'family', 'alias', 'seed', 'role']]
    if len(m) != 6840 or m.duplicated(['model'] + KEYS).any():
        raise ValueError('Test metrics are incomplete or duplicated')
    m = m.merge(model_map, on='model', validate='many_to_one')
    expected_cases = {'internal': {'R5', 'R5-1', 'R5-2', 'R5-6'},
                      'external': {f'R5-{i}' for i in range(19, 27)}}
    for partition, cases in expected_cases.items():
        subset = m[m.partition.eq(partition)]
        if set(subset['case']) != cases or len(subset) != 15 * len(cases) * 2 * 19:
            raise ValueError(f'Unexpected case coverage: {partition}')
    if set(m.target) != {'B35_H2S', 'B35_SO2'} or set(m.evaluation) != {f'H{i}' for i in range(1, 19)} | {'TS'}:
        raise ValueError('Unexpected outputs or horizons')
    baseline = m[m.model.eq('no_gain_seed42')][KEYS + ['n', 'RMSE']].rename(
        columns={'n': 'baseline_n', 'RMSE': 'baseline_rmse'})
    if len(baseline) != 456 or baseline.duplicated(KEYS).any():
        raise ValueError('No-Gain seed 42 baseline is not complete')
    d = m.merge(baseline, on=KEYS, validate='many_to_one')
    if len(d) != 6840 or not d.n.eq(d.baseline_n).all() or not d.baseline_rmse.gt(0).all():
        raise ValueError('Bad baseline alignment')
    if not np.isfinite(d[['RMSE', 'baseline_rmse']]).all().all() or not d.RMSE.ge(0).all():
        raise ValueError('Invalid RMSE')
    d['rmse_ratio'] = d.RMSE / d.baseline_rmse
    d['delta_percent'] = 100 * (d.rmse_ratio - 1)
    official = existing[['model'] + KEYS + ['relative_RMSE', 'NoGain42_RMSE']]
    check = d[d.evaluation.isin(['H1', 'H18', 'TS'])].merge(official, on=['model'] + KEYS, validate='one_to_one')
    if len(check) != 1080:
        raise ValueError('The published relative comparison is incomplete')
    np.testing.assert_allclose(check.rmse_ratio, check.relative_RMSE, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(check.baseline_rmse, check.NoGain42_RMSE, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(d.loc[d.model.eq('no_gain_seed42'), 'rmse_ratio'], 1)
    d = d[['partition', 'case', 'target', 'evaluation', 'n', 'family', 'alias', 'seed', 'role',
           'model', 'RMSE', 'baseline_rmse', 'baseline_n', 'rmse_ratio', 'delta_percent']]
    summary = d.groupby(KEYS + ['n', 'family', 'alias', 'baseline_rmse', 'baseline_n'],
                        as_index=False, sort=False).agg(
        seeds=('seed', 'nunique'), mean_rmse=('RMSE', 'mean'), sd_rmse=('RMSE', 'std'),
        mean_ratio=('rmse_ratio', 'mean'), sd_ratio=('rmse_ratio', 'std'),
        mean_delta_percent=('delta_percent', 'mean'), sd_delta_percentage_points=('delta_percent', 'std'))
    if len(summary) != 2280 or not summary.seeds.eq(3).all():
        raise ValueError('Every family/case/output/horizon needs three seeds')

    records = []
    for key, rows in d.groupby(KEYS, sort=False):
        record = dict(zip(KEYS, key))
        record['n'] = int(rows.n.iloc[0])
        record['baseline_rmse'] = float(rows.baseline_rmse.iloc[0])
        record['models'] = {}
        for family in FAMILIES:
            one = rows[rows.family.eq(family)].sort_values('seed')
            if list(one.seed) != list(SEEDS): raise ValueError('Missing seed in displayed row')
            record['models'][family] = {str(int(x.seed)): {'rmse': float(x.RMSE), 'delta': float(x.delta_percent)}
                                        for x in one.itertuples()}
        records.append(record)
    def case_number(case: str) -> int:
        return int(case.split('-')[1]) if '-' in case else 0
    def horizon_order(value: str) -> int:
        return 19 if value == 'TS' else int(value[1:])
    records.sort(key=lambda x: (x['partition'] != 'internal', case_number(x['case']), x['target'], horizon_order(x['evaluation'])))

    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / 'rmse_per_seed_all_horizons_and_ts.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    summary.to_csv(OUT / 'rmse_three_seed_summary_all_horizons_and_ts.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    d[d.evaluation.isin(['H1', 'H18', 'TS'])].to_csv(OUT / 'rmse_per_seed_H1_H18_TS.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    summary[summary.evaluation.isin(['H1', 'H18', 'TS'])].to_csv(OUT / 'rmse_three_seed_summary_H1_H18_TS.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    render_html(records, labels, OUT / 'rmse_comparison.html')
    readme = '''# Rawsplit test RMSE compared with No-Gain seed 42\n\nAll five validation-selected families and seeds 42, 43, 44 are included.\nThe baseline is the corresponding No-Gain seed 42 RMSE for the same partition, case, target and evaluation.\n`delta_percent = 100 * (RMSE / baseline_rmse - 1)`; negative means smaller RMSE.\nThe all-horizons-and-TS per-seed CSV has 6840 rows: 6480 for H1-H18 and 360 for TS.\nIts three-seed summary has 2280 groups. The separate H1/H18/TS files have 1080 rows and 360 groups.\nSummary values are arithmetic means and sample standard deviations across seeds (ddof=1).\nNo-Gain seed 43 and 44 are also compared against seed 42 and need not equal zero.\nRMSE values are in original output units (mole fraction, kmol/kmol).\nInternal test contains R5, R5-1, R5-2, R5-6; external contains R5-19 through R5-26.\nH1-H18 are horizon-specific; TS is the corresponding time-series evaluation.\nThese test cases were previously used and are not a new blind holdout.\nRebuild with `python -m scripts.analysis.report_rawsplit_test_rmse_vs_nogain42`.\n'''
    (OUT / 'README.md').write_text(readme, encoding='utf8')
    sources = {name: sha(SOURCE / name) for name in ('test_metrics.csv', 'horizon_metrics.csv', 'relative_rmse_components.csv',
              'selected_models.json', 'run_protocol.json', 'independent_npz_audit.json')}
    outputs = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'provenance.json'}
    (OUT / 'provenance.json').write_text(json.dumps({'passed': True, 'models': 15, 'per_seed_rows': len(d),
        'three_seed_groups': len(summary), 'comparison_rows': len(records), 'source_sha256': sources,
        'output_sha256': outputs, 'script_sha256': sha(Path(__file__))}, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'Built RMSE comparison at {OUT}: {len(d)} rows, {len(summary)} groups, {len(records)} cases/output/horizon rows')


def render_html(records: list[dict], labels: dict[str, str], path: Path) -> None:
    payload = json.dumps({'records': records, 'labels': labels, 'families': list(FAMILIES)}, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test RMSE vs No-Gain 42</title>
<style>:root{font-family:system-ui,"Microsoft JhengHei",sans-serif;color:#202a34;background:#f5f7fb}body{margin:0}main{max-width:1800px;margin:auto;padding:26px}h1{font-size:24px;margin:0 0 8px}p{line-height:1.6;color:#4c5d6c;margin:8px 0}.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:20px 0}.filters label{display:grid;gap:5px;font-weight:600;font-size:13px}select{padding:8px 12px;background:white;border:1px solid #b8c4cf;border-radius:6px;font:inherit}.wrap{overflow:auto;background:white;border:1px solid #dce4eb;border-radius:8px}table{border-collapse:collapse;min-width:1250px;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #e5eaf0;text-align:right;vertical-align:top}th{background:#eaf0f6;position:sticky;top:0;z-index:1}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}td strong{display:block;font-size:14px}td small{display:block;color:#5a6976;line-height:1.5;margin-top:3px}.better{color:#176847}.worse{color:#b34230}.same{color:#3d4c59}.tag{font-weight:700} .count{font-size:13px}.hint{font-size:13px}@media print{.filters{display:none}.wrap{overflow:visible}th{position:static}}</style></head><body><main>
<h1>Test RMSE：各模型相對 No-Gain seed 42</h1>
<p>每列比較同一案例、輸出及預測時域。百分比為 100 ×（模型 RMSE ÷ 基準 RMSE − 1）；<b>負值表示誤差降低</b>。原始 RMSE 為莫耳分率（kmol/kmol）。</p>
<p class="hint">三 seed 平均模式顯示平均 ± 樣本標準差；百分比 SD 單位為百分點。No-Gain seed 43／44 也以 seed 42 為基準。H1–H18 為各預測時域，TS 為時序評估。既有 test 已使用過，並非全新盲測。</p>
<div class="filters"><label>區域<select id="region"><option value="all">全部</option><option value="internal">內插</option><option value="external">外插</option></select></label><label>案例<select id="case"><option value="all">全部</option></select></label><label>輸出<select id="target"><option value="all">全部</option><option value="B35_H2S">H₂S</option><option value="B35_SO2">SO₂</option></select></label><label>時域<select id="horizon"><option value="all">全部</option></select></label><label>Seed<select id="seed"><option value="mean">三 seed 平均 ± SD</option><option value="42">42</option><option value="43">43</option><option value="44">44</option></select></label></div>
<p class="count" id="count"></p><div class="wrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div></main>
<script>const DATA=__DATA__; const $=id=>document.getElementById(id); const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const avg=x=>x.reduce((a,b)=>a+b,0)/x.length; const sd=x=>{let m=avg(x);return Math.sqrt(x.reduce((a,b)=>a+(b-m)**2,0)/(x.length-1))};
const pct=x=>(x>0?'+':'')+x.toFixed(2)+'%'; const number=x=>x.toExponential(3); const name=x=>x==='B35_H2S'?'H₂S':'SO₂';
for(let i=1;i<=18;i++)$('horizon').add(new Option('H'+i,'H'+i));$('horizon').add(new Option('TS','TS'));
$('head').innerHTML=['區域','案例','輸出','時域','樣本數 n','No-Gain 42 RMSE',...DATA.families.map(x=>DATA.labels[x])].map(x=>'<th>'+esc(x)+'</th>').join('');
function cases(){let region=$('region').value, sel=$('case'), prev=sel.value;let xs=[...new Set(DATA.records.filter(r=>region==='all'||r.partition===region).map(r=>r.case))];sel.innerHTML='<option value="all">全部</option>'+xs.map(x=>'<option>'+esc(x)+'</option>').join('');sel.value=xs.includes(prev)?prev:'all'}
function render(){let region=$('region').value,which=$('case').value,target=$('target').value,h=$('horizon').value,seed=$('seed').value;let rows=DATA.records.filter(r=>(region==='all'||r.partition===region)&&(which==='all'||r.case===which)&&(target==='all'||r.target===target)&&(h==='all'||r.evaluation===h));$('count').textContent='顯示 '+rows.length+' 組案例／輸出／時域';$('body').innerHTML=rows.map(r=>{let cells=DATA.families.map(f=>{let vals=Object.values(r.models[f]);let a=seed==='mean'?avg(vals.map(v=>v.delta)):r.models[f][seed].delta;let v=seed==='mean'?avg(vals.map(v=>v.rmse)):r.models[f][seed].rmse;let cls=a < -1e-9?'better':a>1e-9?'worse':'same';let delta=pct(a)+(seed==='mean'?' ± '+sd(vals.map(v=>v.delta)).toFixed(2)+' pp':'');let rmse=number(v)+(seed==='mean'?' ± '+number(sd(vals.map(v=>v.rmse))):'');return '<td><strong class="'+cls+'">'+delta+'</strong><small>RMSE '+rmse+'</small></td>'}).join('');return '<tr><td>'+(r.partition==='internal'?'內插':'外插')+'</td><td>'+esc(r.case)+'</td><td>'+name(r.target)+'</td><td>'+esc(r.evaluation)+'</td><td>'+r.n+'</td><td>'+number(r.baseline_rmse)+'</td>'+cells+'</tr>'}).join('')}
$('region').addEventListener('change',()=>{cases();render()});['case','target','horizon','seed'].forEach(id=>$(id).addEventListener('change',render));cases();render();</script></body></html>'''
    path.write_text(page.replace('__DATA__', payload), encoding='utf8')


if __name__ == '__main__':
    build()
