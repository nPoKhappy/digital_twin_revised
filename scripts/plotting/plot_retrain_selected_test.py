"""Render fixed-baseline test results for validation-selected models.

Inputs are immutable completed test outputs. Each family retains all three
seed results; means and sample standard deviations are descriptive summaries.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
OUT = SOURCE / 'comparison_report'
VALIDATION = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline/scores_by_model_seed.csv'
COLORS = {'gain': '#0072B2', 'ss_only': '#D55E00', 'gain_ss': '#009E73',
          'difficult_gradnorm': '#9144A5', 'no_gain': '#333333'}
FAMILY = {'gain': 'Gain = 25', 'ss_only': 'SS-only = 2.5',
          'gain_ss': 'Gain+SS = 2.5 / 2.5',
          'difficult_gradnorm': 'Difficult-GradNorm ρ = 0.015'}
MARKERS = {42: 'o', 43: 's', 44: '^'}


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load():
    passed = json.loads((SOURCE/'evaluation_validation.json').read_text(encoding='utf-8'))
    if not (passed.get('passed') and passed.get('model_count') == 14
            and passed.get('cases') == 12 and passed.get('protected_sources_unchanged')):
        raise ValueError('Completed 14-model test evaluation is required.')
    selection = json.loads((SOURCE/'selected_models.json').read_text(encoding='utf-8'))['selected_models']
    roles = {x['model']: x for x in selection}
    if len(roles) != 14:
        raise ValueError('Selection must contain 14 unique trained models.')
    data = pd.read_csv(SOURCE/'regional_coordinates.csv')
    expected = {(m,p) for m in roles for p in ('internal', 'external')}
    keys = list(zip(data.model, data.partition))
    if len(data) != 28 or len(set(keys)) != 28 or set(keys) != expected:
        raise ValueError('Missing or duplicate test model/region coordinate.')
    for row in data.itertuples(index=False):
        model = roles[row.model]
        if any(getattr(row, k) != model[k] for k in ('family','alias','seed','role')):
            raise ValueError(f'Selection metadata mismatch: {row.model}')
        if not (np.isfinite(row.AvgRelRMSE) and row.AvgRelRMSE > 0 and row.valid > 0
                and 0 <= row.correct <= row.valid and np.isclose(row.KCI_percent, 100*row.correct/row.valid)):
            raise ValueError(f'Invalid test coordinate: {row.model}')
    baseline = data[data.role.eq('fixed_reference')]
    if len(baseline) != 2 or not np.allclose(baseline.AvgRelRMSE, 1):
        raise ValueError('No-Gain seed 42 must be 1 in both regions.')
    parts = pd.read_csv(SOURCE/'relative_rmse_components.csv')
    if len(parts) != 1008 or not np.isfinite(parts.relative_RMSE).all() or not parts.relative_RMSE.gt(0).all():
        raise ValueError('Expected 1008 positive finite matched RMSE ratios.')
    if not np.allclose(parts.relative_RMSE, parts.RMSE/parts.NoGain42_RMSE, rtol=1e-12, atol=1e-12):
        raise ValueError('An RMSE ratio differs from the common baseline.')
    metrics = pd.read_csv(SOURCE/'test_metrics.csv')
    keys = ['partition', 'case', 'target', 'evaluation']
    baseline_metrics = metrics[metrics.model.eq('no_gain_seed42')][keys + ['RMSE']].rename(
        columns={'RMSE': 'actual_baseline_rmse'})
    matched = parts.merge(baseline_metrics, on=keys, validate='many_to_one')
    matched = matched.merge(metrics[['model'] + keys + ['RMSE']], on=['model'] + keys,
                            suffixes=('', '_actual'), validate='one_to_one')
    if (len(matched) != 1008
            or not np.allclose(matched.NoGain42_RMSE, matched.actual_baseline_rmse, rtol=1e-12, atol=1e-12)
            or not np.allclose(matched.RMSE, matched.RMSE_actual, rtol=1e-12, atol=1e-12)):
        raise ValueError('RMSE components do not match the actual model and No-Gain 42 metrics.')
    regional = parts.groupby(['model','partition'],as_index=False).agg(
        recomputed=('relative_RMSE',lambda s:float(np.exp(np.log(s).mean()))), n=('relative_RMSE','size'))
    checked = data.merge(regional,on=['model','partition'],validate='one_to_one')
    if (len(checked) != 28 or not checked.n.eq(checked.components).all()
            or not np.allclose(checked.AvgRelRMSE, checked.recomputed, rtol=1e-12, atol=1e-12)):
        raise ValueError('Regional RMSE does not match matched ratios.')
    kci = pd.read_csv(SOURCE/'kci_components.csv')
    if len(kci) != 112:
        raise ValueError('Expected eight KCI components per model.')
    counts = kci.assign(correct=kci.tp+kci.tn).groupby(['model','distribution'],as_index=False).agg(
        correct=('correct','sum'),valid=('valid_comparisons','sum'))
    counts['partition'] = counts.distribution.map({'in_range':'internal','soft_ood':'external'})
    check = data.merge(counts,on=['model','partition'],suffixes=('_csv','_recomputed'),validate='one_to_one')
    if (len(check)!=28 or not check.correct_csv.eq(check.correct_recomputed).all()
            or not check.valid_csv.eq(check.valid_recomputed).all()):
        raise ValueError('KCI coordinate counts differ from components.')
    main = data[data.role.eq('three_seed_mean_winner')]
    if len(main) != 24 or main.groupby(['family','alias','partition']).seed.nunique().ne(3).any():
        raise ValueError('The four methods need three individual seeds per region.')
    if main.groupby(['family','partition']).alias.nunique().ne(1).any():
        raise ValueError('Multiple hyperparameters cannot be averaged as one method.')
    if set(main.family) != set(FAMILY):
        raise ValueError('Unexpected method families.')
    expected_aliases = {'gain':'gain25', 'ss_only':'ss2p5_only',
                        'gain_ss':'gain2p5_ss2p5', 'difficult_gradnorm':'dg_ratio0p015'}
    if any(set(main[main.family.eq(family)].alias) != {alias}
           for family, alias in expected_aliases.items()):
        raise ValueError('Model aliases do not match the plot labels.')
    extra = data[data.role.eq('supplementary_single_run_validation_winner')]
    if len(extra) != 2 or set(extra.alias) != {'gain2p5'} or set(extra.seed) != {44}:
        raise ValueError('Expected separately labelled Gain 2.5 seed 44.')
    summary = main.groupby(['family','alias','partition'],as_index=False).agg(
        KCI_mean=('KCI_percent','mean'), KCI_sd=('KCI_percent','std'),
        AvgRelRMSE_mean=('AvgRelRMSE','mean'), AvgRelRMSE_sd=('AvgRelRMSE','std'),
        n=('seed','nunique'))
    if len(summary) != 8 or not summary.n.eq(3).all():
        raise ValueError('Expected four methods, two regions and three seeds each.')
    return data, summary, selection


def plot(data, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13,5.6), sharex=True, sharey=True)
    xvalues = list(data.KCI_percent) + list(summary.KCI_mean-summary.KCI_sd) + list(summary.KCI_mean+summary.KCI_sd)
    yvalues = list(data.AvgRelRMSE) + list(summary.AvgRelRMSE_mean-summary.AvgRelRMSE_sd) + list(summary.AvgRelRMSE_mean+summary.AvgRelRMSE_sd) + [1.]
    xmin, xmax = min(xvalues), max(xvalues)
    ymin, ymax = min(yvalues), max(yvalues)
    xp = max((xmax-xmin)*.10, .5); yp = max((ymax-ymin)*.12, .05)
    for axis, region, title in zip(axes, ('internal','external'), ('Interpolation test','Extrapolation test')):
        selected = data[data.partition.eq(region)]
        for family, label in FAMILY.items():
            group = selected[(selected.family == family)&selected.role.eq('three_seed_mean_winner')]
            for row in group.itertuples(index=False):
                axis.scatter(row.KCI_percent, row.AvgRelRMSE, marker=MARKERS[row.seed], s=43,
                             facecolors=COLORS[family], edgecolors='white', linewidths=.6, zorder=4)
            stat = summary[(summary.family == family)&summary.partition.eq(region)].iloc[0]
            axis.errorbar(stat.KCI_mean, stat.AvgRelRMSE_mean, xerr=stat.KCI_sd,
                          yerr=stat.AvgRelRMSE_sd, fmt='D', markersize=7.0,
                          markerfacecolor='none', markeredgecolor=COLORS[family],
                          markeredgewidth=1.5, color=COLORS[family],
                          elinewidth=.8, capsize=2, zorder=5)
        b = selected[selected.role.eq('fixed_reference')].iloc[0]
        axis.scatter(b.KCI_percent, b.AvgRelRMSE, marker='*', s=130, c=COLORS['no_gain'],
                     edgecolors='white', linewidths=.5, zorder=6)
        t = selected[selected.role.eq('supplementary_single_run_validation_winner')].iloc[0]
        axis.scatter(t.KCI_percent, t.AvgRelRMSE, marker='X', s=78, facecolors='none',
                     edgecolors=COLORS['gain'], linewidths=1.6, zorder=6)
        axis.axhline(1, color='#777777', ls='--', lw=.8, zorder=1)
        lower = max(0, ymin-yp) if ymin >= 0 else ymin-yp
        axis.set(title=title, xlabel='KCI(%)', xlim=(xmin-xp,xmax+xp), ylim=(lower,ymax+yp))
        axis.grid(color='#d0d0d0', alpha=.55, linewidth=.6)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('AvgRelRMSE')
    families = [Line2D([],[],marker='o',ls='',color=COLORS[k],label=v,markersize=7) for k,v in FAMILY.items()]
    shapes = [Line2D([],[],marker=MARKERS[s],ls='',color='#777',label=f'Seed {s}',markersize=6) for s in (42,43,44)]
    shapes += [Line2D([],[],marker='D',ls='',markerfacecolor='none',markeredgecolor='#555',label='Mean ± SD (n=3)',markersize=6),
               Line2D([],[],marker='*',ls='',color='#333',label='No-Gain 42',markersize=9),
               Line2D([],[],marker='X',ls='',markerfacecolor='none',markeredgecolor=COLORS['gain'],label='Gain 2.5, seed 44 (supp.)',markersize=7)]
    fig.legend(handles=families,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False,fontsize=9)
    fig.legend(handles=shapes,loc='upper center',bbox_to_anchor=(.5,.90),ncol=6,frameon=False,fontsize=8)
    fig.suptitle('Existing test set | No-Gain seed 42 benchmark',fontsize=14,fontweight='bold',y=1.01)
    fig.tight_layout(rect=(0,0,1,.84))
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'test_kci_vs_avgrelrmse.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def render_html(data, summary, selection, labels=FAMILY):
    # Inline SVG + data: no network dependency. Values are the exact CSV coordinates.
    points = [{k: (float(getattr(r,k)) if k in ('KCI_percent','AvgRelRMSE') else getattr(r,k))
               for k in ('model','partition','family','alias','seed','role','KCI_percent','AvgRelRMSE')}
              for r in data.itertuples(index=False)]
    stats = [{k: (float(getattr(r,k)) if k.endswith(('_mean','_sd')) else getattr(r,k))
              for k in ('family','alias','partition','KCI_mean','KCI_sd','AvgRelRMSE_mean','AvgRelRMSE_sd','n')}
             for r in summary.itertuples(index=False)]
    payload = json.dumps({'points':points,'stats':stats,'colors':COLORS,'labels':labels},ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    page = r'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>已選模型 test 比較</title><style>
:root{color-scheme:light;--bg:#f5f5f7;--fg:#1d1d1f;--card:#fff;--mut:#6e6e73;--line:#e5e5ea;--acc:#0071e3}@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#000;--fg:#f5f5f7;--card:#1c1c1e;--mut:#a1a1a6;--line:#38383a;--acc:#2997ff}}*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.55 'Segoe UI','Microsoft JhengHei',sans-serif;margin:0}main{max-width:1250px;margin:auto;padding:30px 24px 70px}h1{font-size:30px;margin:0}.lead{color:var(--mut);margin:8px 0 22px}.card{background:var(--card);padding:18px;border-radius:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.stack{margin-top:18px}.mut{color:var(--mut)}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.legend span:before{content:'';display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:50%;background:var(--col)}svg{width:100%;height:auto}.axis{fill:var(--mut);font-size:11px}.label{fill:var(--fg);font-size:12px}.gridline{stroke:var(--line)}.dot{cursor:pointer;stroke:var(--card);stroke-width:1.5}.dot:hover,.dot:focus{stroke:var(--fg);stroke-width:3}.dot.selected{stroke:var(--fg);stroke-width:3}.tw{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}td,th{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}th{color:var(--mut);font-weight:500}tr.selectable{cursor:pointer}tr.selected{background:color-mix(in srgb,var(--acc) 15%,transparent)}button{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}button[aria-pressed=false]{opacity:.4}button:focus-visible,.dot:focus-visible{outline:2px solid var(--acc)}.controls{display:flex;gap:8px;flex-wrap:wrap}.note{font-size:12.5px}@media(max-width:820px){.grid{grid-template-columns:1fr}main{padding:20px 14px}h1{font-size:24px}}
</style></head><body><main><h1>已選模型 Test 比較</h1><p class="lead">左：內插組；右：外插組。所有相對 RMSE 使用 No-Gain seed 42 基準。</p><section class="card"><div class="controls" id="filters"></div><p class="mut note">圓、方、三角分別為 seed 42、43、44；菱形為三 seed 平均；細線為各軸 ±1 個樣本標準差。灰色星號是 No-Gain 42，藍色空心叉為補充的 Gain 2.5 seed 44。</p><div class="legend" id="legend"></div></section><section class="grid stack"><div class="card"><h2>內插 test</h2><svg id="internal" viewBox="0 0 560 390" role="img" aria-label="內插 test 散點圖"></svg></div><div class="card"><h2>外插 test</h2><svg id="external" viewBox="0 0 560 390" role="img" aria-label="外插 test 散點圖"></svg></div></section><section class="card stack" id="selection" aria-live="polite">點選或將游標移至圖上的點，查看模型與數值。</section><section class="stack"><h2>三 seed 平均與標準差</h2><div class="card tw"><table><thead><tr><th>模型設定</th><th>區域</th><th>KCI 平均 ± SD (%)</th><th>AvgRelRMSE 平均 ± SD</th><th>n</th></tr></thead><tbody id="means"></tbody></table></div></section><section class="stack"><h2>各次訓練的結果</h2><div class="card tw"><table><thead><tr><th>模型設定</th><th>seed</th><th>內插 KCI (%)</th><th>內插 AvgRelRMSE</th><th>外插 KCI (%)</th><th>外插 AvgRelRMSE</th></tr></thead><tbody id="rows"></tbody></table></div></section><section class="card stack note"><h2>方法與適用範圍</h2><p>四個模型家族的超參數由 validation 三 seed 平均綜合誤差事先選定；Gain 2.5 seed 44 是依 validation 單次表現選出的補充模型，並非三 seed 結果。test 結果未用來重新選模。</p><p>縱軸為動態案例對應條件下的 RMSE／No-Gain seed 42 RMSE，區域內逐項幾何平均；橫軸為另一批穩態探點的 ANN-reference KCI。KCI 不直接證明 Aspen 實際增益方向或閉迴路性能。</p><p>資料為既有 test 案例，非未曾檢視的新盲測。三個 seed 共用相同資料與 ANN；標準差是固定資料與 teacher 下動態訓練的描述性變異，非 joint 信賴區間。平均座標不是任何單一模型。</p></section></main><script id="dataset" type="application/json">__DATA__</script><script>
(function(){'use strict';const d=JSON.parse(document.getElementById('dataset').textContent),N='http://www.w3.org/2000/svg',visible=new Set(Object.keys(d.labels));let selected=null;const xvals=d.points.map(x=>x.KCI_percent).concat(d.stats.flatMap(x=>[x.KCI_mean-x.KCI_sd,x.KCI_mean+x.KCI_sd])),yvals=d.points.map(x=>x.AvgRelRMSE).concat([1],d.stats.flatMap(x=>[x.AvgRelRMSE_mean-x.AvgRelRMSE_sd,x.AvgRelRMSE_mean+x.AvgRelRMSE_sd]));const lo=Math.min(...xvals),hi=Math.max(...xvals),loy=Math.min(...yvals),hiy=Math.max(...yvals),xp=Math.max((hi-lo)*.1,.5),yp=Math.max((hiy-loy)*.12,.05),xmin=lo-xp,xmax=hi+xp,ymin=loy>=0?Math.max(0,loy-yp):loy-yp,ymax=hiy+yp,X=x=>65+(x-xmin)/(xmax-xmin)*470,Y=y=>322-(y-ymin)/(ymax-ymin)*270;function n(tag,a={},text){const e=document.createElementNS(N,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}function f(x,n=3){return Number(x).toFixed(n)}function label(p){return p.role==='fixed_reference'?'No-Gain 42':p.family==='no_gain'?'No-Gain / seed '+p.seed:p.role==='supplementary_single_run_validation_winner'?'Gain 2.5 / seed 44（補充）':d.labels[p.family]+' / seed '+p.seed}function choose(id){selected=id;let p=d.points.filter(x=>x.model===id),box=document.getElementById('selection');box.replaceChildren();let b=document.createElement('b');b.textContent=label(p[0]);box.append(b);for(const region of ['internal','external']){let q=p.find(x=>x.partition===region),s=document.createElement('span');s.textContent='　'+(region==='internal'?'內插':'外插')+'：KCI '+f(q.KCI_percent,2)+'%，AvgRelRMSE '+f(q.AvgRelRMSE,4);box.append(s)}document.querySelectorAll('[data-model]').forEach(e=>e.classList.toggle('selected',e.getAttribute('data-model')===selected))}function shape(x,y,seed){if(seed===42)return n('circle',{cx:X(x),cy:Y(y),r:5});if(seed===43)return n('rect',{x:X(x)-5,y:Y(y)-5,width:10,height:10});return n('path',{d:`M ${X(x)} ${Y(y)-6} L ${X(x)+6} ${Y(y)+5} L ${X(x)-6} ${Y(y)+5} Z`})}function scatter(region){let svg=document.getElementById(region);svg.replaceChildren();for(let i=0;i<5;i++){let x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;svg.append(n('line',{x1:X(x),x2:X(x),y1:52,y2:322,class:'gridline'}),n('text',{x:X(x),y:341,'text-anchor':'middle',class:'axis'},f(x,1)),n('line',{x1:65,x2:535,y1:Y(y),y2:Y(y),class:'gridline'}),n('text',{x:58,y:Y(y)+4,'text-anchor':'end',class:'axis'},f(y,2)))}svg.append(n('line',{x1:65,x2:535,y1:Y(1),y2:Y(1),stroke:'var(--mut)','stroke-dasharray':'4 4'}),n('text',{x:300,y:378,'text-anchor':'middle',class:'label'},'KCI(%)'),n('text',{x:16,y:187,transform:'rotate(-90 16 187)','text-anchor':'middle',class:'label'},'AvgRelRMSE'));for(const s of d.stats.filter(s=>s.partition===region&&visible.has(s.family))){let c=d.colors[s.family];svg.append(n('line',{x1:X(s.KCI_mean-s.KCI_sd),x2:X(s.KCI_mean+s.KCI_sd),y1:Y(s.AvgRelRMSE_mean),y2:Y(s.AvgRelRMSE_mean),stroke:c}),n('line',{x1:X(s.KCI_mean),x2:X(s.KCI_mean),y1:Y(s.AvgRelRMSE_mean-s.AvgRelRMSE_sd),y2:Y(s.AvgRelRMSE_mean+s.AvgRelRMSE_sd),stroke:c}),n('path',{d:`M ${X(s.KCI_mean)} ${Y(s.AvgRelRMSE_mean)-6} L ${X(s.KCI_mean)+6} ${Y(s.AvgRelRMSE_mean)} L ${X(s.KCI_mean)} ${Y(s.AvgRelRMSE_mean)+6} L ${X(s.KCI_mean)-6} ${Y(s.AvgRelRMSE_mean)} Z`,fill:'var(--card)',stroke:c,'stroke-width':2}))}for(const p of d.points.filter(p=>p.partition===region&&(p.family==='no_gain'||visible.has(p.family)))){let e=shape(p.KCI_percent,p.AvgRelRMSE,p.seed),color=d.colors[p.family];if(p.role==='fixed_reference'){let x=X(p.KCI_percent),y=Y(p.AvgRelRMSE);e=n('path',{d:`M ${x} ${y-8} L ${x+2} ${y-2} L ${x+8} ${y-2} L ${x+3} ${y+2} L ${x+5} ${y+8} L ${x} ${y+4} L ${x-5} ${y+8} L ${x-3} ${y+2} L ${x-8} ${y-2} L ${x-2} ${y-2} Z`})}if(p.role==='supplementary_single_run_validation_winner'){e=n('path',{d:`M ${X(p.KCI_percent)-5} ${Y(p.AvgRelRMSE)-5} l 10 10 m -10 0 l 10 -10`,fill:'none','stroke-width':2})}e.setAttribute('class','dot'+(selected===p.model?' selected':''));e.setAttribute('data-model',p.model);if(p.role==='supplementary_single_run_validation_winner')e.style.stroke=color;e.setAttribute('fill',p.role==='supplementary_single_run_validation_winner'?'none':color);e.setAttribute('stroke',p.role==='supplementary_single_run_validation_winner'?color:'var(--card)');e.setAttribute('tabindex','0');e.setAttribute('aria-label',label(p)+': KCI '+f(p.KCI_percent,2)+'%, AvgRelRMSE '+f(p.AvgRelRMSE,4));e.append(n('title',{},label(p)+'\nKCI '+f(p.KCI_percent,2)+'%\nAvgRelRMSE '+f(p.AvgRelRMSE,4)));e.onmouseenter=()=>choose(p.model);e.onclick=()=>choose(p.model);e.onfocus=()=>choose(p.model);e.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(p.model)}};svg.append(e)}}function tables(){let rows=document.getElementById('rows'),means=document.getElementById('means');rows.replaceChildren();means.replaceChildren();for(const s of d.stats.filter(s=>visible.has(s.family))){let tr=document.createElement('tr');for(const text of [d.labels[s.family],s.partition==='internal'?'內插':'外插',f(s.KCI_mean,2)+' ± '+f(s.KCI_sd,2),f(s.AvgRelRMSE_mean,4)+' ± '+f(s.AvgRelRMSE_sd,4),s.n]){let td=document.createElement('td');td.textContent=text;tr.append(td)}means.append(tr)}for(const id of [...new Set(d.points.filter(p=>p.family==='no_gain'||visible.has(p.family)).map(p=>p.model))]){let ps=d.points.filter(x=>x.model===id),a=ps.find(x=>x.partition==='internal'),b=ps.find(x=>x.partition==='external'),tr=document.createElement('tr');tr.className='selectable'+(selected===id?' selected':'');tr.tabIndex=0;tr.setAttribute('data-model',id);for(const text of [label(a),a.seed,f(a.KCI_percent,2),f(a.AvgRelRMSE,4),f(b.KCI_percent,2),f(b.AvgRelRMSE,4)]){let td=document.createElement('td');td.textContent=text;tr.append(td)}tr.onclick=()=>choose(id);tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(id)}};rows.append(tr)}}function draw(){scatter('internal');scatter('external');tables()}for(const family of Object.keys(d.labels)){let btn=document.createElement('button');btn.textContent=d.labels[family];btn.setAttribute('aria-pressed','true');btn.onclick=()=>{visible.has(family)?visible.delete(family):visible.add(family);btn.setAttribute('aria-pressed',visible.has(family));draw()};document.getElementById('filters').append(btn);let item=document.createElement('span');item.textContent=d.labels[family];item.style.setProperty('--col',d.colors[family]);document.getElementById('legend').append(item)}draw()})();
</script></body></html>'''
    return page.replace('__DATA__', payload)


def main_selection(data):
    """Freeze and verify the user's per-family single-run validation selection."""
    validation = pd.read_csv(VALIDATION)
    expected = {'gain': ('gain2p5', 44), 'ss_only': ('ss2p5_only', 44),
                'gain_ss': ('gain2p5_ss2p5', 42), 'difficult_gradnorm': ('dg_ratio0p015', 43)}
    winners = []
    for family, pair in expected.items():
        candidates = validation[validation.family.eq(family)]
        if candidates.empty or not np.isfinite(candidates.overall_50_50).all():
            raise ValueError(f'Invalid validation candidates: {family}')
        minimum = candidates.overall_50_50.min()
        best = candidates[candidates.overall_50_50.eq(minimum)]
        if len(best) != 1 or (best.iloc[0]['model'], int(best.iloc[0]['seed'])) != pair:
            raise ValueError(f'Validation single-run selection changed: {family}')
        winners.append(best.iloc[0].to_dict())
    ids = ['no_gain_seed42'] + [f'{alias}_seed{seed}' for alias, seed in expected.values()]
    selected = data[data.model.isin(ids)].copy()
    if len(selected) != 10 or set(selected.model) != set(ids):
        raise ValueError('Expected five preselected runs in each test region.')
    return selected, pd.DataFrame(winners)


def plot_main(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    xmin, xmax = data.KCI_percent.min(), data.KCI_percent.max()
    ymin, ymax = min(data.AvgRelRMSE.min(), 1), max(data.AvgRelRMSE.max(), 1)
    xp, yp = max((xmax-xmin)*.1, .5), max((ymax-ymin)*.12, .05)
    names = {'no_gain': 'No-Gain, seed 42', 'gain': 'Gain 2.5, seed 44',
             'ss_only': 'SS-only 2.5, seed 44', 'gain_ss': 'Gain+SS 2.5 / 2.5, seed 42',
             'difficult_gradnorm': 'Difficult-GradNorm ρ = 0.015, seed 43'}
    handles = []
    for family, label in names.items():
        row = data[data.family.eq(family)].iloc[0]
        marker = '*' if family == 'no_gain' else MARKERS[row.seed]
        handles.append(Line2D([], [], marker=marker, ls='', color=COLORS[family],
                              label=label, markersize=9 if family == 'no_gain' else 7))
    for ax, region, title in zip(axes, ('internal', 'external'), ('Interpolation test', 'Extrapolation test')):
        for row in data[data.partition.eq(region)].itertuples(index=False):
            marker = '*' if row.family == 'no_gain' else MARKERS[row.seed]
            ax.scatter(row.KCI_percent, row.AvgRelRMSE, marker=marker,
                       s=140 if row.family == 'no_gain' else 80, c=COLORS[row.family],
                       edgecolors='white', linewidths=.8, zorder=4)
        ax.axhline(1, color='#777', ls='--', lw=.8)
        ax.set(title=title, xlabel='KCI(%)', xlim=(xmin-xp, xmax+xp),
               ylim=(max(0, ymin-yp), ymax+yp))
        ax.grid(color='#d0d0d0', alpha=.55, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('AvgRelRMSE')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.96), ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle('Validation-selected single runs | Existing test', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(rect=(0,0,1,.82))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'primary_single_run_comparison.{ext}', dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def primary_html(data, summary, selection, raw):
    shown = data.copy()
    shown.loc[shown.family.ne('no_gain'), 'role'] = 'single_run_validation_winner'
    labels = dict(FAMILY, gain='Gain = 2.5')
    page = render_html(shown, summary.iloc[0:0], selection, labels=labels)
    old = '圓、方、三角分別為 seed 42、43、44；菱形為三 seed 平均；細線為各軸 ±1 個樣本標準差。灰色星號是 No-Gain 42，藍色空心叉為補充的 Gain 2.5 seed 44。'
    new = '圓、方、三角分別為 seed 42、43、44；灰色星號是共同基準 No-Gain 42。四個彩色點各為該家族依 validation 選出的最低單次。'
    old_method = '四個模型家族的超參數由 validation 三 seed 平均綜合誤差事先選定；Gain 2.5 seed 44 是依 validation 單次表現選出的補充模型，並非三 seed 結果。test 結果未用來重新選模。'
    new_method = '四個模型家族各依固定 No-Gain 42 基準的 validation 綜合誤差，事先選出最低單次與其 seed；test 結果未用來重新選模。單次模型不代表三次訓練的平均表現。'
    if old not in page or old_method not in page:
        raise ValueError('Unexpected report template text.')
    page = page.replace(old, new).replace(old_method, new_method)
    old_note = '三個 seed 共用相同資料與 ANN；標準差是固定資料與 teacher 下動態訓練的描述性變異，非 joint 信賴區間。平均座標不是任何單一模型。'
    if old_note not in page:
        raise ValueError('Unexpected report caveat text.')
    page = page.replace(old_note, '各模型共用相同 ANN 與資料；三 seed 變異請參考補充比較報告。')
    page = page.replace('</style>', 'section:has(#means){display:none}</style>', 1)
    table = raw.to_html(index=False, escape=True, classes='raw', float_format=lambda x: f'{x:.6g}')
    extra = ('<section class="card stack"><h2>逐案例原始誤差與配對基準</h2>'
             '<p class="mut">各模型、各案例、輸出與評估方式的 RMSE；單位依輸出資料，'
             '完整數值另見 primary_raw_rmse.csv。</p><div class="tw">' + table + '</div></section>')
    return page.replace('</main><script id="dataset"', extra + '</main><script id="dataset"', 1)


def main():
    data, summary, selection = load()
    inputs = ['evaluation_validation.json','selected_models.json','regional_coordinates.csv',
              'scores_by_model_seed.csv','test_metrics.csv','relative_rmse_components.csv','kci_components.csv']
    hashes = {name: sha(SOURCE/name) for name in inputs}
    validation_hash = sha(VALIDATION)
    primary, winners = main_selection(data)
    raw = pd.read_csv(SOURCE/'relative_rmse_components.csv')
    raw = raw[raw.model.isin(primary.model)].copy()
    if len(raw) != 360:
        raise ValueError('Expected 360 primary raw RMSE components.')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing report: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42})
    summary.to_csv(OUT/'regional_summary_mean_sd.csv',index=False,float_format='%.17g')
    plot(data,summary)
    (OUT/'test_comparison.html').write_text(render_html(data,summary,selection),encoding='utf-8')
    winners.to_csv(OUT/'primary_validation_selection.csv', index=False, float_format='%.17g')
    raw.to_csv(OUT/'primary_raw_rmse.csv', index=False, float_format='%.17g')
    plot_main(primary)
    (OUT/'primary_test_comparison.html').write_text(primary_html(primary,summary,selection,raw),encoding='utf-8')
    description = '''# 既有 Test：固定 No-Gain seed 42 比較\n\n四個模型家族各保留三個 seed 的原始點；Gain 2.5 seed 44 是 validation 單次表現選出的補充模型。菱形為三次訓練之座標算術平均，橫／直誤差棒各為樣本標準差（n=3），不是聯合信賴區間，也不是另一個模型。\n\n動態 test：內插 R5、R5-1、R5-2、R5-6 的 test split；外插 R5-19～R5-26。縱軸為逐條件 RMSE／相同條件 No-Gain seed 42，再在每區取幾何平均。橫軸使用另一組穩態 test 探點，為 ANN 參考增益方向的一致率，不能直接證明 Aspen 真實增益或閉迴路控制表現。\n\n既有 test 案例並非未曾檢視的全新盲測。所有模型共用同一 ANN 和資料；三個 seed 的變異只反映固定資料及 teacher 下動態模型訓練的變異。\n'''
    intro = ('# 主比較與補充比較\n\n'
             '`primary_test_comparison.html` 與 `primary_single_run_comparison` 圖是主要比較：'
             '每個家族依 validation 的單次最低綜合誤差預選模型，加上共同基準 No-Gain 42。'
             '`primary_validation_selection.csv` 記錄選模依據，`primary_raw_rmse.csv` 為逐案例原始 RMSE 及配對基準。'
             '既有 `test_comparison.html` 與 `test_kci_vs_avgrelrmse` 圖為三 seed 平均選模與變異的補充比較。\n\n')
    (OUT/'README.md').write_text(intro + description,encoding='utf-8')
    for name, digest in hashes.items():
        if sha(SOURCE/name) != digest:
            raise ValueError(f'Test source changed while rendering: {name}')
    if sha(VALIDATION) != validation_hash:
        raise ValueError('Validation selection source changed while rendering.')
    result = dict(passed=True, sources_sha256=hashes, validation_selection_sha256=validation_hash,
                  output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
                  regions=2, seed_coordinates_per_region=14, summary_coordinates_per_region=4,
                  baseline='No-Gain seed 42', means_are_models=False, sd='sample SD n=3', jitter=False)
    (OUT/'report_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Created:', OUT, flush=True)


if __name__ == '__main__':
    main()
