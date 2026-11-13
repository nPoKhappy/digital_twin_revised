"""Render the completed seed-42 regional evaluation as an offline HTML report."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'results/visualization/retrain_lhs10_23_21_models_validation_seed42'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_report(directory: Path) -> dict:
    # Refuse partial output while the evaluator is still running.
    result = json.loads((directory / 'evaluation_validation.json').read_text(encoding='utf-8-sig'))
    if not (result.get('passed') is True and result.get('seed') == 42
            and result.get('model_count') == 21 and result.get('per_region_points') == 21
            and result.get('source_hashes_unchanged') is True):
        raise ValueError('Evaluation has not completed successfully; refusing to render partial data.')
    protocol = json.loads((directory / 'run_protocol.json').read_text(encoding='utf-8-sig'))
    coordinates = read_csv(directory / 'regional_plot_coordinates.csv')
    metrics = read_csv(directory / 'validation_metrics.csv')
    relative = read_csv(directory / 'relative_rmse_components.csv')
    kci = read_csv(directory / 'kci_components.csv')
    jobs = protocol['jobs']
    expected = {job['alias']: job['family'] for job in jobs}
    if len(jobs) != 21 or len(expected) != 21 or len(coordinates) != 42:
        raise ValueError('Expected 21 unique models and 42 regional points.')
    if len(metrics) != 21 * 36 or len(relative) != 21 * 36 or len(kci) != 21 * 8:
        raise ValueError('The detail tables are incomplete.')
    points = []
    seen = set()
    for row in coordinates:
        key = (row['model'], row['partition'])
        if key in seen or row['model'] not in expected or row['family'] != expected[row['model']]:
            raise ValueError(f'Unexpected or duplicate model/region: {key}')
        if row['partition'] not in ('internal', 'external'):
            raise ValueError(f'Unexpected partition: {key}')
        seen.add(key)
        x, y = float(row['ANN_reference_KCI_percent']), float(row['AvgRelRMSE'])
        correct, valid, components = int(row['correct']), int(row['valid']), int(row['components'])
        if (not math.isfinite(x) or not math.isfinite(y) or valid <= 0 or y <= 0
                or not 0 <= correct <= valid or components != 18
                or not math.isclose(x, 100 * correct / valid, rel_tol=1e-10)):
            raise ValueError(f'Invalid coordinate: {key}')
        points.append(dict(model=row['model'], partition=row['partition'], family=row['family'],
                           x=x, y=y, correct=correct, valid=valid))
    if seen != {(model, region) for model in expected for region in ('internal', 'external')}:
        raise ValueError('Some model/region combinations are missing.')
    if Counter((r['model'], r['partition']) for r in metrics) != Counter({k: 18 for k in seen}):
        raise ValueError('Unexpected validation metric coverage.')
    groups: dict[tuple[str, str], list[float]] = {}
    for row in relative:
        key = (row['model'], row['partition'])
        ratio, rmse, baseline = float(row['relative_RMSE']), float(row['RMSE']), float(row['baseline_RMSE'])
        if (key not in seen or not all(math.isfinite(v) and v > 0 for v in (ratio, rmse, baseline))
                or not math.isclose(ratio, rmse / baseline, rel_tol=1e-10)):
            raise ValueError(f'Invalid relative RMSE component: {key}.')
        groups.setdefault(key, []).append(ratio)
    for p in points:
        values = groups.get((p['model'], p['partition']), [])
        if len(values) != 18 or not math.isclose(p['y'], math.exp(sum(map(math.log, values)) / 18), rel_tol=1e-10):
            raise ValueError(f'RMSE components disagree with plotted coordinates: {p["model"]}.')
    if Counter((r['model'], r['distribution']) for r in kci) != Counter({(m, d): 4 for m in expected for d in ('in_range', 'soft_ood')}):
        raise ValueError('Unexpected KCI component coverage.')
    details = []
    for r in kci:
        valid = int(r['valid_comparisons'])
        correct = int(r['tp']) + int(r['tn'])
        if valid <= 0 or not 0 <= correct <= valid:
            raise ValueError('Invalid KCI component counts.')
        details.append(dict(model=r['model'], partition={'in_range':'internal', 'soft_ood':'external'}[r['distribution']],
                            mv=r['mv'], target=r['target'], correct=correct, valid=valid))
    counts = Counter()
    for r in details:
        counts[(r['model'], r['partition'], 'correct')] += r['correct']
        counts[(r['model'], r['partition'], 'valid')] += r['valid']
    for p in points:
        if (p['correct'] != counts[(p['model'],p['partition'],'correct')]
                or p['valid'] != counts[(p['model'],p['partition'],'valid')]):
            raise ValueError(f'KCI components disagree with plotted coordinates: {p["model"]}.')
    return dict(points=points, details=details, protocol={k:protocol[k] for k in
                ('internal_cases','external_cases','kci_sources','rmse','kci','note')})


PAGE = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>21 個模型｜內插與外插驗證</title>
<style>
:root{color-scheme:light;--bg:#f5f5f7;--fg:#1d1d1f;--mut:#6e6e73;--card:#fff;--line:#e5e5ea;--acc:#0071e3;--seg:#e8e8ed;--nav:rgba(245,245,247,.88);--f0:#424245;--f1:#0072b2;--f2:#c84f10;--f3:#00845e;--f4:#8248b4}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#000;--fg:#f5f5f7;--mut:#a1a1a6;--card:#1c1c1e;--line:#38383a;--acc:#2997ff;--seg:#2c2c2e;--nav:rgba(22,22,23,.9);--f0:#c6c6ca;--f1:#55a8e5;--f2:#ff9867;--f3:#48d1a1;--f4:#c29bf0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft JhengHei',sans-serif}nav{position:sticky;top:0;z-index:3;background:var(--nav);backdrop-filter:blur(18px);border-bottom:1px solid var(--line)}.navin{max-width:1200px;margin:auto;padding:12px 24px;display:flex;justify-content:space-between;gap:12px}.mut{color:var(--mut)}main{max-width:1200px;margin:auto;padding:38px 24px 72px}h1{font-size:34px;line-height:1.15;letter-spacing:-.025em;margin:0}h2{font-size:21px;letter-spacing:-.015em;margin:0 0 12px}h3{font-size:15px;margin:0}.lead{font-size:17px;color:var(--mut);margin:8px 0 20px}.card{background:var(--card);border-radius:16px;padding:18px 20px}.stack{margin-top:22px}.small{font-size:12.5px}.filters{display:flex;gap:7px;flex-wrap:wrap}.filters button{font:inherit;border:1px solid var(--line);color:var(--fg);background:var(--seg);border-radius:8px;padding:7px 11px;cursor:pointer}.filters button[aria-pressed=true]{background:var(--card);border-color:var(--acc);box-shadow:inset 0 0 0 1px var(--acc)}button:focus-visible,.dot:focus-visible{outline:2px solid var(--acc);outline-offset:2px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}.chart{width:100%;height:auto;display:block;overflow:visible}.chart text{font-family:inherit;fill:var(--mut);font-size:11px}.chart .axis-title{fill:var(--fg);font-size:12px}.chart .grid{stroke:var(--line);stroke-width:1}.chart .baseline{stroke:var(--mut);stroke-dasharray:4 4}.dot{cursor:pointer;stroke:var(--card);stroke-width:1.4}.dot:hover,.dot:focus{stroke:var(--fg);stroke-width:2}.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0 0 6px}.legend span:before{content:'';display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%;background:var(--color)}.description{color:var(--mut);margin:5px 0 0}.selection{min-height:86px}.selection strong{font-size:16px}.detail-grid{display:flex;gap:20px;flex-wrap:wrap;margin:5px 0}.detail-grid span{font-variant-numeric:tabular-nums}.tw{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{color:var(--mut);font-weight:500}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tbody tr{cursor:pointer}tbody tr:hover{background:var(--seg)}.formula{font-family:ui-monospace,Consolas,monospace;overflow-wrap:anywhere}code{font-size:12px}p{margin:8px 0}a{color:var(--acc)}
@media(max-width:820px){.charts{grid-template-columns:1fr}main{padding:26px 16px 52px}.navin{padding:10px 16px}h1{font-size:27px}.card{padding:14px}.chart text{font-size:10px}}
</style></head><body><nav><div class="navin"><b>動態模型驗證</b><span class="mut">seed 42 · 21 個模型</span></div></nav><main>
<h1>準確度與增益方向</h1><p class="lead">左側內插、右側外插：兩個座標分別來自不同的驗證資料集。</p>
<section class="card"><h3>模型家族</h3><p class="description small">可切換家族；將滑鼠移到點上或點選圖表及表格，查看模型數值。兩張圖使用相同座標範圍。</p><div class="filters stack" id="filters" role="group" aria-label="模型家族篩選"></div></section>
<section class="stack"><div class="legend" id="legend"></div><div class="charts"><div class="card"><h2>內插驗證</h2><p class="description small">動態：R5-3、4、5；KCI：in-range 穩態點</p><svg class="chart" id="internal" viewBox="0 0 520 380" role="img" aria-label="內插 KCI 與 AvgRelRMSE 散點圖"></svg></div><div class="card"><h2>外插驗證</h2><p class="description small">動態：R5-27、28、29；KCI：soft-OOD 穩態點</p><svg class="chart" id="external" viewBox="0 0 520 380" role="img" aria-label="外插 KCI 與 AvgRelRMSE 散點圖"></svg></div></div></section>
<section class="card stack selection" id="selection" aria-live="polite"><span class="mut">選擇一個模型點，可查看兩區域的詳細數值。</span></section>
<section class="stack"><h2>模型結果</h2><div class="card tw"><table><thead><tr><th>模型</th><th>家族</th><th>內插 KCI %</th><th>內插 AvgRelRMSE</th><th>外插 KCI %</th><th>外插 AvgRelRMSE</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="stack"><h2>指標與資料來源</h2><div class="card small"><p><b>縱軸 AvgRelRMSE：</b>各區域 3 個動態案例 × 2 個輸出（H₂S、SO₂）× 3 種評估（H1、H18、TS），合計 18 個 RMSE 比值的幾何平均；每項皆除以相同案例的 seed 42 No-Gain RMSE。No-Gain = 1，較低表示相對誤差較小。</p><p><b>橫軸 ANN-reference KCI：</b>使用另外的穩態驗證資料，將 2 個 MV × 2 個輸出、正負擾動方向及擾動後 18 個時間點的有效符號比較合併計算；ANN 參考梯度絕對值小於 10⁻⁵ 者不計，動態增益為零視為不一致。這是對 ANN 增益方向的一致率，不直接證明 Aspen 真實增益或閉迴路效果。</p><p><b>左圖：</b>RMSE 來自 R5-3、R5-4、R5-5；KCI 來自 <code>lhs_valid_in_range.xlsx</code>。<b>右圖：</b>RMSE 來自 R5-27、R5-28、R5-29；KCI 來自 <code>lhs_valid_soft_ood.xlsx</code>。R5-5 跨越核心範圍邊界。</p><p>兩軸不是由同一批案例計算。此圖只有 seed 42，尚未呈現三次重複訓練的變異。圖上虛線為 No-Gain 相對誤差基準。</p></div></section>
</main><script id="dataset" type="application/json">__DATA__</script><script>
(function(){'use strict';const data=JSON.parse(document.getElementById('dataset').textContent);const fam=[['no_gain','No-Gain','--f0'],['gain','Gain','--f1'],['ss_only','SS-only','--f2'],['gain_ss','Gain + SS','--f3'],['difficult_gradnorm','Difficult-GradNorm','--f4']];const labels=Object.fromEntries(fam.map(x=>[x[0],x[1]]));const visible=new Set(fam.map(x=>x[0]));let selected=null;const N='http://www.w3.org/2000/svg';
function node(tag,attrs={},txt){const e=document.createElementNS(N,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(txt!==undefined)e.textContent=txt;return e}function fmt(n,d=3){return Number(n).toFixed(d)}
const fs=document.getElementById('filters'),lg=document.getElementById('legend');for(const[f,l,c]of fam){let b=document.createElement('button');b.textContent=l;b.setAttribute('aria-pressed','true');b.onclick=()=>{visible.has(f)?visible.delete(f):visible.add(f);b.setAttribute('aria-pressed',visible.has(f));draw()};fs.append(b);let s=document.createElement('span');s.style.setProperty('--color',`var(${c})`);s.textContent=l;lg.append(s)}
let xs=data.points.map(p=>p.x),ys=data.points.map(p=>p.y);function limits(a,zero=false){let lo=Math.min(...a),hi=Math.max(...a),p=Math.max((hi-lo)*.08,zero?.05:.5);return [zero?Math.max(0,lo-p):Math.max(0,lo-p),zero?hi+p:Math.min(100,hi+p)]}let [xmin,xmax]=limits(xs),[ymin,ymax]=limits(ys.concat([1]),true);if(xmin===xmax)xmax=xmin+1;if(ymin===ymax)ymax=ymin+1;
const X=x=>70+(x-xmin)/(xmax-xmin)*425,Y=y=>312-(y-ymin)/(ymax-ymin)*270;function select(model){selected=model;const sel=document.getElementById('selection');sel.replaceChildren();let h=document.createElement('strong');h.textContent=model+' · '+labels[data.points.find(p=>p.model===model).family];sel.append(h);let row=document.createElement('div');row.className='detail-grid';for(const reg of ['internal','external']){let p=data.points.find(q=>q.model===model&&q.partition===reg);let s=document.createElement('span');s.textContent=(reg==='internal'?'內插':'外插')+'：KCI '+fmt(p.x,2)+'%（'+p.correct.toLocaleString()+'/'+p.valid.toLocaleString()+'），AvgRelRMSE '+fmt(p.y,4);row.append(s)}sel.append(row);let dd=document.createElement('div');dd.className='mut small';dd.textContent='KCI 各分量：'+data.details.filter(d=>d.model===model).map(d=>(d.partition==='internal'?'內插':'外插')+' '+(d.mv==='air2_SP'?'air2':'T2')+'→'+d.target.replace('B35_','')+' '+fmt(100*d.correct/d.valid,1)+'%').join('　·　');sel.append(dd)}
function scatter(part){let svg=document.getElementById(part);svg.replaceChildren();for(let i=0;i<5;i++){let x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;svg.append(node('line',{x1:X(x),x2:X(x),y1:42,y2:312,class:'grid'}),node('text',{x:X(x),y:330,'text-anchor':'middle'},fmt(x,1)),node('line',{x1:70,x2:495,y1:Y(y),y2:Y(y),class:'grid'}),node('text',{x:62,y:Y(y)+4,'text-anchor':'end'},fmt(y,2)))}if(ymin<=1&&ymax>=1)svg.append(node('line',{x1:70,x2:495,y1:Y(1),y2:Y(1),class:'baseline'}));svg.append(node('text',{x:282,y:365,'text-anchor':'middle',class:'axis-title'},'ANN-reference KCI (%)'),node('text',{x:16,y:177,transform:'rotate(-90 16 177)','text-anchor':'middle',class:'axis-title'},'AvgRelRMSE'));
for(const p of data.points.filter(p=>p.partition===part&&visible.has(p.family))){let color=fam.find(f=>f[0]===p.family)[2],cx=X(p.x),cy=Y(p.y),el;switch(p.family){case 'ss_only':el=node('rect',{x:cx-5,y:cy-5,width:10,height:10});break;case 'gain_ss':el=node('path',{d:`M ${cx} ${cy-7} L ${cx+7} ${cy+5} L ${cx-7} ${cy+5} Z`});break;case 'difficult_gradnorm':el=node('path',{d:`M ${cx} ${cy-7} L ${cx+7} ${cy} L ${cx} ${cy+7} L ${cx-7} ${cy} Z`});break;case 'no_gain':el=node('path',{d:`M ${cx} ${cy-8} L ${cx+2} ${cy-2} L ${cx+8} ${cy-2} L ${cx+3} ${cy+2} L ${cx+5} ${cy+8} L ${cx} ${cy+4} L ${cx-5} ${cy+8} L ${cx-3} ${cy+2} L ${cx-8} ${cy-2} L ${cx-2} ${cy-2} Z`});break;default:el=node('circle',{cx,cy,r:5.5})}el.setAttribute('class','dot');el.setAttribute('fill',`var(${color})`);el.setAttribute('tabindex','0');el.setAttribute('aria-label',`${p.model}: KCI ${fmt(p.x,2)}%, AvgRelRMSE ${fmt(p.y,4)}`);el.append(node('title',{},`${p.model}\nKCI ${fmt(p.x,2)}%\nAvgRelRMSE ${fmt(p.y,4)}`));el.addEventListener('click',()=>select(p.model));el.addEventListener('mouseenter',()=>select(p.model));el.addEventListener('focus',()=>select(p.model));el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(p.model)}});svg.append(el)}}
function table(){let rows=document.getElementById('rows');rows.replaceChildren();let names=[...new Set(data.points.filter(p=>visible.has(p.family)).map(p=>p.model))];for(const name of names){let a=data.points.find(p=>p.model===name&&p.partition==='internal'),b=data.points.find(p=>p.model===name&&p.partition==='external'),tr=document.createElement('tr');for(const v of [name,labels[a.family],fmt(a.x,2),fmt(a.y,4),fmt(b.x,2),fmt(b.y,4)]){let td=document.createElement('td');td.textContent=v;tr.append(td)}tr.tabIndex=0;tr.onclick=()=>select(name);tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(name)}};rows.append(tr)}}function draw(){scatter('internal');scatter('external');table()}draw()})();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--output', type=Path, help='Default: input directory / regional_validation.html')
    args = parser.parse_args()
    data = load_report(args.input_dir)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    page = PAGE.replace('__DATA__', payload)
    output = args.output or args.input_dir / 'regional_validation.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    print(f'Wrote {output}: {len(data["points"])} verified regional points.')


if __name__ == '__main__':
    main()
