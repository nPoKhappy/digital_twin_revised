"""Parameter-labelled, offline visualization of the verified three-seed data."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml


def model_parameters(root: Path):
    manifest = json.loads((root / 'configs/pgin/retrain_lhs10_23/config_manifest.json').read_text(encoding='utf-8'))
    seed_models = {}
    for seed in (42, 43, 44):
        items = {}
        for job in (j for j in manifest if j['seed'] == seed):
            path = root / job['config']
            if hashlib.sha256(path.read_bytes()).hexdigest().lower() != job['config_sha256'].lower():
                raise ValueError(f'Config hash mismatch: {path}')
            cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            family = job['family']
            gain = float(cfg['training'].get('gain_loss_weight', 0))
            ss = float(cfg['training'].get('ss_loss_weight', 0))
            if family == 'difficult_gradnorm':
                args = job['cli_args']
                rho = float(args[args.index('--target-ratio') + 1])
                value = rho
                short = f'D ρ={rho:g}'
                full = f'Difficult-GradNorm · ρ = {rho:g}（Gain = {gain:g}，SS = {ss:g}）'
            elif family == 'no_gain':
                if gain != 0 or ss != 0:
                    raise ValueError('Unexpected No-Gain weights')
                value, short, full = 0., 'NG', 'No-Gain · Gain = 0，SS = 0'
            elif family in ('gain', 'ss_only', 'gain_ss'):
                value = ss if family == 'ss_only' else gain
                if family == 'gain_ss' and gain != ss:
                    raise ValueError('Gain+SS weights do not match')
                prefix = {'gain': 'G', 'ss_only': 'S', 'gain_ss': 'GS'}[family]
                label = {'gain': 'Gain', 'ss_only': 'SS-only', 'gain_ss': 'Gain+SS'}[family]
                short = f'{prefix} {value:g}'
                full = f'{label} · Gain = {gain:g}，SS = {ss:g}'
            else:
                raise ValueError(f'Unexpected family: {family}')
            items[job['alias']] = dict(model=job['alias'], family=family, value=value, short=short, full=full)
        if len(items) != 21:
            raise ValueError(f'Seed {seed}: expected 21 configurations')
        seed_models[seed] = items
    if seed_models[42] != seed_models[43] or seed_models[42] != seed_models[44]:
        raise ValueError('Model parameters differ across seeds; cannot share a legend')
    order = {'no_gain': 0, 'gain': 1, 'ss_only': 2, 'gain_ss': 3, 'difficult_gradnorm': 4}
    models = sorted(seed_models[42].values(), key=lambda m: (order[m['family']], m['value']))
    for number, item in enumerate(models, 1):
        item['number'] = number
    return models


EXTRA_STYLE = r'''
.model-key-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:6px;margin-top:12px}
.model-key{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:7px;text-align:left;cursor:pointer;font:inherit;font-size:12px}
.model-key .badge{display:inline-block;min-width:24px;margin-right:5px;font-weight:700;color:var(--color)}
.model-key.selected, tr.selected{background:var(--seg);box-shadow:inset 0 0 0 2px var(--acc)}
.chart .point-label{font-weight:650;font-size:13px;cursor:pointer;paint-order:stroke;stroke:var(--card);stroke-width:2.5px;stroke-linejoin:round}
.chart .leader{stroke-width:.8;opacity:.55;pointer-events:none}
.chart .dot.selected{stroke:var(--fg);stroke-width:3}
.chart .point-label.selected{font-size:15px;stroke-width:4}
.chart .leader.selected{stroke-width:1.7;opacity:1}
.number-col{font-weight:700}.parameter-col{font-weight:600}
.charts .card{min-width:0}.chart-scroll{max-width:100%;overflow-x:auto}
@media(max-width:820px){.chart-scroll .chart{min-width:600px}}
'''


INTERACTION = r'''
(function(){'use strict';
const payload=JSON.parse(document.getElementById('dataset').textContent),datasets=payload.datasets,models=payload.models;
let seed='42',data=datasets[seed],selected=null;
const fam=[['no_gain','No-Gain','--f0'],['gain','Gain','--f1'],['ss_only','SS-only','--f2'],['gain_ss','Gain + SS','--f3'],['difficult_gradnorm','Difficult-GradNorm','--f4']];
const labels=Object.fromEntries(fam.map(x=>[x[0],x[1]])), meta=Object.fromEntries(models.map(x=>[x.model,x]));
const visible=new Set(fam.map(x=>x[0])), N='http://www.w3.org/2000/svg';
const node=(tag,attrs={},txt)=>{const e=document.createElementNS(N,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(txt!==undefined)e.textContent=txt;return e};
const fmt=(n,d=3)=>Number(n).toFixed(d), number=m=>String(meta[m].number).padStart(2,'0');
const fs=document.getElementById('filters'),lg=document.getElementById('legend');
for(const[f,l,c]of fam){let b=document.createElement('button');b.textContent=l;b.setAttribute('aria-pressed','true');b.onclick=()=>{visible.has(f)?visible.delete(f):visible.add(f);b.setAttribute('aria-pressed',visible.has(f));draw()};fs.append(b);let s=document.createElement('span');s.style.setProperty('--color',`var(${c})`);s.textContent=l;lg.append(s)}
const ss=document.getElementById('seed-selector');for(const key of ['42','43','44']){let b=document.createElement('button');b.textContent='seed '+key;b.setAttribute('aria-pressed',key===seed);b.onclick=()=>{seed=key;data=datasets[key];selected=null;document.getElementById('selection').textContent='選擇圖上的標籤、模型對照表或結果表，可同時標示兩區的對應點。';document.getElementById('seed-status').textContent='seed '+key+' · 21 個模型';for(const child of ss.children)child.setAttribute('aria-pressed',child===b);draw()};ss.append(b)}
const xs=Object.values(datasets).flatMap(d=>d.points.map(p=>p.x)), ys=Object.values(datasets).flatMap(d=>d.points.map(p=>p.y));
function limits(a,y=false){let lo=Math.min(...a),hi=Math.max(...a),p=Math.max((hi-lo)*.08,y?.05:.5);return [Math.max(0,lo-p),y?hi+p:Math.min(100,hi+p)]}
let [xmin,xmax]=limits(xs),[ymin,ymax]=limits(ys.concat([1]),true);if(xmin===xmax)xmax=xmin+1;if(ymin===ymax)ymax=ymin+1;
const X=x=>70+(x-xmin)/(xmax-xmin)*360,Y=y=>450-(y-ymin)/(ymax-ymin)*400;
function highlight(){document.querySelectorAll('[data-model]').forEach(el=>el.classList.toggle('selected',el.getAttribute('data-model')===selected))}
function select(model){selected=model;const info=meta[model],sel=document.getElementById('selection');sel.replaceChildren();let h=document.createElement('strong');h.textContent=number(model)+' · '+info.full;sel.append(h);let row=document.createElement('div');row.className='detail-grid';for(const reg of ['internal','external']){let p=data.points.find(q=>q.model===model&&q.partition===reg),s=document.createElement('span');s.textContent=(reg==='internal'?'內插':'外插')+'：KCI '+fmt(p.x,2)+'%（'+p.correct.toLocaleString()+'/'+p.valid.toLocaleString()+'），AvgRelRMSE '+fmt(p.y,4);row.append(s)}sel.append(row);let dd=document.createElement('div');dd.className='mut small';dd.textContent='KCI 各分量：'+data.details.filter(d=>d.model===model).map(d=>(d.partition==='internal'?'內插':'外插')+' '+(d.mv==='air2_SP'?'air2':'T2')+'→'+d.target.replace('B35_','')+' '+fmt(100*d.correct/d.valid,1)+'%').join('　·　');sel.append(dd);highlight()}
function bind(el,model){el.setAttribute('data-model',model);el.addEventListener('click',()=>select(model));el.addEventListener('mouseenter',()=>select(model));el.addEventListener('focus',()=>select(model));el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(model)}})}
function scatter(part){let svg=document.getElementById(part);svg.replaceChildren();for(let i=0;i<5;i++){let x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;svg.append(node('line',{x1:X(x),x2:X(x),y1:50,y2:450,class:'grid'}),node('text',{x:X(x),y:468,'text-anchor':'middle'},fmt(x,1)),node('line',{x1:70,x2:430,y1:Y(y),y2:Y(y),class:'grid'}),node('text',{x:62,y:Y(y)+4,'text-anchor':'end'},fmt(y,2)))}if(ymin<=1&&ymax>=1)svg.append(node('line',{x1:70,x2:430,y1:Y(1),y2:Y(1),class:'baseline'}));svg.append(node('text',{x:250,y:510,'text-anchor':'middle',class:'axis-title'},'ANN-reference KCI (%)'),node('text',{x:16,y:250,transform:'rotate(-90 16 250)','text-anchor':'middle',class:'axis-title'},'AvgRelRMSE'));
const points=data.points.filter(p=>p.partition===part&&visible.has(p.family)).sort((a,b)=>b.y-a.y);
for(let i=0;i<points.length;i++){const p=points[i],info=meta[p.model],color=fam.find(f=>f[0]===p.family)[2],cx=X(p.x),cy=Y(p.y),labelY=52+(i+.5)*390/points.length;let el;
switch(p.family){case 'ss_only':el=node('rect',{x:cx-5,y:cy-5,width:10,height:10});break;case 'gain_ss':el=node('path',{d:`M ${cx} ${cy-7} L ${cx+7} ${cy+5} L ${cx-7} ${cy+5} Z`});break;case 'difficult_gradnorm':el=node('path',{d:`M ${cx} ${cy-7} L ${cx+7} ${cy} L ${cx} ${cy+7} L ${cx-7} ${cy} Z`});break;case 'no_gain':el=node('path',{d:`M ${cx} ${cy-8} L ${cx+2} ${cy-2} L ${cx+8} ${cy-2} L ${cx+3} ${cy+2} L ${cx+5} ${cy+8} L ${cx} ${cy+4} L ${cx-5} ${cy+8} L ${cx-3} ${cy+2} L ${cx-8} ${cy-2} L ${cx-2} ${cy-2} Z`});break;default:el=node('circle',{cx,cy,r:5.5})}
let line=node('line',{x1:cx,y1:cy,x2:446,y2:labelY,stroke:`var(${color})`,class:'leader','data-model':p.model});svg.append(line);
el.setAttribute('class','dot');el.setAttribute('fill',`var(${color})`);el.setAttribute('tabindex','0');el.setAttribute('aria-label',`${info.full}: KCI ${fmt(p.x,2)}%, AvgRelRMSE ${fmt(p.y,4)}`);el.append(node('title',{},`${info.full}\nKCI ${fmt(p.x,2)}%\nAvgRelRMSE ${fmt(p.y,4)}`));bind(el,p.model);svg.append(el);
let text=node('text',{x:450,y:labelY+3,fill:`var(${color})`,class:'point-label',tabindex:'0'},`${number(p.model)} ${info.short}`);text.style.fill=`var(${color})`;text.setAttribute('aria-label',`${info.full}: KCI ${fmt(p.x,2)}%, AvgRelRMSE ${fmt(p.y,4)}`);bind(text,p.model);svg.append(text)}}
function key(){let grid=document.getElementById('model-keys');grid.replaceChildren();for(const m of models.filter(m=>visible.has(m.family))){let b=document.createElement('button');b.className='model-key';b.style.setProperty('--color',`var(${fam.find(f=>f[0]===m.family)[2]})`);let n=document.createElement('span');n.className='badge';n.textContent=number(m.model);b.append(n,document.createTextNode(m.full));b.setAttribute('aria-label',m.full);bind(b,m.model);grid.append(b)}}
function table(){let rows=document.getElementById('rows');rows.replaceChildren();for(const m of models.filter(m=>visible.has(m.family))){let a=data.points.find(p=>p.model===m.model&&p.partition==='internal'),b=data.points.find(p=>p.model===m.model&&p.partition==='external'),tr=document.createElement('tr');for(const v of [number(m.model)+' · '+m.full,labels[m.family],fmt(a.x,2),fmt(a.y,4),fmt(b.x,2),fmt(b.y,4)]){let td=document.createElement('td');td.textContent=v;tr.append(td)}tr.tabIndex=0;bind(tr,m.model);rows.append(tr)}}
function draw(){scatter('internal');scatter('external');key();table();highlight()}draw();})();
'''


def render_parameter_page(template: str, datasets: dict, root: Path) -> str:
    page = template
    def once(old, new):
        nonlocal page
        if page.count(old) != 1:
            raise ValueError(f'Unexpected template fragment: {old[:70]}')
        page = page.replace(old, new, 1)
    once('</style>', EXTRA_STYLE + '\n</style>')
    once('<span class="mut">seed 42 · 21 個模型</span>', '<span class="mut" id="seed-status">seed 42 · 21 個模型</span>')
    once('<section class="card"><h3>模型家族</h3>', '<section class="card"><h3>訓練 seed</h3><p class="description small">一次顯示一個 seed，兩張圖使用三個 seed 的共同座標範圍。</p><div class="filters stack" id="seed-selector" role="group" aria-label="選擇訓練 seed"></div></section>\n<section class="card stack"><h3>模型家族</h3>')
    once('可切換家族；將滑鼠移到點上或點選圖表及表格，查看模型數值。兩張圖使用相同座標範圍。', '每個點右側都有編號與參數簡寫；點選或移到圖上、對照表或結果表，可同時高亮兩區對應的點。ρ 為 Difficult-GradNorm 的目標梯度比例。手機可左右滑動圖表查看標籤。')
    once('<section class="card stack selection" id="selection" aria-live="polite"><span class="mut">選擇一個模型點，可查看兩區域的詳細數值。</span></section>', '<section class="card stack selection" id="selection" aria-live="polite"><span class="mut">選擇圖上的標籤、模型對照表或結果表，可同時標示兩區的對應點。</span></section>\n<section class="card stack"><h3>編號與完整參數</h3><p class="description small">NG：No-Gain；G：Gain；S：SS-only；GS：Gain+SS；D：Difficult-GradNorm（數字是 ρ）。點選可高亮圖上位置。</p><div id="model-keys" class="model-key-grid"></div></section>')
    page = page.replace('viewBox="0 0 520 380"', 'viewBox="0 0 600 520"')
    page, wrapped = re.subn(r'(<svg class="chart" id="(?:internal|external)"[^>]*></svg>)', r'<div class="chart-scroll">\1</div>', page)
    if wrapped != 2:
        raise ValueError('Expected two chart elements')
    once('相同案例的 seed 42 No-Gain RMSE', '相同案例及目前所選 seed 的 No-Gain RMSE')
    once('此圖只有 seed 42，尚未呈現三次重複訓練的變異。', '可切換 seed 42、43、44；每次僅顯示一個 seed，尚未彙整三次訓練的變異。')
    payload = json.dumps(dict(datasets=datasets, models=model_parameters(root)), ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    marker = '<script id="dataset" type="application/json">'
    head = page.split(marker, 1)[0]
    return head + marker + payload + '</script><script>\n' + INTERACTION + '\n</script></body></html>'
