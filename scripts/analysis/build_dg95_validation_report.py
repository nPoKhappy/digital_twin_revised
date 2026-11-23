"""Replace the DG portion of the fixed-No-Gain-42 validation report."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.analysis import build_expanded_dg_validation_report as prior
from scripts.analysis import plot_jtice_validation as journal

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
DG = ROOT / 'results/visualization/retrain_lhs10_23_dg95_validation'
REF = DG / 'reference_non_dg'
STAGE = DG / 'report_staging'
SEEDS = (42, 43, 44)
ALIASES = ('dg_ratio0p015', 'dg_ratio0p15', 'dg_ratio1p5', 'dg_ratio15', 'dg_ratio150')
CAPS = dict(zip(ALIASES, (1000, 20000, 500000, 10000000, 500000000)))
KEYS = ['partition', 'case', 'target', 'evaluation']


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot():
    names = ['regional_coordinates_63_models.csv', 'relative_rmse_components.csv',
             'kci_components.csv', 'baseline_raw_rmse.csv', 'model_parameter_metadata.csv']
    prov = REF/'previous_provenance.json'
    old = json.loads((prov if prov.exists() else BASE/'provenance.json').read_text(encoding='utf8'))
    for name in names:
        source = REF/name if prov.exists() else BASE/name
        if sha(source) != old['outputs_sha256'][name]:
            raise ValueError(f'Old report mismatch: {name}')
    REF.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = BASE/name
        dest = REF/name
        if not dest.exists():
            shutil.copyfile(source, dest)
    if not prov.exists():
        shutil.copyfile(BASE/'provenance.json', prov)


def main():
    checked = json.loads((DG/'evaluation_validation.json').read_text(encoding='utf8'))
    proto = json.loads((DG/'run_protocol.json').read_text(encoding='utf8'))
    if not (checked['passed'] and checked['model_count'] == 15 and
            checked['dynamic_metrics'] == 540 and checked['kci_components'] == 120):
        raise ValueError('Incomplete DG validation')
    for path, digest in proto['sources_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'DG input changed: {path}')
    reference_dir=ROOT/'results/visualization/retrain_lhs10_23_21_models_validation_seed42/model_data/no_gain'
    marker_path=reference_dir/'complete.json'
    reference_marker=json.loads(marker_path.read_text(encoding='utf8'))
    reference_gains=[reference_dir/'gain_in_range.npz',reference_dir/'gain_soft_ood.npz']
    for path in reference_gains:
        if reference_marker['outputs'].get(path.name)!=sha(path):
            raise ValueError(f'Reference gain NPZ differs from its complete marker: {path}')
    snapshot()
    old_metrics = pd.read_csv(REF/'relative_rmse_components.csv')
    old_coords = pd.read_csv(REF/'regional_coordinates_63_models.csv')
    old_kci = pd.read_csv(REF/'kci_components.csv')
    baseline = pd.read_csv(REF/'baseline_raw_rmse.csv')
    meta = pd.read_csv(REF/'model_parameter_metadata.csv')
    if len(meta) != 21 or baseline.duplicated(KEYS).any() or len(baseline) != 36:
        raise ValueError('Invalid historical metadata/baseline')
    meta.loc[meta.family.eq('difficult_gradnorm'), 'max_scale'] = meta.loc[
        meta.family.eq('difficult_gradnorm'), 'model'].map(CAPS)
    if meta.loc[meta.family.eq('difficult_gradnorm'), 'max_scale'].isna().any():
        raise ValueError('Missing DG metadata')
    retained = old_metrics[~old_metrics.family.eq('difficult_gradnorm')].copy()
    kci_old = old_kci[~old_kci.family.eq('difficult_gradnorm')].copy()
    if len(retained) != 48*36 or len(kci_old) != 48*8:
        raise ValueError('Wrong non-DG reference coverage')
    new = pd.read_csv(DG/'validation_metrics.csv')
    new = new.merge(baseline, on=KEYS, validate='many_to_one')
    if len(new) != 15*36 or not new.n.eq(new.NoGain_seed42_n).all() or not new.NoGain_seed42_RMSE.gt(0).all():
        raise ValueError('Wrong DG RMSE reference alignment')
    new['relative_RMSE'] = new.RMSE/new.NoGain_seed42_RMSE
    new['family'] = 'difficult_gradnorm'
    cols = ['seed','model','family',*KEYS,'n','RMSE','NoGain_seed42_n','NoGain_seed42_RMSE','relative_RMSE']
    components = pd.concat([retained[cols], new[cols]], ignore_index=True)
    if len(components) != 63*36 or components.duplicated(['seed','model',*KEYS]).any():
        raise ValueError('Wrong component count')
    if not np.isfinite(components.relative_RMSE).all() or not components.relative_RMSE.gt(0).all():
        raise ValueError('Nonpositive relative RMSE')
    kci = pd.concat([kci_old, pd.read_csv(DG/'kci_components.csv')], ignore_index=True)
    if len(kci) != 63*8 or kci.duplicated(['seed','model','distribution','target','mv']).any():
        raise ValueError('Wrong KCI count')
    if not kci.groupby(['distribution','target','mv']).valid_comparisons.nunique().eq(1).all():
        raise ValueError('KCI denominators differ')
    kci['correct'] = kci.tp+kci.tn
    grouped = kci.groupby(['seed','model','distribution'], as_index=False).agg(
        correct=('correct','sum'), valid=('valid_comparisons','sum'))
    grouped['KCI_percent'] = 100*grouped.correct/grouped.valid
    grouped['region'] = grouped.distribution.map({'in_range':'internal','soft_ood':'external'})
    regions = components.groupby(['seed','model','family','partition'], as_index=False).agg(
        AvgRelRMSE=('relative_RMSE',lambda x: float(np.exp(np.log(x).mean()))),
        component_count=('relative_RMSE','size'))
    if not regions.component_count.eq(18).all():
        raise ValueError('Expected 18 error ratios per region')
    coords = regions.rename(columns={'partition':'region'}).merge(grouped, on=['seed','model','region'], validate='one_to_one')
    coords = coords[['seed','model','family','region','KCI_percent','AvgRelRMSE','correct','valid']]
    old_non_dg = old_coords[~old_coords.family.eq('difficult_gradnorm')]
    matched = coords[~coords.family.eq('difficult_gradnorm')].merge(old_non_dg, on=['seed','model','family','region'], suffixes=('_new','_old'), validate='one_to_one')
    if len(matched) != 96:
        raise ValueError('Non-DG coordinate count differs')
    for col in ('KCI_percent','AvgRelRMSE','correct','valid'):
        np.testing.assert_allclose(matched[f'{col}_new'],matched[f'{col}_old'],rtol=1e-12,atol=1e-12)
    wide=coords.pivot(index=['seed','model','family'],columns='region',values='AvgRelRMSE').reset_index()
    wide['overall_50_50']=np.sqrt(wide.internal*wide.external)
    wide=wide.sort_values(['overall_50_50','model','seed']).reset_index(drop=True)
    wide.insert(0,'rank_single_run',range(1,64))
    summary=wide.groupby(['family','model'],as_index=False).agg(
        mean=('overall_50_50','mean'),sd=('overall_50_50','std'),
        minimum=('overall_50_50','min'),maximum=('overall_50_50','max'),n_seeds=('seed','nunique'))
    summary=summary.sort_values(['mean','model']).reset_index(drop=True)
    summary.insert(0,'rank_mean',range(1,22))
    if len(coords)!=126 or len(summary)!=21 or not summary.n_seeds.eq(3).all():
        raise ValueError('Wrong summary coverage')
    STAGE.mkdir(parents=True,exist_ok=True)
    fig_dir=STAGE/'figures';fig_dir.mkdir(exist_ok=True)
    frames=[(baseline,'baseline_raw_rmse.csv'),(components,'relative_rmse_components.csv'),
            (kci,'kci_components.csv'),(coords,'regional_coordinates_63_models.csv'),
            (wide,'scores_by_model_seed.csv'),(summary,'scores_by_family_model.csv'),
            (meta,'model_parameter_metadata.csv')]
    for frame,name in frames:
        frame.to_csv(STAGE/name,index=False,float_format='%.17g')
    models=meta.astype(object).where(meta.notna(),None).to_dict('records')
    html=prior.make_html(coords,kci,models)
    old_axis='[Math.max(0,lo-p),y?hi+p:Math.min(100,hi+p)]'
    if html.count(old_axis)!=1:
        raise ValueError('Could not locate interactive x-axis bound')
    html=html.replace(old_axis,'[Math.max(0,lo-p),y?hi+p:100]')
    (STAGE/'regional_validation_nogain42_by_seed.html').write_text(html,encoding='utf8')
    xlim=(max(0.,min(85.8,math.floor(float(coords.KCI_percent.min()))-1)),100.2)
    ylim=(max(0.,min(.35,math.floor(float(coords.AvgRelRMSE.min())*.9*20)/20)),
          max(2.85,math.ceil(float(coords.AvgRelRMSE.max())*1.05*20)/20))
    journal.OUT=fig_dir
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42})
    bounds={'x':xlim,'y':ylim}
    marks=journal.draw(coords,meta,journal.FAMILIES,'nogain42_4x2_linear',False,bounds)
    for start,stop,stem in [(0,2,'nogain42_part1_linear'),(2,4,'nogain42_part2_linear')]:
        journal.draw(coords,meta,journal.FAMILIES[start:stop],stem,False,bounds)
    journal.draw(coords,meta,journal.FAMILIES,'nogain42_4x2_expanded_y',True,bounds)
    for start,stop,stem in [(0,2,'nogain42_part1_expanded_y'),(2,4,'nogain42_part2_expanded_y')]:
        journal.draw(coords,meta,journal.FAMILIES[start:stop],stem,True,bounds)
    if len(marks)!=144:
        raise ValueError('Wrong marker count')
    report=dict(passed=True,model_count=63,regional_points=126,components=2268,kci_components=504,
                seeds=list(SEEDS),reused_historical_models=48,newly_evaluated_dg_models=15,
                baseline_seed=42,retained_coordinates_checked=96,same_x_y_axes_all_panels=True,
                no_jitter=True,optimizer_updates=0)
    (STAGE/'analysis_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    readme=('# DG 95% 上限重訓後的 validation\n\n'
            '63 個模型包含 48 個沿用的非 DG 模型與 15 個重新驗證的 DG 模型。'
            '各區域以 No-Gain seed 42 為共同基準，對 18 個同案例、輸出與時域的 RMSE 比值取幾何平均。'
            'KCI 是分別在穩態 in-range 與 soft-OOD 探點上計算的 ANN 參考方向一致率，與動態 RMSE 資料不同。'
            'R5-5 約 33.19% 的原始時間點超出核心範圍。\n\n'
            '48 個沿用模型的輸出已對照舊報告雜湊並保留快照；先前記錄的 5 個歷史來源已不存在，因此不宣稱全部上游來源均可核對。'
            '新選模須依此 validation 執行；原 test 在重跑前仍是歷史結果。'
            'interactive_dom_validation.json 僅檢查新版 HTML 內嵌資料與 CSV 一致，未宣稱實際瀏覽器點擊測試。\n')
    (STAGE/'README.md').write_text(readme,encoding='utf8')
    old_dom=json.loads((BASE/'interactive_dom_validation.json').read_text(encoding='utf8'))
    old_dom=old_dom.get('historical_record',old_dom)
    html=(STAGE/'regional_validation_nogain42_by_seed.html').read_text(encoding='utf8')
    match=re.search(r'<script id="dataset" type="application/json">(.*?)</script>',html,re.S)
    if not match: raise ValueError('Missing interactive dataset')
    payload=json.loads(match.group(1))
    if len(payload['models'])!=21 or set(payload['datasets'])!={'42','43','44'}:
        raise ValueError('Interactive model/seed roster differs')
    html_points=set()
    for seed, dataset in payload['datasets'].items():
        if len(dataset['points'])!=42 or len(dataset['details'])!=168:
            raise ValueError(f'Wrong interactive rows: seed {seed}')
        for p in dataset['points']:
            html_points.add((int(seed),p['model'],p['partition'],p['x'],p['y'],p['correct'],p['valid']))
    expected={(int(r.seed),r.model,r.region,float(r.KCI_percent),float(r.AvgRelRMSE),int(r.correct),int(r.valid))
              for r in coords.itertuples(index=False)}
    if html_points!=expected or len(html_points)!=126:
        raise ValueError('Interactive data differ from coordinates')
    (STAGE/'interactive_dom_validation.json').write_text(json.dumps({
        'status':'current_embedded_data_verified','passed':True,'points_checked':126,
        'note':'Current embedded coordinates checked against CSV; no browser rendering/click test is claimed.',
        'html_sha256':sha(STAGE/'regional_validation_nogain42_by_seed.html'),
        'historical_record':old_dom},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    current_figures={str(p.relative_to(STAGE)).replace('\\','/'):sha(p) for p in fig_dir.glob('*') if p.is_file()}
    current_figures['regional_validation_nogain42_by_seed.html']=sha(STAGE/'regional_validation_nogain42_by_seed.html')
    (STAGE/'axis_revision_manifest.json').write_text(json.dumps({
        'purpose':'DG95 validation redraw with KCI 100% tick and shared plot bounds',
        'current_chart_output_sha256':current_figures,'bounds':bounds},indent=2)+'\n',encoding='utf8')
    audit_path=STAGE/'independent_expanded_validation_audit.json'
    if audit_path.exists(): audit_path.unlink()
    subprocess.run([sys.executable,'-B','-m','scripts.analysis.audit_dg95_validation',
                    '--report',str(audit_path)],cwd=ROOT,check=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [*REF.glob('*.csv'),REF/'previous_provenance.json',
             DG/'run_protocol.json',DG/'evaluation_validation.json',DG/'validation_metrics.csv',DG/'kci_components.csv',
             marker_path,*reference_gains,
             Path(__file__), ROOT/'scripts/analysis/build_expanded_dg_validation_report.py',
             ROOT/'scripts/analysis/plot_jtice_validation.py',ROOT/'scripts/analysis/audit_dg95_validation.py',
             ROOT/'scripts/plotting/retrain_parameter_ui.py',ROOT/'scripts/plotting/render_retrain_validation_html.py']}
    outputs={str(p.relative_to(STAGE)).replace('\\','/'):sha(p) for p in STAGE.rglob('*') if p.is_file() and p.name!='provenance.json'}
    provenance=dict(sources_sha256=sources,outputs_sha256=outputs,baseline_seed=42,
                    reused_historical_runs=48,new_dg_runs=15,plot_bounds=dict(x=xlim,y=ylim),
                    DG_max_scale=CAPS,jitter=False,
                    note='Prior non-DG coordinates and baseline snapshot checked against old provenance.')
    (STAGE/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'Source changed: {name}')
    # Replace only produced report artifacts, retaining unrelated existing files.
    files=[p for p in STAGE.rglob('*') if p.is_file() and p.name!='provenance.json']
    for path in files + [STAGE/'provenance.json']:
        dest=BASE/path.relative_to(STAGE)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        if sha(dest)!=sha(path):
            raise ValueError(f'Published output differs: {dest}')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
