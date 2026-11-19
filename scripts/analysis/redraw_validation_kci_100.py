"""Redraw existing validation figures with a labelled 100% KCI tick.

Reads already verified coordinates and metadata; does not run inference or alter
historical source provenance. Output directory must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.analysis.build_expanded_dg_validation_report import FAMILIES, COLORS, MARKERS, SEEDS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def figure_with_kci_100(data,models,out,start,stop,stem,expanded,xlim,ylim):
    families=FAMILIES[start:stop];rows=len(families)
    fig=plt.figure(figsize=(9.2,rows*2.8+.9))
    grid=fig.add_gridspec(rows,3,width_ratios=(1,1,.31),left=.08,right=.99,top=.86,
                          bottom=.055 if rows==4 else .085,wspace=.16,hspace=.40)
    split=1.35; lower,upper=ylim
    if expanded:
        if not lower < split < upper: raise ValueError('Expanded scale split is outside y limits')
        slope=(split-lower)/(2.6*(upper-split))
        def forward(v):
            v=np.asarray(v);return np.where(v<=split,v,split+(v-split)*slope)
        def inverse(v):
            v=np.asarray(v);return np.where(v<=split,v,split+(v-split)/slope)
    plotted=[]
    for row,(family,name) in enumerate(families):
        index=start+row; choices=sorted((m for m in models if m['family']==family),key=lambda m:m['value'])
        if len(choices)!=5: raise ValueError(f'Expected five parameters for {family}')
        for col,region in enumerate(('internal','external')):
            ax=fig.add_subplot(grid[row,col]);ax.set_xlim(*xlim);ax.set_ylim(*ylim)
            ax.set_xticks([tick for tick in (86, 90, 94, 98, 100) if xlim[0] <= tick <= xlim[1]])
            if expanded:
                ax.set_yscale('function',functions=(forward,inverse));ax.set_ylim(*ylim)
                lo=[v for v in (.4,.6,.8,1.,1.2,1.35) if lower<=v<=upper]
                hi=[round(float(v),1) for v in np.linspace(split,upper,4)[1:] if round(float(v),1)<=upper]
                ticks=sorted(set(lo+hi));ax.set_yticks(ticks);ax.set_yticklabels([f'{x:g}' for x in ticks])
                ax.axhline(split,color='#999999',linewidth=.65,linestyle=':',zorder=1)
            ax.set_title(f'({"abcdefgh"[index*2+col]})  {name}  |  {"Interpolation" if col==0 else "Extrapolation"}',loc='left',fontsize=9,fontweight='bold',pad=6)
            ax.axhline(1,color='#777777',linewidth=.7,linestyle='--',zorder=1)
            ax.grid(color='#d4d4d4',linewidth=.45,alpha=.7,zorder=0)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=7.5,length=3,pad=2)
            if col==0: ax.set_ylabel('AvgRelRMSE',fontsize=9)
            if row==rows-1: ax.set_xlabel('KCI(%)',fontsize=9)
            subset=data[data.region.eq(region)]
            for item,color in [(m,c) for m,c in zip(choices,COLORS)]+[(dict(model='no_gain'),'#555555')]:
                for seed in SEEDS:
                    point=subset[subset.model.eq(item['model']) & subset.seed.eq(seed)]
                    if len(point)!=1: raise ValueError('Plot data missing a model/seed/region')
                    x=float(point.KCI_percent.iloc[0]);y=float(point.AvgRelRMSE.iloc[0])
                    if not(xlim[0]<=x<=xlim[1] and ylim[0]<=y<=ylim[1]): raise ValueError('Plot limit omits a point')
                    ax.scatter(x,y,marker=MARKERS[seed],s=35 if item['model']=='no_gain' else 32,facecolor=color,edgecolor='white',linewidth=.5,zorder=4)
                    plotted.append((family,region,item['model'],seed,x,y))
        legend=fig.add_subplot(grid[row,2]);legend.axis('off')
        legend.text(.03,.98,'ρ' if family=='difficult_gradnorm' else 'Loss weight',fontsize=8,fontweight='bold',va='top',transform=legend.transAxes)
        for j,(item,color) in enumerate(zip(choices,COLORS)):
            yy=.79-j*.135;legend.plot([.14],[yy],color=color,marker='o',markersize=6,linestyle='',transform=legend.transAxes)
            legend.text(.28,yy,f"{item['value']:g}",fontsize=8,va='center',transform=legend.transAxes)
    handles=[Line2D([],[],marker=MARKERS[s],color='#555555',markersize=6,ls='',markeredgecolor='white',markeredgewidth=.4,label=f'Seed {s}') for s in SEEDS]
    handles.append(Line2D([],[],marker='o',color='#555555',markersize=6,ls='',markeredgecolor='white',markeredgewidth=.4,label='No-Gain reference'))
    fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(.5,.985),handletextpad=.4,columnspacing=1.3)
    fig.text(.5,.938,'Baseline: No-Gain seed 42',fontsize=8,ha='center',va='center')
    fig.text(.5,.902,'Common nonlinear Y: expanded below 1.35; no values omitted' if expanded else 'Common linear X and Y scales across all panels',fontsize=8,ha='center',va='center')
    if len(plotted)!=rows*2*18: raise ValueError('Unexpected number of displayed markers')
    for extension in ('png','pdf'): fig.savefig(out/f'{stem}.{extension}',dpi=350,facecolor='white')
    plt.close(fig)
    return plotted



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must be inside workspace')
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    provenance_path = BASE / 'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    coords_path = BASE / 'regional_coordinates_63_models.csv'
    meta_path = BASE / 'model_parameter_metadata.csv'
    for path in (coords_path, meta_path):
        if sha(path) != provenance['outputs_sha256'][path.name]:
            raise ValueError(f'Input hash mismatch: {path}')
    inputs_before = {str(p.relative_to(ROOT)): sha(p) for p in (coords_path, meta_path, provenance_path, BASE / 'regional_validation_nogain42_by_seed.html')}
    coords = pd.read_csv(coords_path)
    models = pd.read_csv(meta_path).to_dict('records')
    bounds = provenance['plot_bounds']
    xlim, ylim = bounds['x'], bounds['y']
    if not xlim[0] <= 100 <= xlim[1]:
        raise ValueError('Existing plot limits do not include 100')
    out.mkdir(parents=True)
    figures = out / 'figures'
    figures.mkdir()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for expanded in (False, True):
        suffix = 'expanded_y' if expanded else 'linear'
        for start, stop, prefix in ((0, 4, 'nogain42_4x2'), (0, 2, 'nogain42_part1'), (2, 4, 'nogain42_part2')):
            figure_with_kci_100(coords, models, figures, start, stop, f'{prefix}_{suffix}', expanded, xlim, ylim)
    page = (BASE / 'regional_validation_nogain42_by_seed.html').read_text(encoding='utf-8')
    old = 'Math.min(100,hi+p)'
    if page.count(old) == 1:
        page = page.replace(old, '100', 1)
    elif page.count('y?hi+p:100') != 1:
        raise ValueError('Unexpected HTML axis expression')
    (out / 'regional_validation_nogain42_by_seed.html').write_text(page, encoding='utf-8', newline='')
    for path, digest in inputs_before.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Input changed: {path}')
    original = {name: digest for name, digest in provenance['outputs_sha256'].items() if name.startswith('figures\\') or name == 'regional_validation_nogain42_by_seed.html'}
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest = {
        'purpose': 'Replot existing validation results to label KCI 100 percent; HTML x axis ends at 100.',
        'no_inference_or_data_change': True,
        'historical_provenance_preserved': True,
        'original_chart_output_hashes_in_historical_provenance_superseded': original,
        'current_chart_output_sha256': outputs,
        'source_sha256': inputs_before,
        'revision_script_sha256': sha(Path(__file__)),
        'static_kci_ticks': [86, 90, 94, 98, 100],
        'plot_bounds_unchanged': bounds,
    }
    (out / 'axis_revision_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'Created {len(outputs)} figures and HTML in {out}')

if __name__ == '__main__':
    main()
