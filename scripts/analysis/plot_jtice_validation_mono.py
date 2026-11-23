"""Parameter-faceted monochrome companion plots from frozen validation data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.analysis.plot_jtice_validation import BASE, OUT, ROOT, FAMILIES, SEEDS, MARKERS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    coord_file = BASE / 'regional_coordinates_63_models.csv'
    meta_file = BASE / 'model_parameter_metadata.csv'
    prov_file = BASE / 'provenance.json'
    prov = json.loads(prov_file.read_text(encoding='utf-8'))
    for p in (coord_file, meta_file):
        if sha(p) != prov['outputs_sha256'][p.name]:
            raise ValueError(f'Source changed: {p}')
    coords, meta = pd.read_csv(coord_file), pd.read_csv(meta_file)
    source = {str(p.relative_to(ROOT)): sha(p) for p in (coord_file, meta_file, prov_file)}
    dst = OUT / 'monochrome'
    dst.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    lower, upper = prov['plot_bounds']['y']; split = 1.35
    slope = (split-lower) / (2.6*(upper-split))
    def fwd(v):
        v=np.asarray(v); return np.where(v <= split, v, split+(v-split)*slope)
    def inv(v):
        v=np.asarray(v); return np.where(v <= split, v, split+(v-split)/slope)
    counts = {}
    for family, label in FAMILIES:
        fig, axes = plt.subplots(5, 2, figsize=(190/25.4, 210/25.4),
                                 gridspec_kw={'left':.09, 'right':.99, 'top':.94, 'bottom':.06,
                                              'hspace':.48, 'wspace':.2})
        choices = meta[meta.family.eq(family)].sort_values('value').to_dict('records')
        if len(choices) != 5:
            raise ValueError(f'Expected five settings: {family}')
        marks = []
        for row, item in enumerate(choices):
            for col, region in enumerate(('internal','external')):
                ax=axes[row,col]; ax.set_xlim(*prov['plot_bounds']['x']); ax.set_ylim(lower,upper)
                ax.set_yscale('function', functions=(fwd,inv)); ax.set_ylim(lower,upper)
                xlo,xhi=prov['plot_bounds']['x']
                xticks=sorted(set([v for v in (60,70,80,86,90,94,98,100) if xlo<=v<=xhi]))
                yticks=[v for v in (.4,.8,1.,1.35,2.1,3.7) if lower<=v<=upper]
                ax.set_xticks(xticks); ax.set_yticks(yticks)
                ax.set_yticklabels([f'{v:g}' for v in yticks])
                ax.set_xlim(xlo,xhi); ax.set_ylim(lower,upper)
                ax.spines[['top','right']].set_visible(False)
                ax.tick_params(labelsize=6.5, width=.55, length=2, pad=1.5)
                ax.axhline(1, color='.55', ls='--', lw=.6, zorder=1)
                ax.axhline(split, color='.75', ls=':', lw=.5, zorder=1)
                parameter = f"rho = {item['value']:g}" if family == 'difficult_gradnorm' else f"weight = {item['value']:g}"
                ax.set_title(f'({chr(97+row*2+col)}) {parameter} | '+('Interpolation' if col == 0 else 'Extrapolation'),
                             loc='left', fontsize=7, fontweight='bold', pad=3)
                if col == 0: ax.set_ylabel('AvgRelRMSE\n(nonlinear)', fontsize=7)
                if row == 4: ax.set_xlabel('KCI (%)', fontsize=7)
                for model, fill in ((item['model'], 'black'), ('no_gain', 'white')):
                    for seed in SEEDS:
                        r=coords[(coords.model==model)&(coords.seed==seed)&(coords.region==region)]
                        if len(r)!=1: raise ValueError('Missing or duplicate coordinate')
                        x,y=float(r.KCI_percent.iloc[0]),float(r.AvgRelRMSE.iloc[0])
                        ax.scatter(x,y,s=19,marker=MARKERS[seed],facecolor=fill,edgecolor='black',linewidth=.6,zorder=4)
                        marks.append((seed,model,region,x,y))
        handles=[Line2D([],[],marker=MARKERS[s],color='black',ls='',ms=4.5,label=f'Seed {s}') for s in SEEDS]
        handles += [Line2D([],[],marker='o',color='black',markerfacecolor='white',ls='',ms=4.5,label='No-Gain')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,
                   frameon=False,fontsize=7,columnspacing=1.4,handletextpad=.4)
        expected={(int(r.seed),r.model,r.region,float(r.KCI_percent),float(r.AvgRelRMSE))
                  for r in coords.itertuples(index=False) if r.family==family or r.model=='no_gain'}
        if len(marks)!=60 or len(set(marks))!=36 or set(marks)!=expected:
            raise ValueError(f'Wrong coordinate set for {family}')
        counts[family]={'drawn_markers':len(marks),'unique_coordinates':len(set(marks))}
        for ext in ('png','pdf'):
            fig.savefig(dst/f'validation_{family}_mono.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
        caption = (f'Validation comparison of {label}. Each row identifies a weight or target ratio; '
                   'the left and right panels show interpolation and extrapolation, respectively. '
                   'Marker shapes indicate training seeds 42, 43 and 44; filled markers show the model '
                   'and hollow markers show No-Gain. The three No-Gain seeds are repeated in each row. '
                   'AvgRelRMSE is normalized to No-Gain seed 42 in the corresponding region; KCI is '
                   'directional agreement with the shared ANN reference. A shared piecewise-linear y axis '
                   'changes slope at AvgRelRMSE = 1.35; all plotted values are retained.')
        (dst/f'validation_{family}_mono_caption.txt').write_text(caption+'\n', encoding='utf-8')
    if any(sha(ROOT/name)!=digest for name,digest in source.items()):
        raise ValueError('Source changed')
    produced = [dst/f'validation_{family}_mono.{ext}' for family, _ in FAMILIES for ext in ('png','pdf')]
    produced += [dst/f'validation_{family}_mono_caption.txt' for family, _ in FAMILIES]
    manifest={'sources':source,'script_sha256':sha(Path(__file__)),'family_counts':counts,
              'encoding':'parameter identified by panel title; seed by marker shape; No-Gain is hollow',
              'output_sha256':{p.name:sha(p) for p in produced},
              'print_dimensions_mm':[190,210],'status':'JTICE user-provided guide snapshot; verify current instructions'}
    (dst/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
