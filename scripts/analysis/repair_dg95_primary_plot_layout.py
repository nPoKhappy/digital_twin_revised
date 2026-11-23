"""Prepare and publish corrected primary validation plots without touching data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from scripts.analysis import plot_jtice_validation as journal

ROOT=Path(__file__).resolve().parents[2]
BASE=journal.BASE
STAGE=ROOT/'results/visualization/retrain_lhs10_23_dg95_validation/primary_plot_layout'


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    names=('regional_coordinates_63_models.csv','model_parameter_metadata.csv','provenance.json')
    sources={name:sha(BASE/name) for name in names}
    prov=json.loads((BASE/'provenance.json').read_text(encoding='utf8'))
    for name in names[:2]:
        if sources[name]!=prov['outputs_sha256'][name]:
            raise ValueError(f'Source mismatch: {name}')
    coords=pd.read_csv(BASE/names[0]);meta=pd.read_csv(BASE/names[1])
    STAGE.mkdir(parents=True,exist_ok=True)
    journal.OUT=STAGE
    original=Figure.add_gridspec
    def spaced(self,nrows=1,ncols=1,**kwargs):
        if nrows in (2,4) and ncols==3 and kwargs.get('top')==.94:
            kwargs['top']=.87 if nrows==2 else .90
        return original(self,nrows,ncols,**kwargs)
    Figure.add_gridspec=spaced
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42})
    try:
        for expanded in (False,True):
            scale='expanded_y' if expanded else 'linear'
            for subset,label in ((journal.FAMILIES,'4x2'),(journal.FAMILIES[:2],'part1'),(journal.FAMILIES[2:],'part2')):
                marks=journal.draw(coords,meta,subset,f'nogain42_{label}_{scale}',expanded,prov['plot_bounds'])
                if len(marks)!=len(subset)*36:
                    raise ValueError('Wrong plotted marker count')
    finally:
        Figure.add_gridspec=original
    files=list(STAGE.glob('*.png'))+list(STAGE.glob('*.pdf'))
    if len(files)!=12: raise ValueError('Expected 12 images')
    manifest={'status':'prepared','sources_sha256':sources,
              'script_sha256':sha(Path(__file__)),
              'output_sha256':{p.name:sha(p) for p in files}}
    (STAGE/'prepared.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,indent=2))


def publish():
    prepared=json.loads((STAGE/'prepared.json').read_text(encoding='utf8'))
    if prepared['script_sha256']!=sha(Path(__file__)):
        raise ValueError('Plot repair script changed after preparation')
    for name,digest in prepared['sources_sha256'].items():
        if sha(BASE/name)!=digest:
            raise ValueError(f'Validation source changed during plot preparation: {name}')
    for name,digest in prepared['output_sha256'].items():
        if sha(STAGE/name)!=digest: raise ValueError(f'Prepared figure changed: {name}')
    prov=json.loads((BASE/'provenance.json').read_text(encoding='utf8'))
    axis=json.loads((BASE/'axis_revision_manifest.json').read_text(encoding='utf8'))
    for name,digest in prepared['output_sha256'].items():
        key='figures/'+name
        prov['outputs_sha256'][key]=digest
        axis['current_chart_output_sha256'][key]=digest
    axis['plot_layout']='Top legend separated from subplot titles; coordinates and axes unchanged.'
    axis_content=json.dumps(axis,indent=2)+'\n'
    prov['outputs_sha256']['axis_revision_manifest.json']=hashlib.sha256(axis_content.encode('utf8')).hexdigest()
    prov['sources_sha256'][str(Path(__file__).relative_to(ROOT)).replace('\\','/')]=sha(Path(__file__))
    raw=ROOT/'results/visualization/retrain_lhs10_23_dg95_validation'
    protocol_sha=sha(raw/'run_protocol.json')
    markers=sorted((raw/'model_data').glob('seed*/dg_ratio*/complete.json'))
    if len(markers)!=15:
        raise ValueError('Expected 15 DG validation completion markers')
    for marker in markers:
        data=json.loads(marker.read_text(encoding='utf8'))
        if data.get('run_protocol_sha256')!=protocol_sha:
            raise ValueError(f'DG protocol mismatch: {marker}')
        for name,digest in data['outputs'].items():
            if sha(marker.parent/name)!=digest:
                raise ValueError(f'DG model output mismatch: {marker.parent/name}')
        prov['sources_sha256'][str(marker.relative_to(ROOT)).replace('\\','/')]=sha(marker)
    prov['figure_layout_revision']={'previous_provenance_sha256':prepared['sources_sha256']['provenance.json'],
                                   'coordinate_values_changed':False,'figures_updated':12}
    for name,digest in prepared['output_sha256'].items():
        target=BASE/'figures'/name
        shutil.copyfile(STAGE/name,target)
        if sha(target)!=digest: raise ValueError(f'Published figure mismatch: {target}')
    (BASE/'axis_revision_manifest.json').write_text(axis_content,encoding='utf8',newline='')
    (BASE/'provenance.json').write_text(json.dumps(prov,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    for name,digest in prov['outputs_sha256'].items():
        if sha(BASE/name)!=digest: raise ValueError(f'Output mismatch after publication: {name}')
    print('Published 12 corrected figures and verified validation provenance.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    (publish if parser.parse_args().publish else prepare)()
