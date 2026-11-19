"""Independently audit selected-model test results from saved prediction arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGIONS = {'internal': ('in_range', ('R5', 'R5-1', 'R5-2', 'R5-6')),
           'external': ('soft_ood', tuple(f'R5-{i}' for i in range(19, 27)))}
TARGETS = ('B35_H2S', 'B35_SO2')
MVS = ('air2_SP', 'HEATER2_output_T_SP')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='Optional path to save the independent audit report')
    args = parser.parse_args()
    base = args.directory.resolve()
    report = json.loads((base/'evaluation_validation.json').read_text(encoding='utf-8-sig'))
    selected = json.loads((base/'selected_models.json').read_text(encoding='utf-8-sig'))['selected_models']
    if not (report['passed'] and report['model_count'] == 15 and len(selected) == 15):
        raise ValueError('Selected test evaluation is not complete')
    metrics = pd.read_csv(base/'test_metrics.csv')
    horizons = pd.read_csv(base/'horizon_metrics.csv')
    counts = pd.read_csv(base/'kci_components.csv')
    coords = pd.read_csv(base/'regional_coordinates.csv')
    pairs = pd.read_csv(base/'relative_rmse_components.csv')
    scores = pd.read_csv(base/'scores_by_model_seed.csv')
    if (len(metrics),len(horizons),len(counts),len(coords),len(pairs),len(scores)) != (1080,6480,120,30,1080,15):
        raise ValueError('Unexpected aggregate table coverage')
    protocol_sha = sha(base/'run_protocol.json')
    keys = ['model','partition','case','target','evaluation']
    m = metrics.set_index(keys,verify_integrity=True)
    h = horizons.set_index(['model','partition','case','target','horizon'],verify_integrity=True)
    pair = pairs.set_index(keys,verify_integrity=True)
    c = counts.set_index(['model','distribution','target','mv'],verify_integrity=True)
    coord = coords.set_index(['model','partition'],verify_integrity=True)
    score = scores.set_index('model',verify_integrity=True)
    max_rmse=max_horizon=max_relative=max_kci=0.0
    base_dir = base/'model_data'/'no_gain_seed42'
    for item in selected:
        name = item['model']; folder = base/'model_data'/name
        complete = json.loads((folder/'complete.json').read_text(encoding='utf-8-sig'))
        if complete['protocol_sha256'] != protocol_sha:
            raise ValueError(f'Protocol hash mismatch: {name}')
        for filename, digest in complete['outputs'].items():
            if sha(folder/filename) != digest:
                raise ValueError(f'Output hash mismatch: {name}/{filename}')
        region_values = {}
        for partition,(distribution,cases) in REGIONS.items():
            with np.load(folder/f'gain_{distribution}.npz') as data:
                teacher, dynamic = data['teacher_gain'], data['dynamic_gain']
                if dynamic.shape != (2,18,len(teacher),2,2):
                    raise ValueError(f'Invalid gain array shape: {name}')
                total_correct = total_valid = 0
                for i,target in enumerate(TARGETS):
                    for j,mv in enumerate(MVS):
                        truth=np.broadcast_to(teacher[:,i,j],dynamic[:,:,:,i,j].shape)
                        pred=dynamic[:,:,:,i,j]; mask=np.abs(truth)>=1e-5
                        actual={'possible_comparisons':mask.size, 'valid_comparisons':mask.sum(),
                                'tp':((truth>0)&(pred>0)&mask).sum(), 'tn':((truth<0)&(pred<0)&mask).sum(),
                                'fp':((truth<0)&(pred>0)&mask).sum(), 'fn':((truth>0)&(pred<0)&mask).sum(),
                                'zero_gain':((pred==0)&mask).sum()}
                        record = c.loc[(name,distribution,target,mv)]
                        for k,v in actual.items():
                            if int(record[k]) != int(v): raise ValueError(f'KCI count mismatch: {name} {partition} {target} {mv} {k}')
                        total_correct += int(actual['tp']+actual['tn']); total_valid += int(actual['valid_comparisons'])
                point=coord.loc[(name,partition)]
                if int(point.correct)!=total_correct or int(point.valid)!=total_valid:
                    raise ValueError(f'KCI total mismatch: {name} {partition}')
                max_kci=max(max_kci,abs(float(point.KCI_percent)-100*total_correct/total_valid))
            ratios=[]
            for case in cases:
                with np.load(folder/f'{case}_prediction.npz') as data, np.load(base_dir/f'{case}_prediction.npz') as baseline:
                    for j,target in enumerate(TARGETS):
                        np.testing.assert_allclose(data['ts_true'][:,j],baseline['ts_true'][:,j],rtol=1e-7,atol=1e-12)
                        np.testing.assert_allclose(data['horizon_true'][:,:,j],baseline['horizon_true'][:,:,j],rtol=1e-7,atol=1e-12)
                        for horizon in range(18):
                            truth=data['horizon_true'][:,horizon,j]; pred=data['horizon_pred'][:,horizon,j]
                            err=np.linalg.norm(truth-pred)/np.sqrt(len(truth))
                            row=h.loc[(name,partition,case,target,horizon+1)]
                            if int(row.n)!=len(truth): raise ValueError('Horizon sample count mismatch')
                            max_horizon=max(max_horizon,abs(float(row.RMSE)-err))
                        for evaluation in ('H1','H18','TS'):
                            if evaluation=='TS':
                                truth,pred=data['ts_true'][:,j],data['ts_pred'][:,j]
                                bt,bp=baseline['ts_true'][:,j],baseline['ts_pred'][:,j]
                            else:
                                index=int(evaluation[1:])-1
                                truth,pred=data['horizon_true'][:,index,j],data['horizon_pred'][:,index,j]
                                bt,bp=baseline['horizon_true'][:,index,j],baseline['horizon_pred'][:,index,j]
                            err=np.linalg.norm(truth-pred)/np.sqrt(len(truth)); reference=np.linalg.norm(bt-bp)/np.sqrt(len(bt))
                            if reference<=0: raise ValueError('No-Gain baseline must be positive')
                            key=(name,partition,case,target,evaluation); row=m.loc[key]; relative=pair.loc[key]
                            if int(row.n)!=len(truth) or int(relative.NoGain42_n)!=len(truth): raise ValueError('Sample count mismatch')
                            max_rmse=max(max_rmse,abs(float(row.RMSE)-err))
                            max_rmse=max(max_rmse,abs(float(relative.NoGain42_RMSE)-reference))
                            max_relative=max(max_relative,abs(float(relative.relative_RMSE)-err/reference))
                            ratios.append(err/reference)
            expected_len=24 if partition=='internal' else 48
            if len(ratios)!=expected_len or not min(ratios)>0: raise ValueError('Invalid RMSE components')
            avg=float(np.exp(np.mean(np.log(ratios))))
            point=coord.loc[(name,partition)]
            if int(point.components)!=expected_len: raise ValueError('Invalid component count')
            max_relative=max(max_relative,abs(float(point.AvgRelRMSE)-avg))
            region_values[partition]=avg
        max_relative=max(max_relative,abs(float(score.loc[name,'overall_50_50'])-np.sqrt(region_values['internal']*region_values['external'])))
    if max_rmse>=1e-12 or max_horizon>=1e-12 or max_relative>=1e-9 or max_kci>=1e-9:
        raise ValueError('Numeric recomputation mismatch')
    result = dict(passed=True,models=15,regional_points=30,rmse_rows=1080,horizon_rows=6480,kci_components=120,
                  max_rmse_abs_diff=max_rmse,max_horizon_abs_diff=max_horizon,max_relative_abs_diff=max_relative,
                  max_kci_percentage_point_abs_diff=max_kci,
                  source_sha256={name:sha(base/name) for name in ('run_protocol.json','selected_models.json',
                    'validation_selection_plan.json','test_metrics.csv','horizon_metrics.csv','kci_components.csv',
                    'regional_coordinates.csv','relative_rmse_components.csv','scores_by_model_seed.csv')},
                  audit_script_sha256=sha(Path(__file__)))
    serialized=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x',encoding='utf-8') as stream:
            stream.write(serialized)
    print(serialized)


if __name__=='__main__':
    main()
