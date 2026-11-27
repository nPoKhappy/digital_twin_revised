"""Independently recheck every rawsplit validation cache and provenance hash."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prediction.validate_gain_ss_hyperparameters import numpy_counts
from scripts.prediction.validate_hyperparameters import sha, write_json, INTERNAL, EXTERNAL

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline_rawsplit'
TARGETS=['B35_H2S','B35_SO2']
MVS=['air2_SP','HEATER2_output_T_SP']


def main():
    proto=OUT/'run_protocol.json'; protocol=json.loads(proto.read_text(encoding='utf8'))
    check=json.loads((OUT/'evaluation_validation.json').read_text(encoding='utf8'))
    if not check.get('passed') or len(protocol['jobs'])!=63: raise ValueError('Incomplete input')
    teacher={}; rmse_n=kci_n=0
    for job in protocol['jobs']:
        folder=OUT/'model_data'/f"seed{job['seed']}"/job['alias']
        marker=json.loads((folder/'complete.json').read_text(encoding='utf8'))
        if marker['run_protocol_sha256']!=sha(proto): raise ValueError(f'Protocol mismatch: {folder}')
        for name,digest in marker['outputs'].items():
            if sha(folder/name)!=digest: raise ValueError(f'Cache hash mismatch: {folder/name}')
        metrics=pd.read_csv(folder/'validation_metrics.csv')
        for case in (*INTERNAL,*EXTERNAL):
            with np.load(folder/f'{case}_prediction.npz') as data:
                for index,target in enumerate(TARGETS):
                    tests=[('TS',data['ts_true'][:,index],data['ts_pred'][:,index]),
                           ('H1',data['horizon_true'][:,0,index],data['horizon_pred'][:,0,index]),
                           ('H18',data['horizon_true'][:,17,index],data['horizon_pred'][:,17,index])]
                    for evaluation,truth,prediction in tests:
                        row=metrics[(metrics.case==case)&(metrics.target==target)&(metrics.evaluation==evaluation)]
                        if len(row)!=1 or len(truth)!=int(row.n.iloc[0]): raise ValueError('RMSE coverage')
                        value=float(np.sqrt(np.mean((truth-prediction)**2)))
                        np.testing.assert_allclose(value,row.RMSE.iloc[0],rtol=1e-12,atol=1e-15);rmse_n+=1
        recorded=pd.read_csv(folder/'kci_components.csv')
        for dist in ('in_range','soft_ood'):
            with np.load(folder/f'gain_{dist}.npz') as data:
                ann=data['teacher_gain']; dynamic=data['dynamic_gain']
                if not np.isfinite(ann).all() or not np.isfinite(dynamic).all(): raise ValueError('Nonfinite gain')
                if dist not in teacher: teacher[dist]=ann.copy()
                else: np.testing.assert_allclose(ann,teacher[dist],rtol=1e-5,atol=1e-10)
                recounted=numpy_counts(ann,dynamic,1e-5,TARGETS,MVS)
                for row in recounted.itertuples(index=False):
                    expected=recorded[(recorded.distribution==dist)&(recorded.target==row.target)&(recorded.mv==row.mv)]
                    if len(expected)!=1: raise ValueError('KCI row coverage')
                    for col in ('possible_comparisons','valid_comparisons','tp','tn','fp','fn','zero_gain'):
                        if int(getattr(row,col))!=int(expected[col].iloc[0]): raise ValueError(f'KCI mismatch: {folder} {dist} {col}')
                    kci_n+=1
    if (rmse_n,kci_n)!=(2268,504): raise ValueError('Audit counts wrong')
    for path,digest in protocol['sources_sha256'].items():
        if sha(ROOT/path)!=digest: raise ValueError(f'Source changed: {path}')
    write_json(OUT/'independent_npz_audit.json',{'passed':True,'model_count':63,'rmse_checks':rmse_n,'kci_pair_checks':kci_n,'teacher_arrays_consistent':True,'marker_and_source_hashes_checked':True})
    print(f'PASS: {rmse_n} RMSE and {kci_n} KCI pair recounts',flush=True)


if __name__=='__main__': main()
