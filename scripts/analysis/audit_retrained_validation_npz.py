"""Independently recompute a completed seed's regional validation from NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGIONS = {'internal': ('in_range', ('R5-3','R5-4','R5-5')),
           'external': ('soft_ood', ('R5-27','R5-28','R5-29'))}
TARGETS = ('B35_H2S','B35_SO2')
MVS = ('air2_SP','HEATER2_output_T_SP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(42,43,44))
    args = parser.parse_args()
    base = ROOT/f'results/visualization/retrain_lhs10_23_21_models_validation_seed{args.seed}'
    report = json.loads((base/'evaluation_validation.json').read_text(encoding='utf-8-sig'))
    assert report['passed'] and report['seed'] == args.seed and report['model_count'] == 21
    coords = pd.read_csv(base/'regional_plot_coordinates.csv')
    metrics = pd.read_csv(base/'validation_metrics.csv')
    counts = pd.read_csv(base/'kci_components.csv')
    assert len(coords) == 42 and len(metrics) == 756 and len(counts) == 168
    assert not coords.duplicated(['model','partition']).any()
    max_rmse_error = max_rel_error = 0.0
    totals = {r: [0,0] for r in REGIONS}
    for point in coords.itertuples(index=False):
        model, region = point.model, point.partition
        distribution, cases = REGIONS[region]
        folder = base/'model_data'/model
        assert (folder/'complete.json').is_file()
        with np.load(folder/f'gain_{distribution}.npz') as array:
            teacher, dynamic = array['teacher_gain'], array['dynamic_gain']
            assert dynamic.shape == (2,18,len(teacher),2,2)
            for i,target in enumerate(TARGETS):
                for j,mv in enumerate(MVS):
                    truth = np.broadcast_to(teacher[:,i,j], dynamic[:,:,:,i,j].shape)
                    pred = dynamic[:,:,:,i,j]
                    mask = np.abs(truth) >= 1e-5
                    c = counts[(counts.model==model)&(counts.distribution==distribution)&(counts.target==target)&(counts.mv==mv)]
                    assert len(c)==1
                    c = c.iloc[0]
                    actual = {'possible_comparisons':mask.size, 'valid_comparisons':mask.sum(),
                              'tp':((truth>0)&(pred>0)&mask).sum(), 'tn':((truth<0)&(pred<0)&mask).sum(),
                              'fp':((truth<0)&(pred>0)&mask).sum(), 'fn':((truth>0)&(pred<0)&mask).sum(),
                              'zero_gain':((pred==0)&mask).sum()}
                    for k,v in actual.items():
                        assert int(c[k])==v,(model,region,target,mv,k)
            mask = np.broadcast_to((np.abs(teacher)>=1e-5)[None,None],dynamic.shape)
            valid = int(mask.sum())
            correct = int(((np.sign(teacher)[None,None] == np.sign(dynamic))&mask).sum())
            assert point.valid==valid and point.correct==correct
            np.testing.assert_allclose(point.ANN_reference_KCI_percent,100*correct/valid,rtol=1e-12)
            totals[region][0] += correct; totals[region][1] += valid
        ratios = []
        for case in cases:
            with np.load(folder/f'{case}_prediction.npz') as a, np.load(base/'model_data'/'no_gain'/f'{case}_prediction.npz') as b:
                for j,target in enumerate(TARGETS):
                    for evaluation in ('H1','H18','TS'):
                        if evaluation == 'TS':
                            actual, forecast = a['ts_true'][:,j], a['ts_pred'][:,j]
                            base_actual, base_forecast = b['ts_true'][:,j], b['ts_pred'][:,j]
                        else:
                            h = int(evaluation[1:])-1
                            actual, forecast = a['horizon_true'][:,h,j], a['horizon_pred'][:,h,j]
                            base_actual, base_forecast = b['horizon_true'][:,h,j], b['horizon_pred'][:,h,j]
                        np.testing.assert_allclose(actual,base_actual,rtol=1e-7,atol=1e-12)
                        rmse = np.linalg.norm(actual-forecast)/np.sqrt(len(actual))
                        baseline = np.linalg.norm(base_actual-base_forecast)/np.sqrt(len(base_actual))
                        row = metrics[(metrics.model==model)&(metrics.partition==region)&(metrics.case==case)&(metrics.target==target)&(metrics.evaluation==evaluation)]
                        assert len(row)==1 and int(row.iloc[0].n)==len(actual) and baseline>0
                        max_rmse_error = max(max_rmse_error,abs(float(row.iloc[0].RMSE)-rmse))
                        ratios.append(rmse/baseline)
        assert len(ratios)==18 and min(ratios)>0
        avg = np.exp(np.mean(np.log(ratios)))
        max_rel_error = max(max_rel_error,abs(float(point.AvgRelRMSE)-avg))
    assert max_rmse_error<1e-12 and max_rel_error<1e-9
    print(json.dumps({'seed':args.seed,'passed':True,'models':21,'regional_points':42,'rmse_rows':756,'kci_components':168,
                      'max_rmse_abs_diff':max_rmse_error,'max_avgrelrmse_abs_diff':max_rel_error,
                      'summed_correct_valid_for_audit_only':totals},indent=2))


if __name__=='__main__':
    main()
