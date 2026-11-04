"""Refresh the two retained validation comparisons with external R5-27/28/29."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiment_paths import experiment_results_dir
from src import utils
from src.dataset import MultiStepS2SDataset
from src.models import get_model
from src.variable_selection import variable_selection
from scripts.prediction.predict import predict_sliding_window

INTERNAL = ['R5-3', 'R5-4', 'R5-5']
EXTERNAL = ['R5-27', 'R5-28', 'R5-29']
TARGETS = ['B35_H2S', 'B35_SO2']
KEYS = ['partition', 'case', 'target', 'evaluation']
WEIGHTS = [.05, 2.5, 25, 250, 2500]
FOLDERS = {
    'gain': ROOT / 'results/visualization/gain005_gain2p5_gain25_gain250_gain2500_casewise_region50_validation_seed42',
    'ss_only': ROOT / 'results/visualization/ss_only_all_weights_casewise_region50_validation_seed42',
}
STAGE = ROOT / 'results/.validation_external27_28_29_seed42'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def physical(values, columns, mean, std):
    shape = values.shape
    frame = pd.DataFrame(values.reshape(-1, len(columns)), columns=columns)
    frame = utils.inverse_zscore(frame, mean[columns], std[columns])
    return utils.inverse_log_transform(frame, [t for t in TARGETS if t in columns]).to_numpy().reshape(shape)


def evaluate_case(model, frame, config, mean, std, device):
    de, ys, _, en = variable_selection(config['data']['variables_num'])
    indices = [ys.index(t) for t in TARGETS]
    z = utils.apply_zscore(utils.apply_log_transform(frame, TARGETS), mean, std)
    assert en == de + ys and np.isfinite(z[en]).all().all()
    dataset = MultiStepS2SDataset(z, en, de, ys, 18, 18)
    initial = torch.tensor(z[en].iloc[:18].to_numpy(), dtype=torch.float32).unsqueeze(0)
    future = torch.tensor(z[de].iloc[18:].to_numpy(), dtype=torch.float32).unsqueeze(0)
    with contextlib.redirect_stderr(io.StringIO()):
        ts_z = predict_sliding_window(model, initial, future, device, len(ys)).cpu().numpy()[0]
    ts_pred = physical(ts_z, ys, mean, std)[:, indices]
    ts_true = physical(z[ys].iloc[18:].to_numpy(), ys, mean, std)[:, indices]
    predictions, truths = [], []
    with torch.no_grad():
        for history, inputs, targets in torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False):
            history, inputs = history.to(device), inputs.to(device)
            steps = []
            for step in range(18):
                single_input = inputs[:, step:step+1]
                output = model(history, single_input)
                assert output.shape[1] == 1
                steps.append(output)
                history = torch.cat([history[:, 1:], torch.cat([single_input, output], dim=2)], dim=1)
            predictions.append(torch.cat(steps, dim=1).cpu().numpy()[..., indices])
            truths.append(targets.numpy()[..., indices])
    hp = physical(np.concatenate(predictions), TARGETS, mean, std)
    ht = physical(np.concatenate(truths), TARGETS, mean, std)
    raw_horizon_truth = np.stack([frame[TARGETS].iloc[start+18:start+36].to_numpy() for start in range(len(dataset))])
    np.testing.assert_allclose(ht, raw_horizon_truth, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(ts_true, frame[TARGETS].iloc[18:], rtol=1e-12, atol=1e-12)
    arrays = dict(ts_pred=ts_pred, ts_true=ts_true, horizon_pred=hp, horizon_true=ht)
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays


def metric_rows(arrays, alias, case, partition):
    selected, horizons = [], []
    for j, target in enumerate(TARGETS):
        yt, yp = arrays['ts_true'][:, j], arrays['ts_pred'][:, j]
        selected.append(dict(partition=partition, case=case, model=alias, target=target,
                             evaluation='TS', n=len(yt), **utils.calculate_metrics(yt, yp)))
        for h in range(18):
            yt, yp = arrays['horizon_true'][:, h, j], arrays['horizon_pred'][:, h, j]
            row = dict(partition=partition, case=case, model=alias, target=target,
                       horizon=h+1, n=len(yt), **utils.calculate_metrics(yt, yp))
            horizons.append(row)
            if h in [0, 17]:
                selected.append({k: v for k, v in row.items() if k != 'horizon'} | {'evaluation': f'H{h+1}'})
    return selected, horizons


def independent_rmse_rows(arrays, alias, case, partition):
    """Recompute saved-array errors directly, independently of calculate_metrics."""
    rows = []
    for j, target in enumerate(TARGETS):
        for evaluation, truth, prediction in [
            ('TS', arrays['ts_true'][:, j], arrays['ts_pred'][:, j]),
            *[(f'H{h+1}', arrays['horizon_true'][:, h, j], arrays['horizon_pred'][:, h, j]) for h in range(18)],
        ]:
            rows.append(dict(partition=partition, case=case, model=alias, target=target,
                             evaluation=evaluation, n=len(truth),
                             RMSE=float(np.sqrt(np.mean(np.square(prediction-truth))))))
    return rows


def aggregate(frame, keys):
    return frame.groupby(keys, sort=False).agg(
        AvgRelRMSE=('relative_RMSE', lambda s: float(np.exp(np.log(s).mean()))),
        component_count=('relative_RMSE', 'size')).reset_index()


def build_comparison(group, values, horizons, internal_checks, audit, protected, configs):
    original, dest = FOLDERS[group], STAGE / group
    dest.mkdir(exist_ok=True)
    alias_weights = {alias: weight for alias, (kind, weight, _) in configs.items() if kind == group}
    weight_column = 'gain_weight' if group == 'gain' else 'ss_weight'
    selected = values[values.model.isin(alias_weights)].copy()
    selected[weight_column] = selected.model.map(alias_weights)
    baseline = values[values.model == 'no_gain'][KEYS + ['RMSE', 'n']]
    components = selected.merge(baseline.rename(columns={'RMSE': 'NoGain_RMSE', 'n': 'NoGain_n'}), on=KEYS, validate='many_to_one')
    assert len(components) == 180 and components.n.eq(components.NoGain_n).all()
    components['relative_RMSE'] = components.RMSE / components.NoGain_RMSE
    components['overall_log_weight'] = .5 / 18
    regional = aggregate(components, ['model', weight_column, 'partition'])
    overall = regional.pivot(index=['model', weight_column], columns='partition', values='AvgRelRMSE').rename(columns={'internal': 'internal_AvgRelRMSE', 'external': 'external_AvgRelRMSE'}).reset_index()
    overall['AvgRelRMSE'] = np.sqrt(overall.internal_AvgRelRMSE * overall.external_AvgRelRMSE)
    overall['component_count'] = 36
    kci_name = 'kci_pair_counts.csv' if group == 'gain' else 'ss_only_kci_components.csv'
    kci = pd.read_csv(original / kci_name)
    provenance = next(p for p in [original/'kci_provenance.json',
                      original/('plot_validation.json' if group == 'gain' else 'provenance_before_cleanup.json')] if p.exists())
    shutil.copy2(provenance, dest/'kci_provenance.json')
    counts = kci.groupby('model').agg(correct=('correct', 'sum'), valid=('valid_comparisons', 'sum')).reset_index()
    counts['overall_micro_KCI_percent'] = 100 * counts.correct / counts.valid
    overall = overall.merge(counts, on='model', validate='one_to_one').sort_values(['AvgRelRMSE', weight_column])
    overall.insert(0, 'rank', range(1, 6))
    kci_region = kci.groupby(['model', 'distribution']).agg(correct=('correct', 'sum'), valid=('valid_comparisons', 'sum')).reset_index()
    kci_region['partition'] = kci_region.distribution.map({'in_range': 'internal', 'soft_ood': 'external'})
    kci_region['micro_kci_percent'] = 100 * kci_region.correct / kci_region.valid
    regional = regional.merge(kci_region[['model', 'partition', 'micro_kci_percent']], on=['model', 'partition'], validate='one_to_one')
    per_region_evaluation = aggregate(components, ['model', weight_column, 'partition', 'evaluation'])
    per_region_evaluation = per_region_evaluation.merge(kci_region[['model', 'partition', 'micro_kci_percent']], on=['model', 'partition'], validate='many_to_one')
    per_evaluation = per_region_evaluation.pivot(index=['model', weight_column, 'evaluation'], columns='partition', values='AvgRelRMSE').reset_index()
    per_evaluation['AvgRelRMSE'] = np.sqrt(per_evaluation.internal * per_evaluation.external)
    summary = selected.groupby(['partition', 'model', weight_column, 'target', 'evaluation']).agg(cases=('case', 'size'), mean_case_RMSE=('RMSE', 'mean')).reset_index()
    for row in overall.itertuples():
        c = components[components.model == row.model]
        assert len(c) == 36 and all(np.isclose(r.overall_log_weight.sum(), .5) for _, r in c.groupby('partition'))
        np.testing.assert_allclose(row.AvgRelRMSE, np.exp(np.sum(c.overall_log_weight*np.log(c.relative_RMSE))), rtol=1e-12)
    group_values = values[values.model.isin(['no_gain', *alias_weights])]
    group_horizons = horizons[horizons.model.isin(['no_gain', *alias_weights])]
    for frame, filename in [(overall, 'avgrelrmse_overall'), (regional, 'avgrelrmse_by_region'),
                            (per_region_evaluation, 'avgrelrmse_by_region_evaluation'), (per_evaluation, 'avgrelrmse_by_evaluation'),
                            (components, 'relative_rmse_components'), (summary, 'validation_rmse_summary'),
                            (group_values, 'validation_metrics'), (group_horizons, 'horizon_metrics'),
                            (internal_checks[internal_checks.model.isin(['no_gain', *alias_weights])], 'internal_metrics_recheck'),
                            (pd.DataFrame(audit), 'data_audit')]:
        frame.to_csv(dest / f'{filename}.csv', index=False)
    shutil.copy2(original / kci_name, dest / kci_name)
    shutil.copy2(original / 'rebuild_figures.py', dest / 'rebuild_figures.py')
    for alias in ['no_gain', *alias_weights]:
        shutil.copytree(STAGE / 'predictions' / alias, dest / 'predictions' / alias, dirs_exist_ok=True)
    protocol = dict(internal_validation=INTERNAL, external_validation=EXTERNAL, weights=WEIGHTS, seed=42,
                    output_directory=original.relative_to(ROOT).as_posix(),
                    formula='sqrt(GM_internal(same-case RMSE / NoGain RMSE) * GM_external(same-case RMSE / NoGain RMSE))',
                    internal_ratios=18, external_ratios=18, region_weights=[.5, .5],
                    internal_component_log_weight=.5/18, external_component_log_weight=.5/18,
                    regional_RMSE_arithmetic_preaveraging=False, SS_target_scope='all_48_outputs' if group == 'ss_only' else None,
                    preprocessing='10-row median; H2S/SO2 log; each frozen model uses its saved training z-score statistics',
                    H1_H18='Each window starts with 18 true history points; 18-step recursive predictions; H1=10min, H18=180min',
                    TS='18 initial true history points; recursive prediction to the end, with known future decoder inputs',
                    KCI='Reused unchanged steady-state validation KCI: pooled correct/valid over lhs_valid_in_range.xlsx and lhs_valid_soft_ood.xlsx; not calculated from the dynamic R5-27/28/29 trajectories',
                    optimizer_updates=0, script='scripts/prediction/validate_hyperparameters.py')
    write_json(dest / 'metric_protocol.json', protocol)
    report = dict(passed=True, model_count=5, baseline_model_count=1, case_count=6,
                  selected_metric_rows=len(group_values), horizon_metric_rows=len(group_horizons),
                  paired_RMSE_ratios=180, independent_RMSE_rechecks=1368, internal_reference_checks=108,
                  raw_truth_alignment_checked=True, internal_cases_unchanged=True, KCI_reused=True,
                  KCI_inputs_and_checkpoints_hash_verified=True, region50_weighted_log_reconstruction_checked=True,
                  optimizer_updates=0, protected_inputs_unchanged=True, protected_sha256=protected, visual_review='pending')
    write_json(dest / 'evaluation_validation.json', report)
    write_json(dest / 'plot_validation.json', dict(passed=True, model_count=5, cases=INTERNAL+EXTERNAL,
               weighted_log_reconstruction_checked=True, KCI_reused=True, visual_review='pending'))
    lines = [f'# 五個 {"Gain" if group == "gain" else "SS-only"} 超參數 validation 比較', '',
             '權重：0.05、2.5、25、250、2500；seed42。只重跑固定權重推論，未重新訓練。', '',
             '**內插維持 R5-3、R5-4、R5-5；外插改為 R5-27、R5-28、R5-29，全部重新推論。**', '',
             '逐案例 × 目標 × H1/H18/TS 的 RMSE 除以同項 No-Gain RMSE，區內取幾何平均，內外各 50%：`AvgRelRMSE=sqrt(I×E)`。各區 18 項，整體 36 項；No-Gain=1，越低越好。', '',
             '| 權重 | 內部 AvgRelRMSE | 外部 AvgRelRMSE | 整體 AvgRelRMSE | ANN-reference KCI (%) |', '|---:|---:|---:|---:|---:|']
    for row in overall.to_dict('records'):
        lines.append(f'| {row[weight_column]:g} | {row["internal_AvgRelRMSE"]:.9f} | {row["external_AvgRelRMSE"]:.9f} | {row["AvgRelRMSE"]:.9f} | {row["overall_micro_KCI_percent"]:.6f} |')
    lines += ['', '## 資料與圖', '', '- [整體比較圖](overall_kci_vs_avgrelrmse.png)／[PDF](overall_kci_vs_avgrelrmse.pdf)',
              '- [整體 CSV](avgrelrmse_overall.csv)／[區域 CSV](avgrelrmse_by_region.csv)／[H1/H18/TS](avgrelrmse_by_evaluation.csv)',
              '- [180 項逐案例配對比值](relative_rmse_components.csv)／[原單位 RMSE 摘要](validation_rmse_summary.csv)',
              '- [全部 H1/H18/TS 指標，含 No-Gain](validation_metrics.csv)／[全部 H1～H18 指標](horizon_metrics.csv)',
              '- [內部結果與前次比較](internal_metrics_recheck.csv)／[資料範圍](data_audit.csv)／[RMSE 重算](RMSE_recheck.csv)',
              '- [完整預測陣列與 TS CSV](predictions/)／[計算定義](metric_protocol.json)／[核對紀錄](evaluation_validation.json)', '',
              '**KCI 沿用原穩態 validation 探測點的結果，模型、teacher、scaler 與 probe 雜湊均已核對；這次改變的是動態外插驗證案例，並未以 R5-27/28/29 重算 KCI。**', '',
              'R5-5 保留原 internal 分類，部分操作點超出 air2/T2 核心範圍。H1/H18 每次重新給真實歷史，TS 僅最初 18 點使用真實歷史；皆使用已知未來 decoder inputs。', '',
              '## 重跑', '', '僅重畫本資料夾的 CSV：', '', '```powershell',
              f'.\\.venv\\Scripts\\python.exe {str((original/"rebuild_figures.py").relative_to(ROOT)).replace(chr(92), "/")}', '```', '',
              '完整重跑目前兩組比較並覆蓋：', '', '```powershell', '.\\.venv\\Scripts\\python.exe scripts/prediction/validate_hyperparameters.py', '```', '',
              '單一 seed42；相關軌跡不是獨立重複實驗。KCI 為 ANN-reference 方向一致性，不是物理正確性或閉迴路驗證。', '']
    if group == 'ss_only':
        lines += ['其他圖：[區域](regional_kci_vs_avgrelrmse.png)／[H1/H18/TS](horizon_kci_vs_avgrelrmse.png)／[原始 RMSE](validation_rmse_summary.png)。', '']
    (dest / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[{group}]\n{overall.to_string(index=False)}', flush=True)


def main():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = {'no_gain': ('baseline', 0, ROOT / 'configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml')}
    for group, suffix in [('gain', ''), ('ss_only', '_only')]:
        for code, weight in [('005', .05), ('2p5', 2.5), ('25', 25), ('250', 250), ('2500', 2500)]:
            alias = f'gain{code}' if group == 'gain' else f'ss{code}{suffix}'
            configs[alias] = (group, weight, ROOT / f'configs/pgin/{group}/transformer_layerwise_63var_decoder_input_sp_pgin_{alias}_seed42.yaml')
    protected = {}
    def protect(path):
        protected[str(path.relative_to(ROOT))] = sha(path)
    for path in [Path(__file__), ROOT/'scripts/prediction/predict.py', ROOT/'src/dataset.py', ROOT/'src/utils.py', ROOT/'src/variable_selection.py', *(ROOT/'src/models').glob('*.py')]:
        protect(path)
    for folder, report_name in [(FOLDERS['gain'], 'plot_validation.json'), (FOLDERS['ss_only'], 'provenance_before_cleanup.json')]:
        report_path = folder/'kci_provenance.json'
        if not report_path.exists(): report_path = folder/report_name
        report = json.loads(report_path.read_text(encoding='utf-8'))
        for name, digest in report['protected_sha256'].items():
            path = ROOT/name
            if path.suffix == '.pth' or 'zscore_' in path.name or 'lhs_valid_' in path.name:
                assert sha(path) == digest, f'Historical KCI source changed: {name}'
                protect(path)
    old = []
    for group, folder in FOLDERS.items():
        c = pd.read_csv(folder/'relative_rmse_components.csv')
        old.append(c[c.partition == 'internal'][['model']+KEYS+['RMSE']])
        b = c[c.partition == 'internal'][KEYS+['NoGain_RMSE']].drop_duplicates(KEYS).rename(columns={'NoGain_RMSE':'RMSE'})
        b['model']='no_gain'
        if group == 'gain': old.append(b)
        protect(folder/('kci_pair_counts.csv' if group == 'gain' else 'ss_only_kci_components.csv'))
    old = pd.concat(old, ignore_index=True)
    assert len(old) == 198
    data, audits = {}, []
    for case in INTERNAL+EXTERNAL:
        path=ROOT/'data/Claus_dynamic'/f'Test_dataform_change_air2_R=5-{case.split("-")[1]}.{"xlsx" if case in INTERNAL else "csv"}'
        protect(path)
        raw=pd.read_excel(path) if path.suffix=='.xlsx' else pd.read_csv(path)
        frame=raw.rolling(10,min_periods=10).median(numeric_only=True).iloc[9::10].reset_index(drop=True).dropna()
        assert len(frame)>=36 and np.isfinite(frame.to_numpy()).all() and frame[TARGETS].gt(0).all().all()
        data[case]=frame
        inside=raw.air2_SP.between(140,300)&raw.HEATER2_output_T_SP.between(140,240)
        audits.append(dict(case=case, partition='internal' if case in INTERNAL else 'external', source=str(path.relative_to(ROOT)),
                           raw_rows=len(raw),processed_rows=len(frame),TS_points=len(frame)-18,horizon_windows=len(frame)-35,
                           core_percent=float(inside.mean()*100),air2_min=raw.air2_SP.min(),air2_max=raw.air2_SP.max(),
                           T2_min=raw.HEATER2_output_T_SP.min(),T2_max=raw.HEATER2_output_T_SP.max()))
    for alias, (group, weight, path) in configs.items():
        protect(path)
        cfg=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        assert cfg['training']['seed']==42 and cfg['window']['prediction_length']==18
        assert cfg['window']['sampling_interval_min']==10 and cfg['window']['train_window_mins']==180
        assert cfg['training']['gain_loss_weight']==(weight if group=='gain' else 0)
        assert cfg['training']['ss_loss_weight']==(weight if group=='ss_only' else 0)
        for p in [ROOT/'saved_models'/f'{cfg["exp_name"]}.pth', *(experiment_results_dir(cfg)/n for n in ['zscore_mean.csv','zscore_std.csv'])]: protect(p)
    STAGE.mkdir(exist_ok=True)
    protocol=dict(internal=INTERNAL,external=EXTERNAL,weights=WEIGHTS,protected_sha256=protected)
    protocol_path=STAGE/'run_protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text(encoding='utf-8'))==protocol, 'Staging inputs changed.'
    else: write_json(protocol_path,protocol)
    records,horizons=[],[]
    for model_index,(alias,(group,weight,path)) in enumerate(configs.items(),1):
        cfg=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        de,ys,_,en=variable_selection(cfg['data']['variables_num'])
        cfg['data'].update(num_en_input=len(en),num_de_input=len(de),num_output=len(ys))
        mean,std=[pd.read_csv(experiment_results_dir(cfg)/n,index_col=0).squeeze('columns') for n in ['zscore_mean.csv','zscore_std.csv']]
        model=get_model(cfg).to(device)
        model.load_state_dict(torch.load(ROOT/'saved_models'/f'{cfg["exp_name"]}.pth',map_location=device,weights_only=True))
        model.eval()
        for case in INTERNAL+EXTERNAL:
            dest=STAGE/'predictions'/alias/case
            dest.mkdir(parents=True,exist_ok=True)
            npz=dest/'predictions.npz'
            if npz.exists(): arrays=dict(np.load(npz))
            else:
                arrays=evaluate_case(model,data[case],cfg,mean,std,device)
                np.savez_compressed(npz,**arrays)
            selected,full=metric_rows(arrays,alias,case,'internal' if case in INTERNAL else 'external')
            records.extend(selected); horizons.extend(full)
            pd.DataFrame({'time_min':np.arange(18,len(data[case]))*10,
                          **{f'{target}_true':arrays['ts_true'][:,j] for j,target in enumerate(TARGETS)},
                          **{f'{target}_pred':arrays['ts_pred'][:,j] for j,target in enumerate(TARGETS)}}).to_csv(dest/'time_series.csv',index=False)
            print(f'[{model_index}/11] {alias}: {case} complete ({device})',flush=True)
        del model
    values,full=pd.DataFrame(records),pd.DataFrame(horizons)
    assert len(values)==396 and len(full)==2376
    checks=values.merge(old,on=['model']+KEYS,suffixes=('_fresh','_prior'),validate='one_to_one')
    assert len(checks)==198
    np.testing.assert_allclose(checks.RMSE_fresh,checks.RMSE_prior,rtol=1e-5,atol=1e-10)
    print('PASS: 198 internal RMSE checks reproduce the retained original results.',flush=True)
    for group in FOLDERS: build_comparison(group,values,full,checks,audits,protected,configs)
    for group in FOLDERS:
        folder=STAGE/group
        aliases=set(pd.read_csv(folder/'validation_metrics.csv').model)
        rechecks=[]
        for alias in aliases:
            for case in INTERNAL+EXTERNAL:
                arrays=dict(np.load(folder/'predictions'/alias/case/'predictions.npz'))
                rechecks.extend(independent_rmse_rows(arrays,alias,case,'internal' if case in INTERNAL else 'external'))
        assert len(rechecks)==1368
        pd.DataFrame(rechecks).to_csv(folder/'RMSE_recheck.csv',index=False)
        comparison=pd.DataFrame(rechecks).merge(pd.read_csv(folder/'validation_metrics.csv'),on=['model']+KEYS,suffixes=('_recomputed','_csv'),validate='one_to_one')
        assert len(comparison)==216
        np.testing.assert_allclose(comparison.RMSE_recomputed,comparison.RMSE_csv,rtol=1e-12)
        compare_h=pd.DataFrame(rechecks).query("evaluation != 'TS'").copy()
        compare_h['horizon']=compare_h.evaluation.str[1:].astype(int)
        compare_h=compare_h.merge(pd.read_csv(folder/'horizon_metrics.csv'),on=['model','partition','case','target','horizon'],suffixes=('_recomputed','_csv'),validate='one_to_one')
        assert len(compare_h)==1296
        np.testing.assert_allclose(compare_h.RMSE_recomputed,compare_h.RMSE_csv,rtol=1e-12)
        runpy.run_path(str(folder/'rebuild_figures.py'),run_name='__main__')
    assert all(sha(ROOT/name)==digest for name,digest in protected.items()), 'Protected input changed.'
    # Publish only after every numeric check and both figure builds succeed.
    for group,original in FOLDERS.items():
        staged=STAGE/group
        for p in staged.iterdir():
            target=original/p.name
            if p.is_dir(): shutil.copytree(p,target,dirs_exist_ok=True)
            else: shutil.copy2(p,target)
        for name in ['selected_model.json','provenance_before_cleanup.json']:
            p=original/name
            if p.exists(): p.unlink()
    print('PASS: overwrote both retained comparisons; no optimizer updates; 11 models x 6 cases.',flush=True)


if __name__=='__main__':
    main()
