"""Compare the existing ANN teacher with recorded in-range step responses."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scripts.evaluation.evaluate_gain_in_out_distribution import (
    load_teacher, teacher_gain_matrix, TAB_INPUT_COLS, FULL_TARGET_COLS,
)

source_dir = ROOT / 'data/Claus_dynamic/step_change/in_training_distribution'
output_dir = ROOT / 'results/ann_teacher_step_validation_in_range_20261013'
output_dir.mkdir(parents=True, exist_ok=True)
device = torch.device('cpu')
teacher, mean, std, target_mean, target_std = load_teacher(device)
dynamic_std = pd.read_csv(ROOT / 'results/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42/zscore_std.csv', index_col=0).squeeze('columns')


def predict(inputs):
    z = (inputs[TAB_INPUT_COLS] - mean[TAB_INPUT_COLS]) / std[TAB_INPUT_COLS]
    with torch.no_grad():
        return (teacher(torch.tensor(z.to_numpy(), dtype=torch.float32).unsqueeze(0)) * target_std + target_mean).numpy()[0]


details, quality, windows, transient, plot_records, provenance = [], [], [], [], [], []
for path in sorted(source_dir.glob('*.csv')):
    frame = pd.read_csv(path)
    required = TAB_INPUT_COLS + FULL_TARGET_COLS
    assert np.isfinite(frame[required].to_numpy()).all(), path.name
    mv = 'air2_SP' if '_air2_change_' in path.name else 'HEATER2_output_T_SP'
    step_rows = np.flatnonzero(np.abs(np.diff(frame[mv].to_numpy())) > 0.5) + 1
    assert len(step_rows) == 1, (path.name, step_rows)
    step_row = int(step_rows[0])
    assert step_row >= 200 and len(frame) - step_row >= 200
    before = frame.iloc[step_row - 100:step_row].mean(numeric_only=True)
    after = frame.iloc[-100:].mean(numeric_only=True)
    delta = after[mv] - before[mv]
    base_inputs = pd.DataFrame([before[TAB_INPUT_COLS]])
    local = teacher_gain_matrix(base_inputs, teacher, mean, std, target_mean, target_std, FULL_TARGET_COLS, [mv], device).numpy()[0, :, 0]
    y0 = predict(before)
    fixed_after = before.copy()
    fixed_after[mv] = after[mv]
    y_fixed = predict(fixed_after)
    y_actual_inputs = predict(after)
    other_cols = [c for c in TAB_INPUT_COLS if c != mv]
    input_delta_z = (after[other_cols] - before[other_cols]) / std[other_cols]
    max_other_delta_z = float(input_delta_z.abs().max())
    qc = {'case': path.name, 'rows': len(frame), 'step_row_zero_based': step_row, 'mv': mv, 'delta_mv': delta, 'delta_mv_dynamic_sigma': delta / dynamic_std[mv], 'max_other_input_tail_mean_delta_teacher_z': max_other_delta_z}
    for col in TAB_INPUT_COLS:
        qc[col + '_before_mean'] = before[col]
        qc[col + '_after_mean'] = after[col]
        qc[col + '_tail_delta'] = after[col] - before[col]
        qc[col + '_post_max_deviation'] = float((frame[col].iloc[step_row:] - before[col]).abs().max())
    quality.append(qc)
    for target_index, target in enumerate(FULL_TARGET_COLS):
        effect = after[target] - before[target]
        observed_gain = effect / delta
        ann_local_gain = float(local[target_index])
        ann_fixed_secant = float((y_fixed[target_index] - y0[target_index]) / delta)
        ann_path_secant = float((y_actual_inputs[target_index] - y0[target_index]) / delta)
        pre_drift = frame[target].iloc[step_row-50:step_row].mean() - frame[target].iloc[step_row-100:step_row-50].mean()
        post_drift = frame[target].iloc[-50:].mean() - frame[target].iloc[-100:-50].mean()
        pre_sd = frame[target].iloc[step_row-100:step_row].std()
        row = {
            'case': path.name, 'mv': mv, 'target': target, 'delta_mv': delta,
            'observed_before': before[target], 'observed_after': after[target],
            'observed_change': effect, 'observed_finite_step_gain': observed_gain,
            'ann_local_gain_at_before': ann_local_gain,
            'ann_fixed_other_inputs_secant_gain': ann_fixed_secant,
            'ann_actual_input_path_secant_gain': ann_path_secant,
            'observed_gain_sign': int(np.sign(observed_gain)),
            'ann_local_sign': int(np.sign(ann_local_gain)),
            'local_sign_agreement': bool(observed_gain * ann_local_gain > 0),
            'fixed_secant_sign_agreement': bool(observed_gain * ann_fixed_secant > 0),
            'teacher_valid_at_1e_minus5': bool(abs(ann_local_gain) >= 1e-5),
            'local_gain_abs_relative_error_percent': 100 * abs(ann_local_gain - observed_gain) / abs(observed_gain),
            'fixed_secant_abs_relative_error_percent': 100 * abs(ann_fixed_secant - observed_gain) / abs(observed_gain),
            'ann_pred_before': float(y0[target_index]), 'ann_pred_after_actual_inputs': float(y_actual_inputs[target_index]),
            'before_value_error_percent': 100 * (y0[target_index] - before[target]) / before[target],
            'after_value_error_percent': 100 * (y_actual_inputs[target_index] - after[target]) / after[target],
            'pre_window_half_drift_over_effect_percent': 100 * abs(pre_drift / effect),
            'post_window_half_drift_over_effect_percent': 100 * abs(post_drift / effect),
            'effect_over_pre_window_std': abs(effect) / pre_sd if pre_sd > 0 else np.nan,
            'max_other_input_tail_delta_teacher_z': max_other_delta_z,
        }
        details.append(row)
        for size in (50, 100, 200):
            b = frame.iloc[step_row-size:step_row].mean(numeric_only=True)
            a = frame.iloc[-size:].mean(numeric_only=True)
            g = (a[target] - b[target]) / (a[mv] - b[mv])
            windows.append({'case':path.name,'mv':mv,'target':target,'window_rows':size,'observed_gain':g,'same_observed_sign_as_100_rows':bool(g*observed_gain>0),'same_ann_baseline_local_sign':bool(g*ann_local_gain>0)})
        # A descriptive transient check: 18 complete blocks of 10 original rows.
        # No claim that these 18 blocks are statistically independent cases.
        for block in range(18):
            section = frame.iloc[step_row+10*block:step_row+10*(block+1)]
            g = (section[target].median() - before[target]) / (section[mv].median() - before[mv])
            discernible = abs(section[target].median() - before[target]) > 5*pre_sd
            transient.append({'case':path.name,'mv':mv,'target':target,'post_step_block':block+1,'original_rows_per_block':10,'observed_block_gain':g,'same_ann_local_sign':bool(g*ann_local_gain>0),'effect_exceeds_5_pre_std':bool(discernible)})
        plot_records.append((path.name,mv,target,frame[target].to_numpy(),step_row,before[target],after[target],float(y0[target_index]),float(y_actual_inputs[target_index])))
    provenance.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(frame)})

detail = pd.DataFrame(details)
qc = pd.DataFrame(quality)
window = pd.DataFrame(windows)
blocks = pd.DataFrame(transient)
summary_rows = []
for (mv,target), group in detail.groupby(['mv','target']):
    valid = group[group.teacher_valid_at_1e_minus5]
    summary_rows.append({'mv':mv,'target':target,'case_count':len(group),'observed_positive_cases':int((group.observed_gain_sign>0).sum()),'observed_negative_cases':int((group.observed_gain_sign<0).sum()),'local_sign_correct':int(group.local_sign_agreement.sum()),'fixed_secant_sign_correct':int(group.fixed_secant_sign_agreement.sum()),'training_threshold_valid_cases':len(valid),'training_threshold_sign_correct':int(valid.local_sign_agreement.sum()),'median_local_gain_abs_relative_error_percent':group.local_gain_abs_relative_error_percent.median(),'max_local_gain_abs_relative_error_percent':group.local_gain_abs_relative_error_percent.max(),'median_fixed_secant_abs_relative_error_percent':group.fixed_secant_abs_relative_error_percent.median()})
summary = pd.DataFrame(summary_rows)
for data,name in [(detail,'gain_comparison_detail.csv'),(summary,'gain_comparison_summary.csv'),(qc,'case_input_quality.csv'),(window,'window_sensitivity.csv'),(blocks,'first18_blocks_direction.csv')]:
    data.to_csv(output_dir/name,index=False,lineterminator='\n')

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9})
fig, axes = plt.subplots(2,2,figsize=(10,8))
for ax, ((mv,target), group) in zip(axes.flat, detail.groupby(['mv','target'])):
    x = group.observed_finite_step_gain.to_numpy()*1e5
    y = group.ann_local_gain_at_before.to_numpy()*1e5
    ax.scatter(x,y,color='#0072B2',label='ANN local derivative',zorder=3)
    ax.scatter(x,group.ann_fixed_other_inputs_secant_gain.to_numpy()*1e5,color='#D55E00',marker='x',label='ANN finite-step slope',zorder=3)
    lo=min(x.min(),y.min(),0)*1.15; hi=max(x.max(),y.max(),0)*1.15
    ax.plot([lo,hi],[lo,hi],color='gray',linestyle='--',label='Equal magnitude')
    ax.axhline(0,color='gray',linewidth=.5);ax.axvline(0,color='gray',linewidth=.5)
    ax.set_title(('air2' if mv=='air2_SP' else 'T2')+' -> '+target.removeprefix('B35_')+' | 8/8 local signs agree')
    ax.set_xlabel('Recorded finite-step slope (x 1e5)');ax.set_ylabel('ANN slope (x 1e5)')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('ANN teacher vs recorded in-range step cases\nFinite-step response supports direction; not an exact partial-derivative reference')
fig.tight_layout(rect=(0,0,1,.93))
fig.savefig(output_dir/'gain_comparison.png',dpi=180);plt.close(fig)

# All observed trajectories are retained for checking baseline and tail behavior.
fig, axes = plt.subplots(8,4,figsize=(19,23))
for index, record in enumerate(plot_records):
    name,mv,target,values,step_row,b,a,ann_b,ann_a=record
    ax=axes.flat[index]
    ax.plot(np.arange(len(values))-step_row,values,color='#333333',linewidth=1,label='Recorded output')
    ax.axvline(0,color='gray',linestyle=':')
    ax.axhline(b,color='#0072B2',linestyle='--',linewidth=.8,label='Pre-step mean')
    ax.axhline(a,color='#009E73',linestyle='--',linewidth=.8,label='Tail mean')
    ax.axhline(ann_b,color='#CC79A7',linestyle=':',linewidth=.8,label='ANN before')
    ax.axhline(ann_a,color='#D55E00',linestyle=':',linewidth=.8,label='ANN after actual inputs')
    ax.set_title(name.removesuffix('.csv')+'\n'+target,fontsize=8)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3));ax.grid(alpha=.15)
    ax.set_xlabel('Original row index relative to step',fontsize=7)
axes.flat[0].legend(fontsize=6)
fig.suptitle('All 16 recorded cases: H2S and SO2 trajectories',fontsize=15)
fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output_dir/'recorded_responses_all_cases.png',dpi=130);plt.close(fig)

artifacts=['saved_models/Tabular_MLP_5in_2out_QV.pth','results/Tabular_MLP_New/zscore_mean.csv','results/Tabular_MLP_New/zscore_std.csv']
manifest={'method':{'before':'mean of final 100 original rows before detected step','after':'mean of final 100 original rows in file','reference':'recorded finite-step slope, not exact local derivative','ann':'same physical-output ANN checkpoint and scaler as gain training; local derivative at before mean','training_teacher_threshold':1e-5,'window_sensitivity_rows':[50,100,200],'transient':'18 complete post-step blocks, median of 10 original rows each; descriptive only'},'sources':provenance,'ann_artifacts':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in artifacts],'limitations':['Only 8 base operating conditions, 16 trajectories and 32 case/output comparisons.','Finite perturbations and incidental changes of other inputs do not identify exact partial derivatives.','Training-set membership of the actual checkpoint is not certified by a training manifest. Current training source code reads LHS Excel files rather than these step CSVs.','No Soft-OOD or new 0.5-sigma simulation validation is performed.','Raw row indices are used; physical time interval is not independently verified.']}
(output_dir/'method_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(detail)==32 and len(qc)==16 and len(window)==96 and len(blocks)==576
assert summary.case_count.sum()==len(detail)
print(summary.to_string(index=False))
print('WINDOW signs unchanged:',int(window.same_observed_sign_as_100_rows.sum()),'/',len(window))
print('TRANSIENT signs matching:',int(blocks.same_ann_local_sign.sum()),'/',len(blocks),'discernible:',int(blocks.effect_exceeds_5_pre_std.sum()),'discernible matching:',int((blocks.same_ann_local_sign & blocks.effect_exceeds_5_pre_std).sum()))
print('Maximum post-window half drift / total observed effect (%):',detail.post_window_half_drift_over_effect_percent.max())
print('Maximum other-input tail mean delta (teacher z):',qc.max_other_input_tail_mean_delta_teacher_z.max())
print('Outputs:',output_dir)
