"""Evaluate the updated validation winners on the existing test protocol.

Twelve unchanged runs are copied from independently hash-checked prior test
outputs; only the three newly selected DG runs require inference.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from scripts.evaluation import evaluate_gain_in_out_distribution as native
from scripts.prediction import compare_retrained_mean_tradeoff_test as previous
from scripts.prediction.validate_expanded_dg_incremental import preflight as expanded_preflight
from scripts.prediction.validate_hyperparameters import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/visualization/retrain_lhs10_23_selected_test_nogain42'
OUT = OLD / 'updated_validation_best_staging'
VALID = ROOT / 'results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline'
NEW_MANIFEST = ROOT / 'configs/pgin/retrain_lhs10_23/config_manifest_dg_expanded.json'
PLAN = OUT / 'validation_selection_plan.json'
FAMILIES = ('gain', 'ss_only', 'gain_ss', 'difficult_gradnorm')
EXPECTED = {'gain':'gain25','ss_only':'ss2p5_only','gain_ss':'gain25_ss25','difficult_gradnorm':'dg_ratio1p5'}


def frozen_selection(plan_path: Path) -> list[dict]:
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if plan['aliases'] != EXPECTED:
        raise ValueError('Validation winners do not match the independently checked plan')
    old_rows = json.loads((OLD/'selected_models.json').read_text(encoding='utf8'))['selected_models']
    selected = [dict(row) for row in old_rows if row['family'] != 'difficult_gradnorm']
    jobs = json.loads(NEW_MANIFEST.read_text(encoding='utf8'))
    for seed in (42,43,44):
        matches = [j for j in jobs if j['seed'] == seed and j['alias'] == EXPECTED['difficult_gradnorm']]
        if len(matches) != 1:
            raise ValueError(f'No unique DG winner for seed {seed}')
        job = matches[0]
        selected.append(dict(family='difficult_gradnorm',alias=job['alias'],seed=seed,
            role='validation_selected',validation_mean=None,model=f"{job['alias']}_seed{seed}",
            **{k:job[k] for k in ('config','config_sha256','exp_name','checkpoint','results','cli_args')}))
    if len(selected) != 15 or len({r['model'] for r in selected}) != 15:
        raise ValueError('Expected 15 unique selected models')
    record = dict(selection_rule=plan['selection_rule'],selection_plan=plan,
                  selection_plan_sha256=sha(plan_path),
                  validation_baseline='No-Gain seed42 matching partition/case/target/evaluation',
                  selected_models=selected)
    path = OUT/'selected_models.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != record:
        raise ValueError('Frozen selected models changed')
    if not path.exists(): write_json(path,record)
    return selected


def protected_preflight(selected: list[dict], plan_path: Path):
    # Validate the new training jobs and shared teacher against the prior
    # validation protocol, including seed44's completed restart journal.
    jobs, new_configs, _, expanded_protocol = expanded_preflight()
    old_protocol_path = OLD/'run_protocol.json'
    old_protocol = json.loads(old_protocol_path.read_text(encoding='utf8'))
    if old_protocol['selection_sha256'] != sha(OLD/'selected_models.json'):
        raise ValueError('Previous selection/protocol chain changed')
    old_source_errors = {}
    allowed = {str(Path('results/visualization/retrain_lhs10_23_63_models_validation_nogain42_baseline')/n)
               for n in ('analysis_validation.json','scores_by_family_model.csv','scores_by_model_seed.csv')}
    sources = {}
    for name, digest in old_protocol['sources_sha256'].items():
        path = ROOT/name
        current = sha(path) if path.is_file() else None
        if current != digest:
            if name not in allowed:
                raise ValueError(f'Previous test input changed: {name}')
            old_source_errors[name] = dict(expected_previous_sha256=digest,current_sha256=current)
        else:
            sources[name] = current
    if set(old_source_errors) != allowed:
        raise ValueError('Unexpected previous validation provenance state')
    valid_provenance = json.loads((VALID/'provenance.json').read_text(encoding='utf8'))
    missing = {}
    for name, digest in valid_provenance['sources_sha256'].items():
        path = ROOT/name
        if not path.is_file():
            if 'baseline_previous_before_dg_expansion' not in name:
                raise ValueError(f'Unexpected missing validation input: {name}')
            missing[name] = digest
        elif sha(path) != digest:
            raise ValueError(f'Validation source changed: {name}')
        else:
            sources[name] = digest
    if len(missing) != 5:
        raise ValueError('Unexpected missing historical validation archive count')
    for name,digest in valid_provenance['outputs_sha256'].items():
        if sha(VALID/name) != digest: raise ValueError(f'Validation output changed: {name}')
        sources[str((VALID/name).relative_to(ROOT))] = digest
    for name,digest in expanded_protocol['sources_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError(f'DG training source changed: {name}')
        sources[name] = digest
    for p in (Path(__file__),PLAN,NEW_MANIFEST,old_protocol_path,OLD/'selected_models.json',OUT/'selected_models.json'):
        # Staging files are represented separately so their move to the final
        # output directory does not invalidate historical source references.
        if not p.is_relative_to(OUT): sources[str(p.relative_to(ROOT))] = sha(p)
    configs={}
    old_protocol_sha = sha(old_protocol_path)
    for row in selected:
        cfgpath=ROOT/row['config']
        cfg=yaml.safe_load(cfgpath.read_text(encoding='utf8'))
        if sha(cfgpath)!=row['config_sha256'] or cfg['training']['seed']!=row['seed'] or cfg['exp_name']!=row['exp_name']:
            raise ValueError(f'Config mismatch: {row["model"]}')
        configs[row['model']]=cfg
        if row['family']=='difficult_gradnorm':
            native.assert_same_contract(native.model_contract(cfg),native.model_contract(new_configs[(row['seed'],row['alias'])]),row['model'])
            continue
        d=OLD/'model_data'/row['model']
        marker=json.loads((d/'complete.json').read_text(encoding='utf8'))
        if marker['protocol_sha256'] != old_protocol_sha:
            raise ValueError(f'Previous cached model protocol mismatch: {row["model"]}')
        sources[str((d/'complete.json').relative_to(ROOT))]=sha(d/'complete.json')
        for name,digest in marker['outputs'].items():
            path=d/name
            if not path.is_file() or sha(path)!=digest: raise ValueError(f'Previous cache changed: {path}')
            sources[str(path.relative_to(ROOT))]=digest
    contract=native.model_contract(configs['no_gain_seed42'])
    for name,cfg in configs.items(): native.assert_same_contract(contract,native.model_contract(cfg),name)
    native.assert_same_scaling([(r['config'],configs[r['model']]) for r in selected],contract[0]+contract[1]+contract[2])
    protocol = dict(selection_sha256=sha(OUT/'selected_models.json'),selection_plan_sha256=sha(plan_path),
        sources_sha256=sources,selected_models=selected,internal_test=previous.INTERNAL,external_test=previous.EXTERNAL,
        steady_test={k:str(v.relative_to(ROOT)) for k,v in previous.PROBES.items()},
        preprocessing='10-row median; saved model scaler; 18 historical steps',
        baseline='No-Gain seed42 test RMSE, same partition/case/target/evaluation',
        KCI='ANN-reference micro pooled all18; teacher |gain| >= 1e-5; +/-0.5 dynamic std; zero dynamic gain incorrect',
        interpretation='Previously used test trajectories and probes, not a newly blind holdout; future decoder inputs recorded',
        seed_scope='Shared test data and ANN; dynamic training variation only',
        reused_models=12,newly_evaluated_models=3,
        prior_test_old_validation_hashes_replaced=old_source_errors,
        unavailable_previous_validation_archive_sha256=missing,
        source_limit='Current validation numeric outputs and surviving raw sources checked; deleted historic archive prevents full historical provenance verification.',
        optimizer_updates=0)
    proto=OUT/'run_protocol.json'
    if proto.exists() and json.loads(proto.read_text(encoding='utf8')) != protocol:
        raise ValueError('Sources changed since this test run started')
    if not proto.exists(): write_json(proto,protocol)
    new_protocol_sha=sha(proto)
    for row in selected:
        if row['family']=='difficult_gradnorm': continue
        source=OLD/'model_data'/row['model']; dest=OUT/'model_data'/row['model']
        old_marker=json.loads((source/'complete.json').read_text(encoding='utf8'))
        dest.mkdir(parents=True,exist_ok=True)
        for name,digest in old_marker['outputs'].items():
            target=dest/name
            if target.exists() and sha(target)!=digest: raise ValueError(f'Reused output mismatch: {target}')
            if not target.exists(): shutil.copy2(source/name,target)
            if sha(target)!=digest: raise ValueError(f'Failed to copy: {target}')
        marker=dict(protocol_sha256=new_protocol_sha,outputs=old_marker['outputs'],
                    reused_from=str(source.relative_to(ROOT)),old_complete_sha256=sha(source/'complete.json'),
                    old_protocol_sha256=old_protocol_sha)
        if (dest/'complete.json').exists() and json.loads((dest/'complete.json').read_text(encoding='utf8')) != marker:
            raise ValueError(f'Reuse marker differs: {row["model"]}')
        if not (dest/'complete.json').exists(): write_json(dest/'complete.json',marker)
    return configs,contract,protocol


def main():
    if not PLAN.is_file(): raise FileNotFoundError('Freeze validation selection before opening test inputs')
    previous.OUT=OUT
    previous.selection_before_test=frozen_selection
    previous.preflight=protected_preflight
    previous.run(PLAN)
    report=json.loads((OUT/'evaluation_validation.json').read_text(encoding='utf8'))
    report.update(reused_previous_test_models=12,new_test_models=3,
                  full_historical_provenance_available=False,
                  note='Five old validation archive files deleted; three former validation references changed. Other checked sources intact.')
    write_json(OUT/'evaluation_validation.json',report)


if __name__=='__main__': main()
