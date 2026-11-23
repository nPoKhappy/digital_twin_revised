"""Rebind staged test to a layout-only validation provenance revision.

No test prediction is recomputed. Rebinding is allowed only when all 20
validation candidates, selected aliases, and every directly used validation
table remain identical. Model markers are updated after their outputs are
independently verified; a subsequent raw-NPZ audit is required.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.prediction import compare_dg95_selected_test as compare
from scripts.prediction.validate_hyperparameters import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = compare.OUT


def main():
    old_plan = compare.load(OUT/'validation_selection_plan.json')
    old_selection = compare.load(OUT/'selected_models.json')
    old_protocol = compare.load(OUT/'run_protocol.json')
    old_protocol_sha = sha(OUT/'run_protocol.json')
    if (old_protocol['selection_sha256'] != sha(OUT/'selected_models.json') or
            old_protocol['selection_plan_sha256'] != sha(OUT/'validation_selection_plan.json')):
        raise ValueError('Old staged selection chain is inconsistent')
    new_plan, _ = compare.select_validation()
    old_core = {k:v for k,v in old_plan.items() if k != 'validation_sources_sha256'}
    new_core = {k:v for k,v in new_plan.items() if k != 'validation_sources_sha256'}
    if old_core != new_core:
        raise ValueError('Validation metrics or selection changed; do not rebind inference')
    old_sources = old_plan['validation_sources_sha256']
    new_sources = new_plan['validation_sources_sha256']
    if set(old_sources) != set(new_sources):
        raise ValueError('Validation source set changed')
    changed = {k:(old_sources[k],new_sources[k]) for k in old_sources if old_sources[k] != new_sources[k]}
    expected = str((compare.VALID/'provenance.json').relative_to(ROOT))
    if set(changed) != {expected}:
        raise ValueError(f'Expected only a validation provenance revision; got {list(changed)}')
    for relative, expected_hash in old_protocol['sources_sha256'].items():
        if relative in changed:
            if expected_hash != old_sources[relative]:
                raise ValueError('Previous validation provenance hash differs')
        elif sha(ROOT/relative) != expected_hash:
            raise ValueError(f'Non-layout source changed: {relative}')
    markers = []
    for row in old_selection['selected_models']:
        path=OUT/'model_data'/row['model']/'complete.json'; marker=compare.load(path)
        if marker['protocol_sha256'] != old_protocol_sha:
            raise ValueError(f'Marker protocol differs: {row["model"]}')
        for name,digest in marker['outputs'].items():
            if sha(path.parent/name)!=digest:
                raise ValueError(f'Model prediction changed: {row["model"]}/{name}')
        markers.append((path,marker))
    write_json(OUT/'validation_selection_plan.json',new_plan)
    old_selection['selection_plan']=new_plan
    old_selection['selection_plan_sha256']=sha(OUT/'validation_selection_plan.json')
    write_json(OUT/'selected_models.json',old_selection)
    old_protocol['selection_plan_sha256']=sha(OUT/'validation_selection_plan.json')
    old_protocol['selection_sha256']=sha(OUT/'selected_models.json')
    old_protocol['sources_sha256'][expected]=new_sources[expected]
    old_protocol['validation_layout_rebind']={
        'reason':'Validation figure layout revision; validation numeric tables and all 20 candidate scores unchanged.',
        'old_validation_provenance_sha256':old_sources[expected],
        'new_validation_provenance_sha256':new_sources[expected],
        'old_test_protocol_sha256':old_protocol_sha,
        'script_sha256':sha(Path(__file__).resolve()),
    }
    write_json(OUT/'run_protocol.json',old_protocol)
    for path,marker in markers:
        marker['protocol_sha256']=sha(OUT/'run_protocol.json')
        write_json(path,marker)
    print(f'Rebound {len(markers)} test model markers after layout-only validation revision')


if __name__=='__main__': main()
