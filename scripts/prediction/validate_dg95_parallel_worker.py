"""Evaluate a disjoint seed subset and atomically publish completed caches."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from scripts.prediction import validate_retrained_dg95 as base


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(43,44),required=True)
    seed=parser.parse_args().seed
    main_out=base.OUT
    worker_out=main_out/'parallel_workers'/f'seed{seed}'
    base.OUT=worker_out
    original_preflight=base.preflight
    original_write_json=base.write_json
    expanded={'dg_ratio0p15','dg_ratio1p5','dg_ratio15','dg_ratio150'}
    def preflight_subset():
        jobs,configs,contract,protocol=original_preflight()
        jobs=[j for j in jobs if j['seed']==seed and j['alias'] in expanded]
        if len(jobs)!=4:
            raise ValueError(f'Wrong worker roster for seed {seed}')
        return jobs,configs,contract,protocol
    def publish(path,value):
        original_write_json(path,value)
        path=Path(path)
        if path.name!='complete.json':
            return
        main_protocol=main_out/'run_protocol.json'
        if not main_protocol.is_file() or base.sha(main_protocol)!=value['run_protocol_sha256']:
            raise ValueError('Main and worker protocol hashes differ')
        folder=path.parent
        relative=folder.relative_to(worker_out/'model_data')
        dest=main_out/'model_data'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.parent/f'.worker_{uuid.uuid4().hex}'
        shutil.copytree(folder,temp)
        try:
            os.rename(temp,dest)
            print(f'Published completed cache: {relative}',flush=True)
        except OSError:
            if not dest.exists():
                raise
            print(f'Main worker already owns cache: {relative}',flush=True)
            if not temp.resolve().is_relative_to((main_out/'model_data').resolve()):
                raise ValueError('Worker temporary path outside model cache')
            shutil.rmtree(temp)
    base.preflight=preflight_subset
    base.write_json=publish
    try:
        base.run()
    except ValueError as exc:
        if str(exc)!='Wrong combined metric coverage':
            raise
        # The parent evaluator alone aggregates all 15 complete models.
        for alias in expanded:
            marker=worker_out/'model_data'/f'seed{seed}'/alias/'complete.json'
            if not marker.is_file():
                raise ValueError(f'Worker incomplete: {marker}')
        result={'seed':seed,'status':'complete','aliases':sorted(expanded)}
        (worker_out/'worker_complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(f'Worker seed {seed} complete',flush=True)


if __name__=='__main__':
    main()
