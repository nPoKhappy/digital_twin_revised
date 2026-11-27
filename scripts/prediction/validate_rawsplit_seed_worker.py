"""Evaluate a seed in isolation and atomically publish complete caches."""
import argparse
import json
import os
import shutil
import uuid

from scripts.prediction import validate_rawsplit_63 as base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(43,44),required=True)
    seed=parser.parse_args().seed
    output=base.OUT
    scratch=output/f'_worker_seed{seed}'
    base.OUT=scratch
    original_preflight=base.preflight
    original_write_json=base.write_json
    def preflight():
        jobs,configs,contract,protocol=original_preflight()
        jobs=[j for j in jobs if j['seed']==seed]
        if len(jobs)!=21: raise ValueError('Expected 21 seed jobs')
        return jobs,configs,contract,protocol
    def publish(path,value):
        original_write_json(path,value)
        if path.name!='complete.json': return
        protocol=output/'run_protocol.json'
        if not protocol.exists() or base.sha(protocol)!=value['run_protocol_sha256']:
            raise ValueError('Worker and primary validation protocol hashes differ')
        source=path.parent
        relative=source.relative_to(scratch/'model_data')
        destination=output/'model_data'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.parent/f'.publish_{uuid.uuid4().hex}'
        shutil.copytree(source,temporary)
        try:
            os.rename(temporary,destination)
            print(f'Published complete cache: {relative}',flush=True)
        except OSError:
            if not destination.exists(): raise
            if not temporary.resolve().is_relative_to((output/'model_data').resolve()):
                raise ValueError('Temporary publication outside primary model data')
            shutil.rmtree(temporary)
            print(f'Primary already owns cache: {relative}',flush=True)
    base.preflight=preflight
    base.write_json=publish
    try:
        base.run()
    except ValueError as exc:
        if str(exc)!='Wrong validation coverage': raise
        jobs,_,_,_=preflight()
        if any(not (scratch/'model_data'/f'seed{seed}'/j['alias']/'complete.json').is_file() for j in jobs):
            raise ValueError('Incomplete seed worker') from exc
        print(f'Completed seed {seed} isolated worker; primary aggregates all 63.',flush=True)


if __name__=='__main__': main()
