"""Measure rollout and learning separately at full baseline shapes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--options',default='{}')
    parser.add_argument('--samples',type=int,default=3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(JAX_PLATFORMS='cuda,cpu',XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import jax
    import numpy as np
    from train import load_config,load_algorithm
    from benchmarks.variants import load_variant
    options={**json.loads(args.options),'host_logging':True,'split_phases':True}
    config=load_config(ROOT/'configs/coord_ring.yaml',42)
    reference=load_algorithm('ippo_rnn');init,_=reference.make_trainer(config.copy())
    candidate=load_variant(options,args.output/'algorithm_snapshot.py')
    _,_,collect,learn=candidate.make_trainer(config)
    def wait(tree):
        for leaf in jax.tree.leaves(tree):leaf.block_until_ready()
        jax.effects_barrier()
    state=jax.jit(init)(jax.random.split(jax.random.PRNGKey(42),1)[0]);wait(state)
    samples=[]
    collect=jax.jit(collect);learn=jax.jit(learn)
    for i in range(args.samples+1):
        t=time.perf_counter();state,trajectory,memory=collect(state);wait((state,trajectory,memory));rollout=time.perf_counter()-t
        t=time.perf_counter();state,metrics=learn(state,trajectory,memory);wait((state,metrics));learning=time.perf_counter()-t
        row=dict(update=i+1,rollout_seconds=rollout,learning_seconds=learning)
        samples.append(row);print(json.dumps(row),flush=True)
        del trajectory,memory
    result=dict(options=options,samples=samples,
        rollout_median_seconds=statistics.median(x['rollout_seconds'] for x in samples[1:]),
        learning_median_seconds=statistics.median(x['learning_seconds'] for x in samples[1:]))
    (args.output/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
