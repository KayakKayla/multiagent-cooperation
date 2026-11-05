"""Measure throughput of two independent PPO seeds vectorized on one GPU."""
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
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(JAX_PLATFORMS='cuda,cpu',XLA_FLAGS='--xla_gpu_deterministic_ops=true',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.65')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from train import load_config,load_algorithm
    from benchmarks.variants import load_variant
    config=load_config(ROOT/'configs/coord_ring.yaml',42)
    init,_=load_algorithm('ippo_rnn').make_trainer(config.copy())
    candidate=load_variant({'host_logging':True},args.output/'algorithm_snapshot.py')
    _,updates=candidate.make_trainer(config)
    init=jax.jit(init)
    states=[init(jax.random.split(jax.random.PRNGKey(seed),1)[0]) for seed in (42,43)]
    state=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    def wait(tree):
        for leaf in jax.tree.leaves(tree):leaf.block_until_ready()
    wait(state)
    t=time.perf_counter()
    execute=jax.jit(jax.vmap(lambda s:updates(s,1))).lower(state).compile()
    compile_seconds=time.perf_counter()-t
    durations=[]
    for step in range(1,6):
        t=time.perf_counter();state,metrics=execute(state);wait((state,metrics));duration=time.perf_counter()-t
        durations.append(duration);print(json.dumps(dict(update=step,seconds=duration)),flush=True)
        if step==2:
            first=jax.tree.map(lambda x:x[0],(state,metrics))
            arrays=jax.tree.leaves(jax.device_get(first))
            np.savez(args.output/'probe-update002.npz',**{f'leaf_{i:04}':x for i,x in enumerate(arrays)})
    median=statistics.median(durations[2:])
    result=dict(status='complete',parallel_seeds=[42,43],compile_seconds=compile_seconds,durations_seconds=durations,
                seconds_per_two_seed_update=median,seconds_per_seed_update=median/2,
                aggregate_environment_steps_per_second=2*65536/median,device_memory_stats=jax.devices()[0].memory_stats())
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
