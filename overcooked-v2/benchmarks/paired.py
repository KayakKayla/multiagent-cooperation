"""Interleave baseline/candidate calls to control for desktop/GPU timing drift."""
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
    parser.add_argument('--options',required=True)
    parser.add_argument('--samples',type=int,default=6)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    options=json.loads(args.options)
    if options.get('donate'):raise ValueError('Paired timing must retain the shared input state')
    os.environ.update(JAX_PLATFORMS='cuda,cpu',XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import jax
    from train import load_config,load_algorithm
    from benchmarks.variants import load_variant
    config=load_config(ROOT/'configs/coord_ring.yaml',42)
    reference=load_algorithm('ippo_rnn')
    init,_=reference.make_trainer(config.copy())
    functions={};compilation={}
    state=jax.jit(init)(jax.random.split(jax.random.PRNGKey(42),1)[0])
    def wait(tree):
        for x in jax.tree.leaves(tree):x.block_until_ready()
        jax.effects_barrier()
    wait(state)
    for name,settings in [('baseline',{}),('candidate',options)]:
        module=load_variant(settings,args.output/(name+'_snapshot.py'))
        module.wandb.log=lambda _:None
        _,updates=module.make_trainer(config.copy())
        t=time.perf_counter()
        functions[name]=jax.jit(lambda s,f=updates:f(s,1)).lower(state).compile()
        compilation[name]=time.perf_counter()-t
    # Warm both executables, then hold a common real two-update input state.
    for name in functions:
        out=functions[name](state);wait(out)
    state,_=functions['baseline'](state);wait(state)
    state,_=functions['baseline'](state);wait(state)
    rows=[]
    for repeat in range(args.samples):
        order=('baseline','candidate') if repeat%2==0 else ('candidate','baseline')
        for name in order:
            t=time.perf_counter();out=functions[name](state);wait(out)
            row=dict(repeat=repeat,case=name,seconds=time.perf_counter()-t)
            rows.append(row);print(json.dumps(row),flush=True)
            del out
    medians={name:statistics.median(x['seconds'] for x in rows if x['case']==name) for name in functions}
    ratios=[]
    for repeat in range(args.samples):
        times={x['case']:x['seconds'] for x in rows if x['repeat']==repeat}
        ratios.append(times['baseline']/times['candidate'])
    result=dict(options=options,compile_seconds=compilation,samples=rows,medians=medians,
                speedup=medians['baseline']/medians['candidate'],paired_speedups=ratios)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
