"""Compare real atomic checkpoints on project storage and Linux temporary storage."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(JAX_PLATFORMS='cuda,cpu',XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_deterministic_ops=true',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import jax
    import numpy as np
    import checkpoint as ckpt
    from train import load_config,load_algorithm,ALGORITHMS
    from benchmarks.variants import load_variant
    config=load_config(ROOT/'configs/coord_ring.yaml',42)
    module=load_variant({'host_logging':True})
    init,_=load_algorithm('ippo_rnn').make_trainer(config.copy())
    _,updates=module.make_trainer(config)
    def wait(tree):
        for x in jax.tree.leaves(tree):x.block_until_ready()
    state=jax.jit(init)(jax.random.split(jax.random.PRNGKey(42),1)[0]);wait(state)
    compile_times={}
    for n in (1,2,8,10):
        t=time.perf_counter()
        fn=jax.jit(lambda s:updates(s,n)).lower(state).compile()
        compile_times[str(n)]=time.perf_counter()-t
        if n==2:two=fn
    state,metrics=two(state);wait((state,metrics))
    rows=[]
    with tempfile.TemporaryDirectory(prefix='se390-checkpoint-') as tmp:
        for repeat in range(3):
            for label,root in [('windows_project',args.output/f'windows-{repeat}'),('linux_tmp',Path(tmp)/f'linux-{repeat}')]:
                t=time.perf_counter()
                saved=ckpt.save_checkpoint(root,state,[metrics],2,config,ALGORITHMS['ippo_rnn'])
                duration=time.perf_counter()-t
                t=time.perf_counter();restored,_,_=ckpt.load_checkpoint(saved,state,config,ALGORITHMS['ippo_rnn']);wait(restored)
                restore=time.perf_counter()-t
                for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
                    np.testing.assert_array_equal(a,b)
                rows.append(dict(storage=label,repeat=repeat,save_seconds=duration,restore_seconds=restore,
                                 bytes=sum(p.stat().st_size for p in saved.iterdir())))
    result=dict(status='complete',compile_seconds_by_chunk=compile_times,samples=rows,
                median_save_seconds={label:statistics.median(r['save_seconds'] for r in rows if r['storage']==label)
                                     for label in ('windows_project','linux_tmp')})
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
