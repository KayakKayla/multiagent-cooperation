"""Score retained production policies without rendering or training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True,help='production_run.py result folder')
    parser.add_argument('--episodes',type=int,default=32)
    parser.add_argument('--eval-seed',type=int,default=7000)
    parser.add_argument('--only-final',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(JAX_PLATFORMS='cuda,cpu',XLA_FLAGS='--xla_gpu_enable_command_buffer=FUSION --xla_gpu_deterministic_ops=true',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from train import load_algorithm
    from scenarios import make_env
    from evaluation_core import make_episode
    config=json.loads((args.run/'config.json').read_text())
    module=load_algorithm('ippo_rnn')
    env=make_env(env_name=config['ENV_NAME'],**config['ENV_KWARGS'])
    network=module.ActorCriticRNN(env.action_space(env.agents[0]).n,config)
    episode=make_episode(module,config,env,network)
    def scores(params,seed):
        _,trace=episode(params,seed)
        states,_,reward,correct,wrong,finite=trace
        return reward.sum(),correct.sum(),wrong.sum(),finite.all()&states.done[-1]
    play=jax.jit(jax.vmap(scores,in_axes=(None,0)))
    seeds=jnp.arange(args.eval_seed,args.eval_seed+args.episodes,dtype=jnp.uint32)
    rows=[]
    policies=json.loads((args.run/'policies.json').read_text())
    for policy in policies[-1:] if args.only_final else policies:
        params=jax.device_put(serialization.msgpack_restore((args.run/policy['file']).read_bytes()))
        t=time.perf_counter()
        result=play(params,seeds)
        for leaf in jax.tree.leaves(result):leaf.block_until_ready()
        rewards,correct,wrong,finite=jax.device_get(result)
        if not finite.all():raise ValueError('Invalid frozen evaluation')
        row={**policy,'episodes':args.episodes,'eval_seeds':list(range(args.eval_seed,args.eval_seed+args.episodes)),
             'rewards':rewards.tolist(),'correct_dishes':correct.tolist(),'wrong_dishes':wrong.tolist(),
             'mean_reward':float(rewards.mean()),'mean_dishes':float(correct.mean()),
             'evaluation_seconds':time.perf_counter()-t}
        rows.append(row);(args.output/'evaluations.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
    (args.output/'result.json').write_text(json.dumps(dict(status='complete',evaluations=rows),indent=2))


if __name__=='__main__':main()
