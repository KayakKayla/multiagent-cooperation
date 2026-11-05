"""Bounded full-shape PPO benchmarks and paired learning comparisons.

Run with the project's Linux Python: python benchmarks/ppo.py --help.
Every case runs in a fresh process; no production configuration is changed.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import shutil
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--options', default='{}', help='JSON source-variant options')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minibatches', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--stop-after', type=int, help='Learning run; keep original 152-update schedules')
    parser.add_argument('--resume',type=Path,help='Resume a benchmark learning checkpoint into a new output folder')
    parser.add_argument('--eval-every', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--eval-seed', type=int, default=7000)
    parser.add_argument('--cache-dir')
    parser.add_argument('--command-buffer', default='')
    parser.add_argument('--preallocate', choices=('false', 'true'), default='false')
    parser.add_argument('--memory-fraction', default='0.60')
    parser.add_argument('--precision')
    parser.add_argument('--backend',choices=('gpu','cpu'),default='gpu')
    parser.add_argument('--no-deterministic', action='store_true')
    parser.add_argument('--extra-xla', default='')
    args = parser.parse_args()
    if args.samples<1 or args.warmup<0 or args.chunk<1 or args.episodes<1:
        parser.error('Use positive sample/chunk/episode counts and nonnegative warmup')
    args.output.mkdir(parents=True, exist_ok=False)
    options = json.loads(args.options)
    flags = '' if args.command_buffer=='default' else f'--xla_gpu_enable_command_buffer={args.command_buffer}'
    if not args.no_deterministic:
        flags += ' --xla_gpu_deterministic_ops=true'
    os.environ.update(JAX_PLATFORMS='cuda,cpu' if args.backend=='gpu' else 'cpu', XLA_FLAGS=(flags+' '+args.extra_xla).strip(),
        XLA_PYTHON_CLIENT_PREALLOCATE=args.preallocate,
        XLA_PYTHON_CLIENT_MEM_FRACTION=args.memory_fraction, WANDB_MODE='disabled', MPLBACKEND='Agg')
    if args.cache_dir:
        os.environ['JAX_COMPILATION_CACHE_DIR'] = args.cache_dir
        os.environ['JAX_EXPLAIN_CACHE_MISSES'] = 'true'
    if args.precision:
        os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = args.precision
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from train import load_config, load_algorithm, git_state
    from benchmarks.variants import load_variant, SOURCE

    if jax.default_backend() != args.backend:
        raise RuntimeError(f'This experiment requires {args.backend}; no backend fallback')
    config = load_config(ROOT/'configs/coord_ring.yaml', args.seed)
    for name, value in [('NUM_MINIBATCHES', args.minibatches), ('NUM_ENVS', args.num_envs),
                        ('NUM_STEPS', args.num_steps), ('UPDATE_EPOCHS', args.epochs)]:
        if value is not None:
            config[name] = value
    if config['ENV_KWARGS']['layout'] != 'coord_ring':
        raise ValueError('These experimental variants are restricted to Coordination Ring')
    config['PERFORMANCE_EXPERIMENT']=options
    import checkpoint as ckpt
    config['PERFORMANCE_EXPERIMENT_SOURCES']={name:ckpt.file_hash(ROOT/'benchmarks'/name)
                                           for name in ('kernels.py','variants.py','ppo.py')}
    max_updates=config['TOTAL_TIMESTEPS']//config['NUM_ENVS']//config['NUM_STEPS']
    if args.stop_after is not None and not 1<=args.stop_after<=max_updates:
        raise ValueError('Stop point must lie within the original training budget')
    module = load_variant(options, args.output/'algorithm_snapshot.py')
    reference = load_algorithm('ippo_rnn')
    # All variants start from the original network/optimizer and exact RNG state.
    init, _ = reference.make_trainer(config.copy())
    _, updates = module.make_trainer(config)
    # Save actual derived batch sizes after make_trainer updates the config.
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    progress = (args.output/'progress.jsonl').open('w')
    def log(metrics):
        from flax.traverse_util import flatten_dict
        row = {k: float(np.asarray(v)) for k,v in flatten_dict(metrics, sep='/').items()}
        progress.write(json.dumps(row)+'\n'); progress.flush()
    module.wandb.log = log
    def wait(tree):
        for leaf in jax.tree.leaves(tree):
            if isinstance(leaf, jax.Array):
                leaf.block_until_ready()
        jax.effects_barrier()
    def save_probe(state, metrics, name):
        arrays = jax.tree.leaves(jax.device_get((state, metrics)))
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError('Non-finite runner state or metrics')
        np.savez(args.output/name, **{f'leaf_{i:04}': x for i,x in enumerate(arrays)})
        paths,_=jax.tree_util.tree_flatten_with_path((state,metrics))
        (args.output/'probe-paths.json').write_text(json.dumps({f'leaf_{i:04}':jax.tree_util.keystr(path)
             for i,(path,_) in enumerate(paths)},indent=2))
    metadata = dict(options=options, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        jax=jax.__version__, devices=[str(x) for x in jax.devices()], xla_flags=os.environ['XLA_FLAGS'],
        upstream_revision=(ROOT/'UPSTREAM_REVISION').read_text().strip(),backend=jax.default_backend(),
        source_run_id='run:'+str(uuid4()),started_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), **git_state())
    for filename in ('kernels.py','variants.py','ppo.py'):
        source=ROOT/'benchmarks'/filename
        shutil.copyfile(source,args.output/('benchmark_'+filename))
        metadata[filename+'_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    started = time.perf_counter()
    key = jax.random.split(jax.random.PRNGKey(args.seed), 1)[0]
    initial = jax.jit(init)(key); wait(initial)
    init_seconds = time.perf_counter()-started
    t = time.perf_counter()
    fn = jax.jit(lambda state: updates(state, args.chunk), donate_argnums=(0,) if options.get('donate') else ())
    executable = fn.lower(initial).compile()
    compile_seconds = time.perf_counter()-t
    memory_analysis = executable.memory_analysis()
    print(json.dumps(dict(event='compiled', init_seconds=init_seconds, compile_seconds=compile_seconds,
                          memory_analysis=str(memory_analysis))), flush=True)
    state = initial
    completed=0
    metric_history=[]
    if args.resume:
        import checkpoint as ckpt
        state,previous_metrics,completed=ckpt.load_checkpoint(args.resume,initial,config,args.output/'algorithm_snapshot.py')
        metric_history.append(previous_metrics)
        print(json.dumps(dict(event='restored',update=completed,checkpoint=str(args.resume))),flush=True)
    durations = []
    eval_rows = []
    evaluations = None
    def evaluate(state, step):
        nonlocal evaluations
        if evaluations is None:
            from scenarios import make_env
            from evaluation_core import make_episode
            env = make_env(env_name=config['ENV_NAME'], **config['ENV_KWARGS'])
            network = reference.ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
            episode = make_episode(reference, config, env, network)
            def scores(params, seed):
                _, trace = episode(params, seed)
                _, _, rewards, correct, wrong, finite = trace
                return rewards.sum(), correct.sum(), wrong.sum(), finite.all()
            evaluations = jax.jit(jax.vmap(scores, in_axes=(None,0)))
        t = time.perf_counter()
        reward, correct, wrong, finite = jax.device_get(evaluations(state[0].params,
            jnp.arange(args.eval_seed, args.eval_seed+args.episodes, dtype=jnp.uint32)))
        if not finite.all():
            raise ValueError('Non-finite evaluation predictions')
        weights = serialization.to_bytes(jax.device_get(state[0].params))
        (args.output/f'params-{step:03}.msgpack').write_bytes(weights)
        temporary=args.output/'params.msgpack.tmp';temporary.write_bytes(weights)
        os.replace(temporary,args.output/'params.msgpack')
        row = dict(update=step, environment_steps=step*config['NUM_ENVS']*config['NUM_STEPS'],
            training_seconds=sum(durations), compilation_seconds=compile_seconds+init_seconds,
            evaluation_seconds=time.perf_counter()-t, seeds=list(range(args.eval_seed,args.eval_seed+args.episodes)),
            rewards=reward.tolist(), correct_dishes=correct.tolist(), wrong_dishes=wrong.tolist(),
            mean_reward=float(reward.mean()), mean_dishes=float(correct.mean()),
            sha256=hashlib.sha256(weights).hexdigest())
        eval_rows.append(row)
        (args.output/'evaluations.json').write_text(json.dumps(eval_rows, indent=2))
        import checkpoint as ckpt
        saved=ckpt.save_checkpoint(args.output,state,metric_history,step,config,args.output/'algorithm_snapshot.py',keep_last=3)
        (args.output/'checkpoint.json').write_text((saved/'checkpoint.json').read_text())
        print(json.dumps(dict(event='evaluation', **row)), flush=True)
    if args.stop_after and (args.stop_after-completed)%args.chunk:
        raise ValueError('Stop point must align with chunk length')
    total_calls = (args.stop_after-completed)//args.chunk if args.stop_after else args.warmup+args.samples
    for i in range(total_calls):
        t = time.perf_counter()
        state, metrics = executable(state); wait((state,metrics))
        duration = time.perf_counter()-t
        if options.get('host_logging'):
            for row in range(args.chunk):
                log(jax.tree.map(lambda x:x[row],jax.device_get(metrics)))
        durations.append(duration)
        step = completed+(i+1)*args.chunk
        if args.stop_after:
            metric_history.append(jax.device_get(metrics))
        print(json.dumps(dict(event='update', update=step, seconds=duration,
            training_return=float(np.asarray(metrics['returned_episode_returns'][-1])))), flush=True)
        if step in (1,2):
            save_probe(state, metrics, f'probe-update{step:03}.npz')
        if args.stop_after and (step % args.eval_every == 0 or step == args.stop_after):
            evaluate(state, step)
    if args.stop_after:
        save_probe(state, metrics, 'probe-final.npz')
    if args.stop_after:
        measured = durations
    else:
        measured = durations[args.warmup:]
    result = dict(status='complete', init_seconds=init_seconds, compile_seconds=compile_seconds,
        durations_seconds=durations, warmed_median_seconds_per_update=statistics.median(measured)/args.chunk,
        warmed_mean_seconds_per_update=statistics.mean(measured)/args.chunk,
        environment_steps_per_second=config['NUM_ENVS']*config['NUM_STEPS']*args.chunk/statistics.median(measured),
        memory_analysis=str(memory_analysis), wall_seconds=time.perf_counter()-started,
        device_memory_stats=jax.devices()[0].memory_stats() or {}, evaluations=eval_rows)
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    progress.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
