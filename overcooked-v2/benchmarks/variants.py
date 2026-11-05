"""Auditable source variants for PPO experiments, never upstream mutations."""
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'algorithms/ippo_rnn.py'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old!r}')
    return source.replace(old, new)


def load_variant(options, snapshot=None):
    source = SOURCE.read_text()
    if options.get('host_logging'):
        source = replace_once(source, 'if not metal_compat:\n            # Metal cannot lower host callbacks;',
                              'if False:\n            # Host logging experiment;')
    if options.get('conditional_reset'):
        source = replace_once(source,
            'obsv, env_state, reward, done, info = jax.vmap(\n                env.step, in_axes=(0, 0, 0)\n            )(rng_step, env_state, env_act)',
            'obsv, env_state, reward, done, info = conditional_log_step(env, rng_step, env_state, env_act)')
        source = 'from benchmarks.kernels import conditional_log_step\n' + source
    if options.get('compact_obs'):
        source = replace_once(source, '                obs_batch,\n                info,',
                              '                obs_batch.astype(jnp.uint8),\n                info,')
        source = replace_once(source, '(traj_batch.obs, traj_batch.done),',
                              '(traj_batch.obs.astype(jnp.float32), traj_batch.done),')
    if options.get('fused_gru'):
        begin, end = source.index('class ScannedRNN(nn.Module):'), source.index('\n\nclass CNN(nn.Module):')
        source = source[:begin] + 'from benchmarks.kernels import FusedScannedRNN as ScannedRNN\n' + source[end:]
        # Explicit name preserves the existing checkpoint's parameter paths.
        source = replace_once(source, 'hidden, embedding = ScannedRNN(\n',
                              'hidden, embedding = ScannedRNN(name="ScannedRNN_0",\n')
    if options.get('gru_unroll'):
        if options.get('fused_gru'):
            source=replace_once(source,'ScannedRNN(name="ScannedRNN_0",',
                                f'ScannedRNN(name="ScannedRNN_0", unroll={int(options["gru_unroll"])},')
        else:
            source = replace_once(source, '        split_rngs={"params": False},',
                                  f'        split_rngs={{"params": False}},\n        unroll={int(options["gru_unroll"])},')
    if options.get('bf16_cnn'):
        begin,end=source.index('class CNN(nn.Module):'),source.index('class ActorCriticRNN(nn.Module):')
        cnn=source[begin:end].replace('nn.Conv(\n','nn.Conv(\n            dtype=jnp.bfloat16,\n')
        cnn=cnn.replace('nn.Dense(\n','nn.Dense(\n            dtype=jnp.bfloat16,\n')
        source=source[:begin]+cnn+source[end:]
        source=replace_once(source,'embedding = jax.vmap(embed_model)(embedding)',
                            'embedding = jax.vmap(embed_model)(embedding).astype(jnp.float32)')
    if options.get('split_phases'):
        start=source.index('    # TRAIN LOOP\n')
        rollout=source.index('        # COLLECT TRAJECTORIES\n',start)
        learning=source.index('        # CALCULATE ADVANTAGE\n',rollout)
        end=source.index('    def init_runner_state(rng):',learning)
        source=(source[:start]+'    def collect(runner_state):\n'+source[rollout:learning]+
                '        return runner_state, traj_batch, initial_hstate\n\n'+
                '    def learn(runner_state, traj_batch, initial_hstate):\n'+source[learning:end]+
                '    def _update_step(runner_state, unused):\n'+
                '        state, trajectory, memory = collect(runner_state)\n'+
                '        return learn(state, trajectory, memory)\n\n'+source[end:])
        source=replace_once(source,'    return init_runner_state, run_updates\n',
                            '    return init_runner_state, run_updates, collect, learn\n')
    if snapshot:
        Path(snapshot).write_text(source)
    module = types.ModuleType('ppo_performance_variant')
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    return module
