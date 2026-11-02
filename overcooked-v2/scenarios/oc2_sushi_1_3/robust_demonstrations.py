"""Training-only concurrent demonstrations with varied starts and action delays.

Starts are either reached by legal prefixes or declared empty-kitchen spawn
variations. Food/resources/timers are unchanged. Neural evaluation uses no planner.
"""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scenarios import make_env
from .demonstrations import _DelayedScript, _environment, save_demo
from .layout import Tile
from .throughput_expert import ThroughputExpert
from .throughput_schedule import CompactedSchedule


@lru_cache(maxsize=1)
def _reference_environment():
    return make_env('oc2_sushi_1_3', agent_view_size=None, rules={'max_steps': 1200})


def generate_robust_demo(seed, prefix_steps=8, delay_probability=.05, *,
                         initial_positions=None, initial_directions=None):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Use an unsigned 32-bit seed')
    if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int) or not 0 <= prefix_steps <= 32:
        raise ValueError('Prefix steps must be an integer from zero through 32')
    if not np.isfinite(delay_probability) or not 0 <= delay_probability <= 1:
        raise ValueError('Delay probability must be finite and between zero and one')
    spawn_variation = initial_positions is not None or initial_directions is not None
    if spawn_variation and (initial_positions is None or initial_directions is None or prefix_steps != 0):
        raise ValueError('Spawn variation requires both poses and prefix_steps=0')
    with jax.default_device(jax.devices('cpu')[0]):
        env = _environment(None)
        prefix = _DelayedScript(env, seed, 0, 0).prefix(prefix_steps)
        initial = prefix.state
        if spawn_variation:
            positions, directions = np.asarray(initial_positions), np.asarray(initial_directions)
            if (positions.shape != (2, 2) or directions.shape != (2,)
                    or not np.issubdtype(positions.dtype, np.integer)
                    or not np.issubdtype(directions.dtype, np.integer)):
                raise ValueError('Spawn positions/directions must be integer arrays of shapes (2,2)/(2,)')
            inside = ((positions >= 0) & (positions < [env.width, env.height])).all()
            if (not inside or np.array_equal(positions[0], positions[1])
                    or np.any((directions < 0) | (directions > 3))
                    or any(int(env.tiles[y, x]) != Tile.FLOOR for x, y in positions)):
                raise ValueError('Spawn poses must have distinct floor cells and directions0..3')
            reset = initial
            initial = initial._replace(positions=jnp.asarray(positions, dtype=jnp.int32),
                                       directions=jnp.asarray(directions, dtype=jnp.int32))
            for field in reset._fields:
                if field not in ('positions', 'directions'):
                    np.testing.assert_array_equal(getattr(reset, field), getattr(initial, field))
        # Start the teacher from the actually reached state, keeping elapsed time.
        reference = ThroughputExpert(_reference_environment(), routing='short',
                                     nori_first=True, prefetch_rice=True)
        reference.state = initial
        reference.states = [initial]
        reference.labels = ['After legal unsupervised movement prefix']
        reference.run()
        plan = CompactedSchedule(env, reference, initial_state=initial,
                                 delay_probability=delay_probability, delay_seed=seed).run()
        if plan.deadlock:
            raise RuntimeError('Training teacher deadlocked: ' + json.dumps(plan.deadlock))
        before = jax.tree.map(lambda *parts: np.stack(parts), *plan.states[:-1])
        obs = jax.jit(jax.vmap(env.get_obs))(before)
        observations = np.stack([np.asarray(obs[a]) for a in env.agents], axis=1).astype(np.float32)
        states = tuple(jax.device_get(plan.states))
    actions = np.asarray(plan.actions, dtype=np.int32)
    mask = np.asarray(plan.supervision, dtype=np.float32)
    assert len(actions) == env.max_steps - prefix_steps and len(states) == len(actions) + 1
    assert int(states[0].time) == prefix_steps and int(states[-1].time) == env.max_steps
    forced = np.asarray(plan.imposed_delays, dtype=bool)
    assert np.all(actions[forced] == 4) and np.all(mask[forced] == 0), 'Imposed delays execute masked STAY'
    metadata = {
        'schema_version': 1, 'layout': env.name, 'seed': int(seed),
        'teacher_training_only': True, 'expert_used_at_neural_evaluation': False,
        'seed_scope': ('declared spawn poses fixed by caller; seed varies per-chef delays' if spawn_variation
                       else 'legal movement prefix and independent per-chef action delays'),
        'prefix_steps': prefix_steps, 'prefix_actions': prefix.prefix_actions,
        'initial_state_origin': ('declared valid empty-kitchen spawn variation' if spawn_variation
                                 else 'after recorded legal-prefix actions; context discarded'),
        'declared_spawn_variation': spawn_variation,
        'initial_state_time': int(initial.time),
        'initial_positions': np.asarray(initial.positions).tolist(),
        'initial_directions': np.asarray(initial.directions).tolist(),
        'delay_probability': float(delay_probability),
        'masked_agent_decisions': int(np.count_nonzero(mask == 0)),
        'imposed_delay_decisions': int(forced.sum()),
        'partner_delay_masked_decisions': plan.partner_delay_masked_decisions,
        'supervision_version': 2,
        'supervision': 'own delays and same-step partner-delay-dependent labels masked; state-based waits supervised',
        'steps': len(actions), 'deliveries': int(states[-1].deliveries),
        'delivery_steps_absolute': plan.delivery_steps, 'rules': asdict(env.rules),
        'reference_episode_steps': 1200, 'reference_routing': 'short',
        'nori_first': True, 'prefetch_rice': True,
        'parallel_action_ticks': plan.parallel_ticks, 'joint_stay_ticks': plan.wait_ticks,
        'alignment': 'observations[t]=get_obs(states[t]); actions[t] produces states[t+1]',
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('robust_demonstrations.py', 'throughput_schedule.py',
                                       'throughput_expert.py', 'demonstrations.py',
                                       'scripted.py', 'environment.py', 'layout.py')},
    }
    return dict(observations=observations, actions=actions, supervision_mask=mask,
                lengths=np.asarray(len(actions), dtype=np.int32), states=states, metadata=metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prefix-steps', type=int, default=8)
    parser.add_argument('--delay-probability', type=float, default=.05)
    parser.add_argument('--positions', type=int, nargs=4, help='Two declared spawn(x,y) pairs; use prefix-steps0')
    parser.add_argument('--directions', type=int, nargs=2, help='Declared spawn directions0..3')
    args = parser.parse_args()
    positions = np.asarray(args.positions).reshape(2, 2) if args.positions is not None else None
    demo = generate_robust_demo(args.seed, args.prefix_steps, args.delay_probability,
                                initial_positions=positions, initial_directions=args.directions)
    save_demo(demo, args.output)
    print(json.dumps(demo['metadata'], indent=2))


if __name__ == '__main__':
    main()
