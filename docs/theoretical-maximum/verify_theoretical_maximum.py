"""Replay the saved 400-step schedule in the actual pinned JaxMARL simulator.

Run with the project's simulator dependencies and JaxMARL on PYTHONPATH.
No policy training or randomized-start evaluation is performed.
"""
import ast
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actions', type=Path, default=HERE / 'theoretical-maximum-actions.json')
    parser.add_argument('--expected-soups', type=int, default=18)
    parser.add_argument('--gif', action='store_true', help='Save an interactive HTML replay, GIF and images.')
    parser.add_argument('--output', type=Path, default=HERE / 'media' / '18-soups')
    args = parser.parse_args()
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=Path(jaxmarl.__file__).resolve().parent.parent,
        text=True).strip()
    assert revision == '976aeb152cb184a5095021968bba94da96eb6394', revision
    actions = json.loads(args.actions.read_text())
    if args.actions.resolve() == (HERE / 'theoretical-maximum-actions.json').resolve():
        with (HERE / 'theoretical-maximum-schedule.csv').open() as f:
            expected = list(csv.DictReader(f))
    else:
        # Independently compare new search candidates to the small audit model.
        from theoretical_maximum import replay
        _, rows = replay(actions)
        fields = ['step', 'chef0_action', 'chef1_action', 'chef0_position', 'chef1_position',
                  'chef0_facing', 'chef1_facing', 'chef0_inventory', 'chef1_inventory',
                  'north_pot', 'east_pot', 'central_counter', 'deliveries', 'events']
        expected = [dict(zip(fields, row)) for row in rows]
    assert len(actions) == len(expected) == 400
    env = jaxmarl.make('overcooked_v2', layout='coord_ring', max_steps=400,
                      random_reset=False, random_agent_positions=False,
                      start_cooking_interaction=False, negative_rewards=True,
                      sample_recipe_on_delivery=False, agent_view_size=2)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    initial = state
    recorded = []
    step_rewards = []
    np.testing.assert_array_equal(state.agents.pos.x, [2, 1])
    np.testing.assert_array_equal(state.agents.pos.y, [1, 2])
    np.testing.assert_array_equal(state.agents.dir, [0, 0])
    step = jax.jit(env.step_env)
    inventory_codes = [0, 4, 1, 15]
    reward = 0
    delivery_steps = []
    for t, (pair, row) in enumerate(zip(actions, expected), 1):
        key, subkey = jax.random.split(key)
        _, state, rewards, done, _ = step(subkey, state,
                                         {f'agent_{a}': jnp.array(pair[a]) for a in range(2)})
        positions = [(int(state.agents.pos.x[a]), int(state.agents.pos.y[a])) for a in range(2)]
        assert positions[0] != positions[1], (t, positions)
        for a in range(2):
            assert positions[a] == ast.literal_eval(row[f'chef{a}_position']), t
            assert int(state.agents.dir[a]) == int(row[f'chef{a}_facing']), t
            assert int(state.agents.inventory[a]) == inventory_codes[int(row[f'chef{a}_inventory'])], t
        for col, x, y in [('north_pot', 3, 0), ('east_pot', 4, 1)]:
            code = int(row[col])
            ingredients = (code-10)*4 if code < 13 else 14 if code == 13 else 12
            timer = code-13 if code >= 14 else 0
            assert int(state.grid[y, x, 1]) == ingredients, (t, col)
            assert int(state.grid[y, x, 2]) == timer, (t, col)
        assert int(state.grid[2, 2, 1]) == inventory_codes[int(row['central_counter'])], t
        r = float(rewards['agent_0'])
        assert r == float(rewards['agent_1'])
        assert r in (0, 20)
        reward += r
        if args.gif:
            recorded.append(jax.device_get(state))
            step_rewards.append(r)
        if r:
            delivery_steps.append(t)
        assert reward == int(row['deliveries'])*20, t
        assert bool(done['__all__']) == (t == 400), t
    result = {'backend': jax.default_backend(), 'steps': 400,
              'deliveries': len(delivery_steps), 'reward': reward,
              'delivery_steps': delivery_steps, 'overlaps': 0,
              'upstream_revision': '976aeb152cb184a5095021968bba94da96eb6394'}
    assert reward == 20 * args.expected_soups and len(delivery_steps) == args.expected_soups
    if args.gif:
        # Reuse the existing spectator dashboard, with explicit scripted labels.
        sys.path.insert(0, str(HERE.parent.parent / 'overcooked-v2'))
        from preview import save_preview
        states = jax.tree.map(lambda *values: np.stack(values), *recorded)
        raw_rewards = np.asarray(step_rewards)
        trace = (states, np.asarray(actions), raw_rewards, raw_rewards / 20,
                 np.zeros(400), np.ones(400, dtype=bool))
        args.output.mkdir(parents=True, exist_ok=True)
        save_preview(initial, trace, args.output,
                     {'ENV_KWARGS': {'layout': 'coord_ring', 'agent_view_size': 2}},
                     'scripted', 0)
        (args.output / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(f"Replay (open in a browser): {args.output / 'replay.html'}")
        print(f"GIF saved to {args.output / 'rollout.gif'}")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
