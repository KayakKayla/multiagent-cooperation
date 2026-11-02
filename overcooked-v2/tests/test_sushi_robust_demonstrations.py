"""Verify legal start augmentation, masked delays, and real transition replay."""
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'JaxMARL')]
from scenarios.oc2_sushi_1_3.demonstrations import _environment
from scenarios.oc2_sushi_1_3.robust_demonstrations import generate_robust_demo
from scenarios.oc2_sushi_1_3.throughput_schedule import generate_compacted_demo


class RobustDemonstrationTests(unittest.TestCase):
    def test_no_augmentation_preserves_seven_dish_teacher(self):
        actual = generate_robust_demo(42, prefix_steps=0, delay_probability=0.)
        expected = generate_compacted_demo(nori_first=True, prefetch_rice=True)
        self.assertEqual(actual['metadata']['deliveries'], 7)
        self.assertEqual(actual['metadata']['delivery_steps_absolute'], [78, 140, 211, 285, 375, 465, 555])
        for field in ('observations', 'actions', 'supervision_mask'):
            np.testing.assert_array_equal(actual[field], expected[field])
        for a, b in zip(actual['states'], expected['states']):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_prefix_delays_and_all_transitions_independently_replay(self):
        demo = generate_robust_demo(200, prefix_steps=8, delay_probability=.05)
        env = _environment(None)
        key = jax.random.PRNGKey(0)
        _, prefix_state = env.reset(key)
        for actions in demo['metadata']['prefix_actions']:
            _, prefix_state, _, _, _ = env.step_env(key, prefix_state, dict(zip(env.agents, actions)))
        for actual, expected in zip(demo['states'][0], prefix_state):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(int(prefix_state.time), 8)
        _, reset = env.reset(key)
        for field in ('kind', 'location', 'contents', 'progress', 'return_due'):
            np.testing.assert_array_equal(getattr(prefix_state, field), getattr(reset, field))
        mask = demo['supervision_mask']
        self.assertTrue(np.any(mask == 0))
        self.assertTrue(np.any((mask == 0).any(axis=1) & (mask == 1).any(axis=1)))
        self.assertEqual(int(np.count_nonzero(mask == 0)),
                         demo['metadata']['imposed_delay_decisions'] +
                         demo['metadata']['partner_delay_masked_decisions'])
        self.assertEqual(len(demo['actions']), 592)
        self.assertEqual(int(demo['states'][-1].time), 600)
        self.assertTrue(bool(demo['states'][-1].done))

        @jax.jit
        def replay(initial, actions):
            def step(state, pair):
                obs, after, _, _, _ = env.step_env(key, state, dict(zip(env.agents, pair)))
                return after, after
            return jax.lax.scan(step, initial, actions)[1]
        replayed = jax.device_get(replay(prefix_state, jnp.asarray(demo['actions'])))
        recorded = jax.tree.map(lambda *parts: np.stack(parts), *demo['states'][1:])
        for a, b in zip(replayed, recorded):
            np.testing.assert_array_equal(a, b)
        before = jax.tree.map(lambda *parts: np.stack(parts), *demo['states'][:-1])
        obs = jax.jit(jax.vmap(env.get_obs))(before)
        expected_obs = np.stack([np.asarray(obs[a]) for a in env.agents], axis=1)
        np.testing.assert_array_equal(expected_obs, demo['observations'])
        self.assertGreaterEqual(demo['metadata']['deliveries'], 6)

    def test_partner_delay_cannot_change_a_supervised_label_at_identical_history(self):
        canonical = generate_robust_demo(0, prefix_steps=0, delay_probability=0.)
        delayed = generate_robust_demo(207, prefix_steps=0, delay_probability=.05)
        # At the first rice handoff, chef 1 has the exact same entire input history.
        np.testing.assert_array_equal(canonical['observations'][:9, 1], delayed['observations'][:9, 1])
        self.assertEqual(int(canonical['actions'][8, 1]), 5)
        self.assertEqual(int(delayed['actions'][8, 1]), 4)
        self.assertEqual(float(delayed['supervision_mask'][8, 1]), 0.)
        self.assertGreater(delayed['metadata']['partner_delay_masked_decisions'], 0)
        self.assertEqual(delayed['metadata']['supervision_version'], 2)

    def test_declared_spawn_changes_only_poses_and_keeps_full_horizon(self):
        demo = generate_robust_demo(300, prefix_steps=0, delay_probability=.05,
                                   initial_positions=((5, 3), (11, 4)), initial_directions=(0, 1))
        env = _environment(None)
        _, reset = env.reset(jax.random.PRNGKey(0))
        initial = demo['states'][0]
        np.testing.assert_array_equal(initial.positions, ((5, 3), (11, 4)))
        np.testing.assert_array_equal(initial.directions, (0, 1))
        for field in reset._fields:
            if field not in ('positions', 'directions'):
                np.testing.assert_array_equal(getattr(reset, field), getattr(initial, field))
        self.assertEqual(len(demo['actions']), 600)
        self.assertEqual(int(demo['states'][-1].time), 600)
        self.assertTrue(demo['metadata']['declared_spawn_variation'])
        self.assertEqual(demo['metadata']['initial_state_origin'], 'declared valid empty-kitchen spawn variation')
        self.assertEqual(demo['metadata']['prefix_actions'], [])
        self.assertGreaterEqual(demo['metadata']['deliveries'], 6)

    def test_invalid_spawn_poses_are_rejected(self):
        cases = [dict(initial_positions=((5, 3), (11, 3))),
                 dict(initial_directions=(0, 1)),
                 dict(initial_positions=((5, 3), (11, 3)), initial_directions=(0, 1), prefix_steps=8),
                 dict(initial_positions=((6, 3), (11, 3)), initial_directions=(0, 1)),
                 dict(initial_positions=((5, 3), (5, 3)), initial_directions=(0, 1)),
                 dict(initial_positions=((-1, 3), (11, 3)), initial_directions=(0, 1)),
                 dict(initial_positions=((5., 3.), (11., 3.)), initial_directions=(0, 1)),
                 dict(initial_positions=((5, 3), (11, 3)), initial_directions=(4, 1)),
                 dict(initial_positions=((5, 3), (11, 3)), initial_directions=(0,))]
        for case in cases:
            args = dict(prefix_steps=0, **{k: v for k, v in case.items() if k != 'prefix_steps'})
            args['prefix_steps'] = case.get('prefix_steps', 0)
            with self.subTest(case=case), self.assertRaises(ValueError):
                generate_robust_demo(0, **args)

    def test_argument_bounds(self):
        for kwargs in ({'seed': -1}, {'seed': True}, {'seed': 0, 'prefix_steps': -1},
                       {'seed': 0, 'prefix_steps': 33}, {'seed': 0, 'prefix_steps': 1.5},
                       {'seed': 0, 'delay_probability': float('nan')},
                       {'seed': 0, 'delay_probability': 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_robust_demo(**kwargs)


if __name__ == '__main__':
    unittest.main()
