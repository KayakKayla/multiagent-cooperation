"""PPO interface checks; requires the full training dependencies."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenarios import make_env, wrap_training
from scenarios.oc2_sushi_1_3.environment import Action, RECIPE
from scenarios.oc2_sushi_1_3.preview import save_policy_preview


class SushiTrainingTests(unittest.TestCase):
    def test_terminal_reward_logging_and_autoreset(self):
        env = make_env('oc2_sushi_1_3', rules={'max_steps': 2}, agent_view_size=2)
        wrapper = wrap_training(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapper.reset(key)
        self.assertEqual(env.observation_space().shape, (5, 5, 48))
        self.assertEqual(env.action_space(env.agents[0]).n, 7)
        self.assertEqual(env.max_steps, 2)
        stay = {a: Action.STAY for a in env.agents}
        _, state, _, _, _ = wrapper.step(key, state, stay)
        # Finish a prepared dish on the episode's last step.
        raw = state.env_state
        raw = raw._replace(positions=raw.positions.at[0].set(jnp.array([1, 6])),
            directions=raw.directions.at[0].set(Action.LEFT),
            location=raw.location.at[3].set(-1), contents=raw.contents.at[3].set(RECIPE))
        state = state.replace(env_state=raw)
        obs, after, rewards, dones, info = jax.jit(wrapper.step)(key, state,
            {env.agents[0]: Action.PICKUP_PLACE, env.agents[1]: Action.STAY})
        self.assertTrue(bool(dones['__all__']))
        self.assertEqual(int(after.env_state.time), 0)
        np.testing.assert_array_equal(info['returned_episode_lengths'], [2, 2])
        np.testing.assert_array_equal(info['returned_episode_returns'], [20., 20.])
        self.assertEqual(set(info), {'returned_episode_returns', 'returned_episode_lengths',
                                    'returned_episode', 'shaped_reward'})
        for agent in env.agents:
            self.assertEqual(float(rewards[agent]), 20.)
            self.assertEqual(float(info['shaped_reward'][agent]), 0.)
            self.assertTrue(np.isfinite(np.asarray(obs[agent])).all())

    def test_legacy_logging_wrapper_is_unchanged(self):
        from jaxmarl.wrappers.baselines import OvercookedV2LogWrapper
        env = make_env('coord_ring', agent_view_size=2)
        self.assertIsInstance(wrap_training(env), OvercookedV2LogWrapper)

    def test_policy_preview_reuses_renderer_with_real_actions_and_policy_label(self):
        env = make_env('oc2_sushi_1_3', rules={'max_steps': 2})
        key = jax.random.PRNGKey(0)
        _, initial = env.reset(key)
        states = []
        current = initial
        actions = jnp.array([[0, 4], [4, 4]])
        for pair in actions:
            _, current, _, _, _ = env.step_env(key, current, dict(zip(env.agents, pair)))
            states.append(current)
        batched = jax.tree.map(lambda *x: jnp.stack(x), *states)
        trace = (batched, actions, jnp.zeros(2), jnp.zeros(2), jnp.zeros(2), jnp.ones(2))
        with TemporaryDirectory() as output, patch('replay.save_replay') as save:
            save_policy_preview(env, initial, trace, Path(output), 'trained', 1000)
            data, frames, _ = save.call_args.args
            self.assertEqual(len(frames), 3)
            self.assertEqual(data['metadata']['mode'], 'trained')
            self.assertNotIn('script', data['metadata']['perspective'].lower())
            self.assertIsNone(data['steps'][0]['agents'][0]['action'])
            self.assertEqual(data['steps'][1]['agents'][0]['action'], 'right')
            self.assertEqual(data['steps'][-1]['totals']['team_reward'], 0.)
            self.assertTrue((Path(output) / 'rollout.gif').exists())


if __name__ == '__main__':
    unittest.main()
