"""Optional real-GPU simulator action gates; no neural training or expert policy.

Run with JAX_PLATFORMS=METAL,cpu in the isolated Metal environment.
Fixture state edits prepare individual mechanics; policy evaluation never uses them.
"""
import os
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "JaxMARL")]
from scenarios import make_env
from scenarios.oc2_sushi_1_3.environment import Action, Item, QUEUED, RECIPE, RICE_BIT
from scenarios.oc2_sushi_1_3.layout import DRYING_CELL, RETURN_CELL


@unittest.skipUnless("metal" in os.environ.get("JAX_PLATFORMS", "").lower(),
                     "explicit Metal environment required")
class MetalSushiActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert jax.default_backend().lower() == "metal", "CPU fallback is not a GPU test"
        cls.cpu, cls.gpu = jax.devices("cpu")[0], jax.devices("METAL")[0]
        with jax.default_device(cls.cpu):
            cls.env = make_env("oc2_sushi_1_3", agent_view_size=2)
            cls.full_env = make_env("oc2_sushi_1_3", agent_view_size=None)
            cls.key = jax.random.PRNGKey(0)
            _, cls.initial = cls.env.reset(cls.key)

        def step(state, action_pair):
            _, after, rewards, dones, info = cls.env.step_env(
                cls.key, state, dict(zip(cls.env.agents, action_pair)))
            assert after.done.dtype == jnp.bool_
            assert all(info[name].dtype == jnp.bool_ for name in
                       ("delivered", "interaction_succeeded", "invalid_action", "blocked"))
            # Widen within the tested program. Casting on the host would conceal
            # whether the backend lost a boolean in an intermediate buffer.
            return jax.tree.map(
                lambda x: x.astype(jnp.int32) if x.dtype == jnp.bool_ else x,
                (after, rewards, dones, info))
        from types import SimpleNamespace
        cls.programs = SimpleNamespace(cpu=jax.jit(step, backend="cpu"),
                                       gpu=jax.jit(step, backend="METAL"),
                                       obs_cpu=jax.jit(cls.full_env.get_obs, backend="cpu"),
                                       obs_gpu=jax.jit(cls.full_env.get_obs, backend="METAL"))

    def check_case(self, state, action=Action.STAY, other_action=Action.STAY):
        with jax.default_device(self.cpu):
            pair = jnp.array([action, other_action], dtype=jnp.int32)
        reference = self.programs.cpu(state, pair)
        actual = self.programs.gpu(jax.device_put(state, self.gpu), jax.device_put(pair, self.gpu))
        self.assertTrue(all(self.gpu in x.devices() for x in jax.tree.leaves(actual)))
        # Positive-state observations must agree too: PPO consumes these after
        # interactions, whereas imitation demonstrations are generated on CPU.
        expected_obs = self.programs.obs_cpu(reference[0])
        actual_obs = self.programs.obs_gpu(actual[0])
        self.assertTrue(all(self.gpu in x.devices() for x in jax.tree.leaves(actual_obs)))
        expected_obs, actual_obs = jax.device_get((expected_obs, actual_obs))
        for agent in self.env.agents:
            self.assertEqual(actual_obs[agent].shape, self.full_env.obs_shape)
            self.assertEqual(actual_obs[agent].dtype, expected_obs[agent].dtype)
            self.assertTrue(np.isfinite(actual_obs[agent]).all())
            np.testing.assert_allclose(actual_obs[agent], expected_obs[agent], rtol=1e-6, atol=1e-7,
                                       err_msg=f"CPU/Metal full-map observation for {agent}")
        self.latest_full_observations = actual_obs
        expected, actual = jax.device_get((reference, actual))
        for i, (a, b) in enumerate(zip(jax.tree.leaves(expected), jax.tree.leaves(actual))):
            self.assertEqual(a.dtype, b.dtype, f"leaf {i} dtype")
            np.testing.assert_array_equal(a, b, err_msg=f"CPU/Metal leaf {i}")
        return actual

    def test_prepared_valid_delivery_preserves_reward_counter_and_event(self):
        with jax.default_device(self.cpu):
            s = self.initial._replace(
                positions=self.initial.positions.at[0].set(jnp.array([1, 6])),
                directions=self.initial.directions.at[0].set(Action.LEFT),
                location=self.initial.location.at[3].set(-1),
                contents=self.initial.contents.at[3].set(RECIPE))
        after, rewards, _, info = self.check_case(s, Action.PICKUP_PLACE)
        self.assertEqual(int(after.kind[3]), Item.DIRTY_PLATE)
        self.assertEqual(int(after.location[3]), QUEUED)
        self.assertEqual(int(after.deliveries), 1)
        self.assertEqual(float(after.score), 20.)
        self.assertEqual(float(rewards["agent_0"]), 20.)
        self.assertEqual(float(rewards["agent_1"]), 20.)
        np.testing.assert_array_equal(info["delivered"], [1, 0])
        np.testing.assert_array_equal(info["interaction_succeeded"], [1, 0])
        np.testing.assert_array_equal(info["invalid_action"], [0, 0])

    def test_lifted_cooked_pot_has_correct_held_observations(self):
        with jax.default_device(self.cpu):
            s = self.initial._replace(
                positions=self.initial.positions.at[0].set(jnp.array([10, 1])),
                directions=self.initial.directions.at[0].set(Action.UP),
                contents=self.initial.contents.at[0].set(RICE_BIT),
                progress=self.initial.progress.at[0].set(self.env.cook_ticks))
        after, rewards, _, info = self.check_case(s, Action.PICKUP_PLACE)
        self.assertEqual(int(after.kind[0]), Item.POT)
        self.assertEqual(int(after.location[0]), -1)
        self.assertEqual(int(after.contents[0]), RICE_BIT)
        self.assertEqual(int(after.progress[0]), self.env.cook_ticks)
        self.assertEqual(float(rewards["agent_0"]), 0.)
        np.testing.assert_array_equal(info["interaction_succeeded"], [1, 0])
        inventory = self.latest_full_observations["agent_0"][..., -12:]
        np.testing.assert_array_equal(inventory[..., int(Item.POT)], 1.)
        np.testing.assert_array_equal(inventory[..., 9], 1.)  # Rice bit in held features.
        np.testing.assert_array_equal(inventory[..., 11], 1.)  # Fully cooked progress.
        np.testing.assert_array_equal(self.latest_full_observations["agent_1"][..., -12:], 0.)

    def test_partial_and_completed_chopping(self):
        for progress, kind in [(0, Item.RAW_CUCUMBER), (self.env.chop_ticks - 1, Item.CHOPPED_CUCUMBER)]:
            with self.subTest(progress=progress), jax.default_device(self.cpu):
                s = self.initial._replace(
                    positions=self.initial.positions.at[0].set(jnp.array([5, 2])),
                    directions=self.initial.directions.at[0].set(Action.UP),
                    kind=self.initial.kind.at[7].set(Item.RAW_CUCUMBER),
                    location=self.initial.location.at[7].set(self.env.cell((5, 1))),
                    progress=self.initial.progress.at[7].set(progress))
                after, rewards, _, info = self.check_case(s, Action.WORK)
                self.assertEqual(int(after.kind[7]), kind)
                self.assertEqual(int(after.progress[7]), progress + 1)
                self.assertEqual(float(rewards["agent_0"]), 0.)
                self.assertEqual(int(after.deliveries), 0)
                np.testing.assert_array_equal(info["interaction_succeeded"], [1, 0])
                np.testing.assert_array_equal(info["invalid_action"], [0, 0])

    def test_partial_and_completed_washing(self):
        for progress in (0, self.env.wash_ticks - 1):
            with self.subTest(progress=progress), jax.default_device(self.cpu):
                s = self.initial._replace(
                    positions=self.initial.positions.at[0].set(jnp.array([10, 6])),
                    directions=self.initial.directions.at[0].set(Action.DOWN),
                    kind=self.initial.kind.at[3].set(Item.DIRTY_PLATE),
                    location=self.initial.location.at[3].set(self.env.cell((10, 7))),
                    progress=self.initial.progress.at[3].set(progress))
                after, rewards, _, info = self.check_case(s, Action.WORK)
                complete = progress + 1 == self.env.wash_ticks
                self.assertEqual(int(after.kind[3]), Item.PLATE if complete else Item.DIRTY_PLATE)
                self.assertEqual(int(after.location[3]), self.env.cell(DRYING_CELL if complete else (10, 7)))
                self.assertEqual(int(after.progress[3]), 0 if complete else progress + 1)
                self.assertEqual(float(rewards["agent_0"]), 0.)
                np.testing.assert_array_equal(info["interaction_succeeded"], [1, 0])
                np.testing.assert_array_equal(info["invalid_action"], [0, 0])

    def test_simultaneous_washing_applies_only_one_tick(self):
        # The sink has two legal approach cells; both empty-handed chefs work.
        for time in (0, 1):
            with self.subTest(priority=time), jax.default_device(self.cpu):
                s = self.initial._replace(
                    positions=jnp.array([[10, 6], [10, 8]], dtype=jnp.int32),
                    directions=jnp.array([Action.DOWN, Action.UP], dtype=jnp.int32),
                    kind=self.initial.kind.at[3].set(Item.DIRTY_PLATE),
                    location=self.initial.location.at[3].set(self.env.cell((10, 7))),
                    time=jnp.int32(time))
                after, rewards, _, info = self.check_case(s, Action.WORK, Action.WORK)
                self.assertEqual(int(after.progress[3]), 1)
                self.assertEqual(int(after.kind[3]), Item.DIRTY_PLATE)
                self.assertEqual(float(rewards["agent_0"]), 0.)
                np.testing.assert_array_equal(info["interaction_succeeded"], [1 - time, time])
                np.testing.assert_array_equal(info["invalid_action"], [time, 1 - time])

    def test_second_chef_delivery_is_counted_in_either_priority_order(self):
        for time in (0, 1):
            with self.subTest(priority=time), jax.default_device(self.cpu):
                s = self.initial._replace(
                    positions=self.initial.positions.at[1].set(jnp.array([1, 6])),
                    directions=self.initial.directions.at[1].set(Action.LEFT),
                    location=self.initial.location.at[4].set(-2),
                    contents=self.initial.contents.at[4].set(RECIPE), time=jnp.int32(time))
                after, rewards, _, info = self.check_case(s, Action.STAY, Action.PICKUP_PLACE)
                self.assertEqual(int(after.kind[4]), Item.DIRTY_PLATE)
                self.assertEqual(int(after.location[4]), QUEUED)
                self.assertEqual(int(after.deliveries), 1)
                self.assertEqual(float(rewards["agent_0"]), 20.)
                np.testing.assert_array_equal(info["delivered"], [0, 1])
                np.testing.assert_array_equal(info["interaction_succeeded"], [0, 1])

    def test_queued_plate_returns_when_due(self):
        with jax.default_device(self.cpu):
            s = self.initial._replace(kind=self.initial.kind.at[3].set(Item.DIRTY_PLATE),
                location=self.initial.location.at[3].set(QUEUED),
                return_due=self.initial.return_due.at[3].set(1))
        after, rewards, _, info = self.check_case(s)
        self.assertEqual(int(after.kind[3]), Item.DIRTY_PLATE)
        self.assertEqual(int(after.location[3]), self.env.cell(RETURN_CELL))
        self.assertEqual(int(after.return_due[3]), -1)
        self.assertEqual(float(rewards["agent_0"]), 0.)
        np.testing.assert_array_equal(info["delivered"], [0, 0])

    def test_rice_cooking_progress_and_completion(self):
        for progress in (0, self.env.cook_ticks - 1):
            with self.subTest(progress=progress), jax.default_device(self.cpu):
                s = self.initial._replace(contents=self.initial.contents.at[0].set(RICE_BIT),
                                          progress=self.initial.progress.at[0].set(progress))
                after, rewards, _, info = self.check_case(s)
                self.assertEqual(int(after.contents[0]), RICE_BIT)
                self.assertEqual(int(after.progress[0]), progress + 1)
                self.assertEqual(int(after.kind[0]), Item.POT)
                self.assertEqual(float(rewards["agent_0"]), 0.)
                np.testing.assert_array_equal(info["delivered"], [0, 0])


if __name__ == "__main__":
    unittest.main()
