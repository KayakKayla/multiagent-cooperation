"""Focused rules and JAX checks; no policy training or Steam connection."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenarios import make_env
from scenarios.oc2_sushi_1_3.environment import Action, Item, Rules, RECIPE
from scenarios.oc2_sushi_1_3.scripted import Script, audit, compare_reference, food_counts


class SushiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env("oc2_sushi_1_3")
        cls.key = jax.random.PRNGKey(0)

    def setUp(self):
        self.obs, self.s = self.env.reset(self.key)

    def step(self, state, a=Action.STAY, b=Action.STAY):
        return self.env.step_env(self.key, state, {"agent_0": a, "agent_1": b})

    def pose(self, state, agent, xy, direction):
        return state._replace(positions=state.positions.at[agent].set(jnp.array(xy)),
                              directions=state.directions.at[agent].set(direction))

    def item(self, state, slot, kind, location, contents=0, progress=0):
        return state._replace(kind=state.kind.at[slot].set(int(kind)),
                              location=state.location.at[slot].set(location),
                              contents=state.contents.at[slot].set(contents),
                              progress=state.progress.at[slot].set(progress))

    def test_script_two_dishes_reuses_washed_plate_and_preserves_food(self):
        script = Script(self.env).run()
        self.assertEqual(int(script.state.deliveries), 2)
        self.assertLessEqual(int(script.state.time), self.env.rules.max_steps)
        np.testing.assert_array_equal(script.spawn_counts, [2, 2, 2])
        np.testing.assert_array_equal(food_counts(script.state), [0, 0, 0])
        self.assertTrue(all(m.get("plate_slot", 3) == 3 for m in script.milestones))
        audit(self.env, script.state)
        self.assertTrue(compare_reference(script)["semantic_checkpoints_passed"])

    def test_measured_solid_obstacles_block_movement_but_lower_passage_remains(self):
        for start, action in (((4, 7), Action.LEFT), ((8, 8), Action.RIGHT)):
            s = self.pose(self.s, 0, start, action)
            _, z, _, _, info = self.step(s, action)
            np.testing.assert_array_equal(z.positions[0], start)
            self.assertTrue(bool(info['blocked'][0]))
        s = self.pose(self.s, 0, (2, 8), Action.RIGHT)
        _, z, _, _, _ = self.step(s, Action.RIGHT)
        np.testing.assert_array_equal(z.positions[0], (3, 8))

    def test_simultaneous_collision_swap_and_following(self):
        for positions in ([[4, 3], [5, 3]], [[3, 3], [5, 3]]):
            s = self.s._replace(positions=jnp.array(positions))
            _, z, _, _, info = self.step(s, Action.RIGHT, Action.LEFT)
            np.testing.assert_array_equal(z.positions, positions)
            self.assertTrue(np.asarray(info["blocked"]).all())
        s = self.s._replace(positions=jnp.array([[4, 8], [5, 8]]))
        _, z, _, _, _ = self.step(s, Action.RIGHT, Action.RIGHT)
        np.testing.assert_array_equal(z.positions, [[5, 8], [6, 8]])

    def test_blocked_movement_turns_and_out_of_range_action_is_noop(self):
        s = self.pose(self.s, 0, (5, 4), 3)  # Shared counter is immediately right.
        _, z, _, _, info = self.step(s, Action.RIGHT, 99)
        np.testing.assert_array_equal(z.positions, s.positions)
        self.assertEqual(int(z.directions[0]), Action.RIGHT)
        self.assertTrue(bool(info["blocked"][0]))
        self.assertTrue(bool(info["invalid_action"][1]))

    def test_simultaneous_pickup_has_only_one_owner(self):
        s = self.item(self.s, 7, Item.RAW_CUCUMBER, self.env.cell((6, 4)))
        s = self.pose(self.pose(s, 0, (5, 4), 0), 1, (7, 4), 2)
        _, z, _, _, info = self.step(s, Action.PICKUP_PLACE, Action.PICKUP_PLACE)
        self.assertEqual(int(z.location[7]), -1)
        self.assertEqual(int(np.asarray(info["interaction_succeeded"]).sum()), 1)
        self.assertEqual(int(np.asarray(info["invalid_action"]).sum()), 1)
        audit(self.env, z)

    def test_chopping_requires_work_empty_hands_and_retains_partial_progress(self):
        s = self.item(self.s, 7, Item.RAW_CUCUMBER, self.env.cell((5, 1)))
        s = self.pose(s, 0, (5, 2), Action.UP)
        _, s, _, _, _ = self.step(s, Action.WORK)
        self.assertEqual(int(s.progress[7]), 1)
        _, s, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(s.location[7]), -1)
        _, z, _, _, info = self.step(s, Action.WORK)
        self.assertTrue(bool(info["invalid_action"][0]))
        self.assertEqual(int(z.progress[7]), 1)
        _, s, _, _, _ = self.step(z, Action.PICKUP_PLACE)
        for _ in range(self.env.chop_ticks - 1):
            _, s, _, _, _ = self.step(s, Action.WORK)
        self.assertEqual(int(s.kind[7]), Item.CHOPPED_CUCUMBER)
        audit(self.env, s)

    def test_invalid_raw_and_duplicate_plate_ingredients_are_preserved(self):
        s = self.pose(self.s, 0, (4, 4), Action.LEFT)
        s = self.item(s, 7, Item.RAW_CUCUMBER, -1)
        _, z, _, _, info = self.step(s, Action.PICKUP_PLACE)
        self.assertTrue(bool(info["invalid_action"][0]))
        np.testing.assert_array_equal(z.contents, s.contents)
        np.testing.assert_array_equal(z.location, s.location)
        s = self.item(z, 7, Item.CHOPPED_CUCUMBER, -1)
        _, s, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(s.contents[3]), 1)
        self.assertEqual(int(s.kind[7]), Item.NONE)
        s = self.item(s, 7, Item.NORI, -1)
        _, s, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(s.contents[3]), 5)
        s = self.item(s, 7, Item.NORI, -1)
        _, z, _, _, info = self.step(s, Action.PICKUP_PLACE)
        self.assertTrue(bool(info["invalid_action"][0]))
        np.testing.assert_array_equal(food_counts(z), food_counts(s))
        self.assertEqual(int(z.location[7]), -1)

    def test_rice_cooks_only_in_pot_on_heat_and_pouring_resets_pot(self):
        partial = self.item(self.s, 0, Item.POT, self.env.cell((6, 4)), 2, 4)
        _, off_heat, _, _, _ = self.step(partial)
        self.assertEqual(int(off_heat.progress[0]), 4)
        partial = self.item(off_heat, 0, Item.POT, -1, 2, 4)
        _, carried, _, _, _ = self.step(partial)
        self.assertEqual(int(carried.progress[0]), 4)
        s = self.item(self.s, 0, Item.POT, self.env.cell((10, 0)), 2, self.env.cook_ticks - 1)
        _, s, _, _, _ = self.step(s)
        self.assertEqual(int(s.progress[0]), self.env.cook_ticks)
        s = self.item(s, 0, Item.POT, -1, 2, self.env.cook_ticks)
        s = self.pose(s, 0, (4, 4), Action.LEFT)
        s = self.item(s, 3, Item.PLATE, self.env.cell((3, 4)), 5)
        before = food_counts(s)
        _, s, _, _, _ = self.step(s)
        self.assertEqual(int(s.progress[0]), self.env.cook_ticks)
        _, z, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(z.contents[3]), RECIPE)
        self.assertEqual(int(z.contents[0]), 0)
        self.assertEqual(int(z.progress[0]), 0)
        self.assertEqual(int(z.location[0]), -1)
        np.testing.assert_array_equal(food_counts(z), before)

    def test_held_plate_collects_food_and_held_rice_enters_pot(self):
        s = self.item(self.s, 3, Item.PLATE, -1)
        s = self.item(s, 7, Item.CHOPPED_CUCUMBER, self.env.cell((6, 4)))
        s = self.pose(s, 0, (5, 4), Action.RIGHT)
        _, z, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(z.contents[3]), 1)
        self.assertEqual(int(z.kind[7]), Item.NONE)
        s = self.item(self.s, 7, Item.RICE, -2)
        s = self.pose(s, 1, (10, 1), Action.UP)
        _, z, _, _, _ = self.step(s, b=Action.PICKUP_PLACE)
        self.assertEqual(int(z.contents[0]), 2)
        self.assertEqual(int(z.progress[0]), 1)
        self.assertEqual(int(z.kind[7]), Item.NONE)
        audit(self.env, z)

    def test_return_queue_waits_for_space_without_losing_plates(self):
        s = self.item(self.s, 3, Item.PLATE, -1, RECIPE)
        s = self.item(s, 4, Item.DIRTY_PLATE, self.env.return_cell)
        s = self.pose(s, 0, (1, 6), Action.LEFT)
        _, s, rewards, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(float(rewards["agent_0"]), self.env.rules.delivery_reward)
        for _ in range(self.env.return_ticks):
            _, s, _, _, _ = self.step(s)
        self.assertEqual(int(s.location[3]), -3)
        s = self.item(s, 4, Item.DIRTY_PLATE, self.env.cell((3, 6)))
        _, z, _, _, _ = self.step(s)
        self.assertEqual(int(z.location[3]), self.env.return_cell)
        audit(self.env, z)

    def test_washing_once_per_tick_and_blocked_drying_area(self):
        s = self.item(self.s, 3, Item.DIRTY_PLATE, self.env.cell((10, 7)))
        s = self.pose(self.pose(s, 0, (10, 6), Action.DOWN), 1, (10, 8), Action.UP)
        _, s, _, _, _ = self.step(s, Action.WORK, Action.WORK)
        self.assertEqual(int(s.progress[3]), 1)
        s = self.item(s, 3, Item.DIRTY_PLATE, self.env.cell((10, 7)), progress=self.env.wash_ticks - 1)
        s = self.item(s, 4, Item.PLATE, self.env.drying_cell)
        _, s, _, _, _ = self.step(s, Action.WORK)
        self.assertEqual(int(s.kind[3]), Item.DIRTY_PLATE)
        self.assertEqual(int(s.progress[3]), self.env.wash_ticks)
        s = self.item(s, 4, Item.PLATE, self.env.cell((3, 6)))
        _, z, _, _, _ = self.step(s, Action.WORK)
        self.assertEqual(int(z.kind[3]), Item.PLATE)
        self.assertEqual(int(z.location[3]), self.env.drying_cell)
        self.assertEqual(int(z.progress[3]), 0)
        audit(self.env, z)

    def test_full_ingredient_pool_rejects_spawn(self):
        env = make_env("oc2_sushi_1_3", rules=Rules(capacity=8))
        _, s = env.reset(self.key)
        s = self.item(s, 7, Item.NORI, env.cell((6, 4)))
        s = self.pose(s, 0, (1, 3), Action.LEFT)
        _, z, _, _, info = env.step_env(self.key, s, {"agent_0": 5, "agent_1": 4})
        self.assertTrue(bool(info["invalid_action"][0]))
        np.testing.assert_array_equal(z.kind, s.kind)
        np.testing.assert_array_equal(z.location, s.location)
        audit(env, z)

    def test_bin_empties_containers_without_destroying_finite_resources(self):
        s = self.pose(self.s, 0, (6, 8), Action.UP)
        s = self.item(s, 0, Item.POT, -1, 2, 4)
        _, z, _, _, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertEqual(int(z.kind[0]), Item.POT)
        self.assertEqual(int(z.contents[0]), 0)
        audit(self.env, z)

    def test_terminal_state_freezes_and_autoreset_preserves_last_reward(self):
        s = self.item(self.s, 3, Item.PLATE, -1, RECIPE)
        s = self.pose(s, 0, (1, 6), Action.LEFT)._replace(time=jnp.int32(self.env.rules.max_steps - 1))
        _, z, _, dones, _ = self.step(s, Action.PICKUP_PLACE)
        self.assertTrue(bool(dones["__all__"]))
        _, frozen, rewards, _, _ = self.step(z, Action.RIGHT)
        for a, b in zip(jax.tree.leaves(z), jax.tree.leaves(frozen)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(float(rewards["agent_0"]), 0)
        _, reset, rewards, dones, _ = self.env.step(self.key, s, {"agent_0": 5, "agent_1": 4})
        self.assertTrue(bool(dones["__all__"]))
        self.assertEqual(int(reset.time), 0)
        self.assertEqual(float(rewards["agent_0"]), self.env.rules.delivery_reward)

    def test_jit_vmap_and_local_observations(self):
        keys = jax.random.split(self.key, 2)
        obs, batch = jax.vmap(self.env.reset)(keys)
        actions = {"agent_0": jnp.array([0, 4]), "agent_1": jnp.array([4, 1])}
        result = jax.jit(jax.vmap(self.env.step_env))(keys, batch, actions)
        self.assertEqual(result[0]["agent_0"].shape, (2, *self.env.obs_shape))
        self.assertTrue(np.isfinite(np.asarray(result[0]["agent_0"])).all())
        local = make_env("oc2_sushi_1_3", agent_view_size=2)
        first = local.get_obs(self.s)["agent_0"]
        second = local.get_obs(self.s._replace(directions=self.s.directions.at[1].set(0)))["agent_0"]
        np.testing.assert_array_equal(first, second)  # Unseen teammate facing must not leak.
        self.assertEqual(first.shape, (5, 5, 48))
        self.assertTrue(np.asarray((first >= 0) & (first <= 1)).all())
        carrying = self.item(self.s, 7, Item.NORI, -2)
        np.testing.assert_array_equal(first, local.get_obs(carrying)["agent_0"])
        near = self.pose(self.s, 1, (5, 3), Action.UP)
        near_carrying = self.item(near, 7, Item.NORI, -2)
        self.assertFalse(np.array_equal(local.get_obs(near)["agent_0"], local.get_obs(near_carrying)["agent_0"]))

    def test_rules_validation_and_legacy_factory_delegation(self):
        for changes in ({"step_seconds": 0}, {"cook_seconds": float("nan")}, {"wash_seconds": -1},
                        {"capacity": 7}, {"max_steps": 0}, {"delivery_reward": float("inf")}):
            with self.assertRaises(ValueError):
                replace(Rules(), **changes)
        with patch.dict(sys.modules, {"jaxmarl": unittest.mock.Mock()}) as modules:
            make_env("coord_ring", agent_view_size=2)
            modules["jaxmarl"].make.assert_called_once_with("overcooked_v2", layout="coord_ring", agent_view_size=2)


if __name__ == "__main__":
    unittest.main()
