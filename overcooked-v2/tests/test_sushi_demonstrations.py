"""Demonstrations preserve action/state alignment and the audited kitchen cycle."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenarios import make_env
from scenarios.oc2_sushi_1_3.demonstrations import generate_demo, save_demo


class SushiDemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = generate_demo(0, random_wait_max=0)
        cls.augmented = generate_demo(7, random_wait_max=2)

    def test_two_deliveries_and_complete_washed_plate_cycles(self):
        demo = self.augmented
        self.assertEqual(demo["metadata"]["deliveries"], 2)
        self.assertTrue(demo["metadata"]["semantic_reference_passed"])
        names = {entry["name"] for entry in demo["metadata"]["milestones"]}
        for cycle in (1, 2):
            self.assertIn(f"Dish {cycle}: served", names)
            self.assertIn(f"Dish {cycle}: washed plate", names)
            self.assertIn(f"Dish {cycle}: plate ready for reuse", names)
        self.assertEqual(demo["observations"].shape, (len(demo["actions"]), 2, 10, 14, 48))
        self.assertEqual(demo["observations"].dtype, np.float32)
        self.assertEqual(demo["actions"].dtype, np.int32)
        self.assertEqual(demo["supervision_mask"].shape, demo["actions"].shape)
        self.assertEqual(int(np.count_nonzero(demo["supervision_mask"] == 0)),
                         2 * demo["metadata"]["inserted_wait_steps"])
        self.assertEqual(int(demo["lengths"]), len(demo["actions"]))
        self.assertTrue(np.all(self.plain["supervision_mask"] == 1))
        self.assertEqual(len(demo["states"]), len(demo["actions"]) + 1)
        self.assertTrue(np.isfinite(demo["observations"]).all())

    def test_every_recorded_action_produces_the_next_recorded_state(self):
        demo = self.augmented
        env = make_env("oc2_sushi_1_3")
        key = jax.random.PRNGKey(0)
        for t, actions in enumerate(demo["actions"]):
            _, after, _, _, _ = env.step_env(key, demo["states"][t], dict(zip(env.agents, actions)))
            for actual, expected in zip(after, demo["states"][t + 1]):
                np.testing.assert_array_equal(actual, expected, err_msg=f"transition {t}")
        for t in (0, 1, 37, len(demo["actions"]) // 2, len(demo["actions"]) - 1):
            observations = env.get_obs(demo["states"][t])
            for agent, name in enumerate(env.agents):
                np.testing.assert_array_equal(demo["observations"][t, agent], observations[name])

    def test_seeded_delays_are_bounded_and_reproducible(self):
        repeated = generate_demo(7, random_wait_max=2)
        np.testing.assert_array_equal(repeated["actions"], self.augmented["actions"])
        np.testing.assert_array_equal(repeated["observations"], self.augmented["observations"])
        self.assertEqual(repeated["metadata"], self.augmented["metadata"])
        self.assertGreater(repeated["metadata"]["inserted_wait_steps"], 0)
        self.assertLessEqual(repeated["metadata"]["inserted_wait_steps"], repeated["metadata"]["delay_budget"])
        self.assertLessEqual(len(repeated["actions"]), repeated["metadata"]["reference_steps"] + repeated["metadata"]["delay_budget"])
        self.assertLess(len(repeated["actions"]), repeated["metadata"]["rules"]["max_steps"])
        self.assertGreater(len(repeated["actions"]), len(self.plain["actions"]))

    def test_local_view_generation_preserves_the_same_plain_actions(self):
        local = generate_demo(0, agent_view_size=2, random_wait_max=0)
        self.assertEqual(local["observations"].shape, (len(local["actions"]), 2, 5, 5, 48))
        np.testing.assert_array_equal(local["actions"], self.plain["actions"])
        env = make_env("oc2_sushi_1_3", agent_view_size=2)
        np.testing.assert_array_equal(local["observations"][0, 0], env.get_obs(local["states"][0])[env.agents[0]])

    def test_npz_roundtrip_requires_no_pickle(self):
        with TemporaryDirectory() as folder:
            path = save_demo(self.augmented, Path(folder) / "demo.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["observations"], self.augmented["observations"])
                np.testing.assert_array_equal(saved["actions"], self.augmented["actions"])
                np.testing.assert_array_equal(saved["supervision_mask"], self.augmented["supervision_mask"])
                self.assertEqual(int(saved["lengths"]), len(self.augmented["actions"]))
                self.assertEqual(json.loads(str(saved["metadata_json"])), self.augmented["metadata"])
                self.assertEqual(saved["state_positions"].shape, (len(self.augmented["actions"]) + 1, 2, 2))
                np.testing.assert_array_equal(saved["state_deliveries"][-1], 2)

    def test_rejects_invalid_augmentation_inputs(self):
        for seed in (-1, 2**32, 0.5, True):
            with self.assertRaises(ValueError):
                generate_demo(seed)
        for maximum in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                generate_demo(0, random_wait_max=maximum)
        for view in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                generate_demo(0, agent_view_size=view)


    def test_zero_prefix_and_original_route_order_preserve_existing_data(self):
        explicit = generate_demo(0, random_wait_max=0, prefix_random_steps=0, randomize_routes=False)
        np.testing.assert_array_equal(explicit["actions"], self.plain["actions"])
        np.testing.assert_array_equal(explicit["observations"], self.plain["observations"])
        self.assertEqual(explicit["metadata"]["prefix_actions"], [])
        self.assertEqual(explicit["metadata"]["prefix_action_counts"], [[0] * 5, [0] * 5])

    def test_movement_prefix_is_legal_empty_and_unsupervised(self):
        demo = generate_demo(11, random_wait_max=0, prefix_random_steps=16)
        env = make_env("oc2_sushi_1_3")
        self.assertEqual(demo["metadata"]["prefix_random_steps"], 16)
        np.testing.assert_array_equal(demo["actions"][:16], demo["metadata"]["prefix_actions"])
        np.testing.assert_array_equal(demo["supervision_mask"][:16], 0)
        self.assertTrue(np.all((demo["actions"][:16] >= 0) & (demo["actions"][:16] <= 4)))
        for t, pair in enumerate(demo["actions"][:16]):
            _, after, _, _, info = env.step_env(jax.random.PRNGKey(0), demo["states"][t], dict(zip(env.agents, pair)))
            self.assertFalse(np.asarray(info["blocked"]).any())
            np.testing.assert_array_equal(after.positions, demo["states"][t + 1].positions)
        initial, after = demo["states"][0], demo["states"][16]
        for field in ("kind", "location", "contents", "progress", "return_due"):
            np.testing.assert_array_equal(getattr(initial, field), getattr(after, field))
        np.testing.assert_array_equal(after.positions, demo["metadata"]["expert_start_positions"])
        self.assertEqual(int(demo["states"][-1].deliveries), 2)
        self.assertTrue(demo["metadata"]["semantic_reference_passed"])
        self.assertEqual(int(np.count_nonzero(demo["supervision_mask"] == 0)), 32)

    def test_prefix_and_randomized_routes_are_reproducible_and_bounded(self):
        first = generate_demo(4, random_wait_max=1, prefix_random_steps=12, randomize_routes=True)
        repeated = generate_demo(4, random_wait_max=1, prefix_random_steps=12, randomize_routes=True)
        np.testing.assert_array_equal(first["actions"], repeated["actions"])
        np.testing.assert_array_equal(first["observations"], repeated["observations"])
        self.assertEqual(first["metadata"], repeated["metadata"])
        self.assertTrue(first["metadata"]["randomize_routes"])
        self.assertGreater(len(first["metadata"]["route_direction_orders"]), 0)
        for route in first["metadata"]["route_direction_orders"]:
            self.assertEqual(sorted(route["order"]), [0, 1, 2, 3])
        self.assertLessEqual(len(first["actions"]), 512)
        self.assertLess(len(first["actions"]), 600)
        self.assertEqual(int(first["states"][-1].deliveries), 2)
        zeros = 2 * (12 + first["metadata"]["inserted_wait_steps"])
        self.assertEqual(int(np.count_nonzero(first["supervision_mask"] == 0)), zeros)

    def test_maximum_prefix_and_large_wait_request_fit_the_episode_budget(self):
        demo = generate_demo(12, random_wait_max=100000, prefix_random_steps=64, randomize_routes=True)
        self.assertLessEqual(len(demo["actions"]), 512)
        self.assertLess(len(demo["actions"]), demo["metadata"]["rules"]["max_steps"])
        self.assertEqual(int(demo["states"][-1].deliveries), 2)
        self.assertEqual(demo["metadata"]["prefix_random_steps"], 64)
        self.assertLessEqual(demo["metadata"]["inserted_wait_steps"], demo["metadata"]["delay_budget"])
        self.assertTrue(demo["metadata"]["semantic_reference_passed"])

    def test_discarded_prefix_starts_at_real_positive_time_pose(self):
        full = generate_demo(11, random_wait_max=1, prefix_random_steps=16)
        cropped = generate_demo(11, random_wait_max=1, prefix_random_steps=16,
                                discard_prefix_context=True)
        self.assertEqual(int(cropped["states"][0].time), 16)
        self.assertEqual(cropped["metadata"]["initial_state_time"], 16)
        self.assertEqual(cropped["metadata"]["discarded_prefix_steps"], 16)
        self.assertEqual(len(cropped["actions"]), len(full["actions"]) - 16)
        np.testing.assert_array_equal(cropped["actions"], full["actions"][16:])
        np.testing.assert_array_equal(cropped["observations"], full["observations"][16:])
        np.testing.assert_array_equal(cropped["supervision_mask"], full["supervision_mask"][16:])
        for actual, original in zip(cropped["states"], full["states"][16:]):
            for a, b in zip(actual, original):
                np.testing.assert_array_equal(a, b)
        env = make_env("oc2_sushi_1_3")
        for agent, name in enumerate(env.agents):
            np.testing.assert_array_equal(cropped["observations"][0, agent], env.get_obs(cropped["states"][0])[name])
        _, after, _, _, _ = env.step_env(jax.random.PRNGKey(0), cropped["states"][0],
                                       dict(zip(env.agents, cropped["actions"][0])))
        for actual, expected in zip(after, cropped["states"][1]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(cropped["metadata"]["prefix_actions"], full["metadata"]["prefix_actions"])
        for relative, original in zip(cropped["metadata"]["milestones"], full["metadata"]["milestones"]):
            self.assertEqual(relative["step"], original["step"] - 16)
            self.assertEqual(relative["absolute_step"], original["step"])
        self.assertEqual(int(cropped["states"][-1].deliveries), 2)

    def test_discarding_empty_prefix_preserves_default_arrays(self):
        cropped = generate_demo(0, random_wait_max=0, discard_prefix_context=True)
        for field in ("observations", "actions", "supervision_mask", "lengths"):
            np.testing.assert_array_equal(cropped[field], self.plain[field])
        self.assertEqual(cropped["metadata"]["initial_state_time"], 0)

    def test_prefix_validation_rejects_unbounded_or_noninteger_inputs(self):
        for prefix in (-1, 65, 1.5, True):
            with self.assertRaises(ValueError):
                generate_demo(0, prefix_random_steps=prefix)
        with self.assertRaises(ValueError):
            generate_demo(0, randomize_routes="yes")
        with self.assertRaises(ValueError):
            generate_demo(0, discard_prefix_context="yes")


if __name__ == "__main__":
    unittest.main()
