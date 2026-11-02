"""The throughput teacher must earn every dish through unchanged real rules."""
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
from scenarios.oc2_sushi_1_3.demonstrations import save_demo
from scenarios.oc2_sushi_1_3.environment import Item, QUEUED, RECIPE
from scenarios.oc2_sushi_1_3.throughput_expert import generate_throughput_demo


class SushiThroughputExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = generate_throughput_demo()

    def test_full_original_episode_serves_at_least_five_dishes(self):
        d = self.demo
        self.assertEqual(d['observations'].shape, (600, 2, 10, 14, 48))
        self.assertEqual(d['actions'].shape, (600, 2))
        self.assertEqual(len(d['states']), 601)
        self.assertEqual(int(d['lengths']), 600)
        self.assertEqual(d['metadata']['rules']['max_steps'], 600)
        self.assertEqual(d['metadata']['rules']['cook_seconds'], 12.)
        self.assertEqual(d['metadata']['rules']['wash_seconds'], 3.)
        self.assertTrue(bool(d['states'][-1].done))
        self.assertGreaterEqual(int(d['states'][-1].deliveries), 5)
        self.assertEqual(float(d['states'][-1].score), 20 * int(d['states'][-1].deliveries))
        self.assertEqual(d['metadata']['return_wait_steps'], 0)
        self.assertEqual(d['metadata']['cook_wait_steps'], 0)
        self.assertTrue(np.all(d['supervision_mask'] == 1))
        self.assertTrue(d['metadata']['teacher_training_only'])

    def test_all_four_initial_clean_plates_are_used_before_first_wash(self):
        d = self.demo
        initial = d['states'][0]
        initial_clean = set(np.flatnonzero(np.asarray(initial.kind) == Item.PLATE).tolist())
        events = d['metadata']['delivery_events']
        self.assertEqual(set(e['plate_slot'] for e in events[:4]), initial_clean)
        self.assertGreater(d['metadata']['wash_events'][0]['step'], events[3]['step'])
        self.assertEqual([e['source'] for e in d['metadata']['plate_selection'][:4]], ['clean'] * 4)
        reused = events[4]['plate_slot']
        self.assertIn(reused, initial_clean)
        self.assertTrue(any(e['plate_slot'] == reused and e['step'] < events[4]['step']
                            for e in d['metadata']['wash_events']))
        for state in d['states']:
            kinds = np.asarray(state.kind)
            self.assertEqual(int(np.sum(kinds == Item.POT)), 3)
            self.assertEqual(int(np.sum((kinds == Item.PLATE) | (kinds == Item.DIRTY_PLATE))), 4)

    def test_every_recorded_action_replays_and_observations_precede_actions(self):
        d = self.demo
        env = make_env('oc2_sushi_1_3')
        key = jax.random.PRNGKey(0)
        _, reset = env.reset(key)
        for a, b in zip(reset, d['states'][0]):
            np.testing.assert_array_equal(a, b)
        for t, pair in enumerate(d['actions']):
            _, after, _, _, _ = env.step_env(key, d['states'][t], dict(zip(env.agents, pair)))
            for actual, expected in zip(after, d['states'][t + 1]):
                np.testing.assert_array_equal(actual, expected, err_msg=f'transition {t}')
        for t in (0, 95, 96, 307, 551, 599):
            observations = env.get_obs(d['states'][t])
            for agent, name in enumerate(env.agents):
                np.testing.assert_array_equal(observations[name], d['observations'][t, agent])

    def test_stationary_pots_and_delivery_logs_match_real_item_transitions(self):
        d = self.demo
        initial = d['states'][0]
        for state in d['states']:
            np.testing.assert_array_equal(state.location[:3], initial.location[:3])
        deliveries, washes = [], []
        for step, (before, after) in enumerate(zip(d['states'][:-1], d['states'][1:]), start=1):
            delivered = np.flatnonzero((np.asarray(before.kind) == Item.PLATE)
                & (np.asarray(before.contents) == RECIPE)
                & (np.asarray(after.kind) == Item.DIRTY_PLATE)
                & (np.asarray(after.location) == QUEUED))
            washed = np.flatnonzero((np.asarray(before.kind) == Item.DIRTY_PLATE)
                                   & (np.asarray(after.kind) == Item.PLATE))
            deliveries.extend({'step': step, 'plate_slot': int(slot)} for slot in delivered)
            washes.extend({'step': step, 'plate_slot': int(slot)} for slot in washed)
        self.assertEqual(deliveries, d['metadata']['delivery_events'])
        self.assertEqual(washes, d['metadata']['wash_events'])
        self.assertEqual(d['metadata']['delivery_steps'], [e['step'] for e in deliveries])

    def test_short_variant_records_native_stations_and_earlier_deliveries(self):
        short = generate_throughput_demo(routing="short")
        self.assertEqual(short["metadata"]["delivery_steps"], [83, 175, 274, 378, 497])
        self.assertEqual(short["metadata"]["first_delivery_seconds"], 20.75)
        self.assertEqual(short["metadata"]["last_delivery_seconds"], 124.25)
        self.assertEqual(short["metadata"]["routing"], "short")
        self.assertEqual(short["metadata"]["stations"], {"shared": [6, 3], "board": [3, 1],
                         "assembly": [3, 4], "hob": [10, 0]})
        self.assertEqual(short["metadata"]["deliveries"], 5)
        self.assertLess(short["metadata"]["last_delivery_step"], self.demo["metadata"]["last_delivery_step"])
        self.assertEqual(self.demo["metadata"]["delivery_steps"], [96, 197, 308, 422, 551])
        with self.assertRaises(ValueError):
            generate_throughput_demo(routing="unknown")

    def test_optional_nori_first_reorders_recipe_without_changing_rules(self):
        demo = generate_throughput_demo(routing="short", nori_first=True)
        self.assertTrue(demo["metadata"]["nori_first"])
        self.assertFalse(self.demo["metadata"]["nori_first"])
        self.assertEqual(demo["metadata"]["delivery_steps"], [87, 180, 280, 385, 505])
        first_nori = next(i for i, state in enumerate(demo["states"])
                          if int(state.kind[3]) == Item.PLATE and int(state.contents[3]) & 4)
        first_cucumber = next(i for i, state in enumerate(demo["states"])
                             if int(state.kind[3]) == Item.PLATE and int(state.contents[3]) & 1)
        self.assertLess(first_nori, first_cucumber)
        self.assertEqual(demo["metadata"]["rules"], self.demo["metadata"]["rules"])
        self.assertEqual(int(demo["states"][-1].deliveries), 5)
        env = make_env("oc2_sushi_1_3")
        key = jax.random.PRNGKey(0)
        for t, actions in enumerate(demo["actions"]):
            _, after, _, _, _ = env.step_env(key, demo["states"][t], dict(zip(env.agents, actions)))
            for actual, expected in zip(after, demo["states"][t + 1]):
                np.testing.assert_array_equal(actual, expected, err_msg=f"nori-first transition {t}")
        with self.assertRaises(ValueError):
            generate_throughput_demo(nori_first="yes")

    def test_optional_second_counter_prefetch_loads_next_rice_before_serving(self):
        demo = generate_throughput_demo(routing="short", nori_first=True, prefetch_rice=True)
        self.assertTrue(demo["metadata"]["prefetch_rice"])
        self.assertFalse(self.demo["metadata"]["prefetch_rice"])
        self.assertEqual(demo["metadata"]["rice_buffer"], [6, 2])
        self.assertEqual(demo["metadata"]["delivery_steps"], [109, 200, 312, 429, 561])
        loads = demo["metadata"]["rice_load_events"]
        self.assertEqual([e["for_dish"] for e in loads], [1, 2, 3, 4, 5, 6])
        self.assertFalse(loads[0]["prefetched"])
        for event in loads[1:]:
            self.assertTrue(event["prefetched"])
            self.assertLess(event["step"], demo["metadata"]["delivery_steps"][event["for_dish"] - 2])
        env = make_env("oc2_sushi_1_3")
        buffer_cell = env.cell((6, 2))
        buffered = [i for i, state in enumerate(demo["states"])
                    if np.any((np.asarray(state.kind) == Item.RICE)
                              & (np.asarray(state.location) == buffer_cell))]
        self.assertTrue(buffered)
        self.assertLess(buffered[0], demo["metadata"]["first_delivery_step"])
        for t, pair in enumerate(demo["actions"]):
            _, after, _, _, _ = env.step_env(jax.random.PRNGKey(0), demo["states"][t], dict(zip(env.agents, pair)))
            for actual, expected in zip(after, demo["states"][t + 1]):
                np.testing.assert_array_equal(actual, expected, err_msg=f"prefetch transition {t}")
        with self.assertRaises(ValueError):
            generate_throughput_demo(prefetch_rice="yes")

    def test_safe_npz_training_schema(self):
        with TemporaryDirectory() as folder:
            path = save_demo(self.demo, Path(folder) / 'throughput.npz')
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(int(stored['lengths']), 600)
                self.assertEqual(stored['observations'].shape[1:], (2, 10, 14, 48))
                np.testing.assert_array_equal(stored['actions'], self.demo['actions'])
                self.assertGreaterEqual(json.loads(str(stored['metadata_json']))['deliveries'], 5)
                self.assertEqual(stored['state_positions'].shape, (601, 2, 2))


if __name__ == '__main__':
    unittest.main()
