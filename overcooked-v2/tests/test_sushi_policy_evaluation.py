"""Closed-loop sushi evaluation checks; CPU only, with no expert dependencies."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "JaxMARL")]
from evaluate import initialize_policy, make_episode as legacy_episode
import evaluate_sushi_policy as evaluation
from scenarios import make_env
from scenarios.oc2_sushi_1_3.environment import Action, Item, QUEUED, RECIPE
from scenarios.oc2_sushi_1_3.layout import Tile
from train import load_baseline


class FakePolicy:
    def __init__(self):
        self.probs = jax.nn.one_hot(jnp.array([[Action.STAY, Action.STAY]]), 7)

    def sample(self, seed):
        del seed
        return jnp.array([[Action.RIGHT, Action.RIGHT]])


class FakeNetwork:
    def apply(self, params, hidden, inputs):
        del params, inputs
        return hidden, FakePolicy(), jnp.zeros((1, 2))


class SushiPolicyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.config = load_baseline(ROOT / "configs/oc2_sushi_check.yaml", None)
        cls.env = make_env("oc2_sushi_1_3", rules={"max_steps": 4}, agent_view_size=2)
        _, cls.params = initialize_policy(cls.module, cls.config, cls.env)

    def test_actions_come_from_policy_in_actual_environment(self):
        helper = SimpleNamespace(ScannedRNN=SimpleNamespace(
            initialize_carry=lambda count, width: jnp.zeros((count, width))))
        traces = {}
        for mode in ("sampled", "greedy"):
            play = evaluation.make_episode(helper, {"GRU_HIDDEN_DIM": 2}, self.env, FakeNetwork(), mode)
            initial, trace, diagnostics = jax.device_get(play({}, np.uint32(1000)))
            row = evaluation.summarize_episode(initial, trace, diagnostics, self.env, 1000, 0)
            self.assertEqual(row["correct_dishes"], 0)
            self.assertEqual(row["steps"], 4)
            traces[mode] = (initial, trace)
        np.testing.assert_array_equal(traces["greedy"][1][1], np.full((4, 2), Action.STAY))
        np.testing.assert_array_equal(traces["sampled"][1][1], np.full((4, 2), Action.RIGHT))
        np.testing.assert_array_equal(traces["greedy"][1][0].positions[-1], traces["greedy"][0].positions)
        self.assertFalse(np.array_equal(traces["sampled"][1][0].positions[-1], traces["sampled"][0].positions))

    def test_zero_noise_preserves_existing_policy_and_reset_rng(self):
        network, _ = initialize_policy(self.module, self.config, self.env)
        old = legacy_episode(self.module, self.config, self.env, network)
        new = evaluation.make_episode(self.module, self.config, self.env, network,
                                      action_dropout=0., random_action_prob=0., spawn_jitter=0)
        for seed in (1000, 1001):
            reference = jax.device_get(old(self.params, np.uint32(seed)))
            initial, trace, diagnostics = jax.device_get(new(self.params, np.uint32(seed)))
            for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves((initial, trace))):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(diagnostics[4], trace[1])
            self.assertFalse(np.asarray(diagnostics[5]).any())
            self.assertFalse(np.asarray(diagnostics[6]).any())

    def test_action_dropout_overrides_only_executed_actions(self):
        helper = SimpleNamespace(ScannedRNN=SimpleNamespace(
            initialize_carry=lambda count, width: jnp.zeros((count, width))))
        play = evaluation.make_episode(helper, {"GRU_HIDDEN_DIM": 2}, self.env, FakeNetwork(),
                                       action_dropout=1.)
        initial, trace, diagnostics = jax.device_get(play({}, np.uint32(1000)))
        row = evaluation.summarize_episode(initial, trace, diagnostics, self.env, 1000, 0)
        np.testing.assert_array_equal(row["proposed_actions"], np.full((4, 2), Action.RIGHT))
        np.testing.assert_array_equal(row["executed_actions"], np.full((4, 2), Action.STAY))
        np.testing.assert_array_equal(trace[0].positions[-1], initial.positions)
        self.assertEqual(row["dropout_selections"], 8)
        self.assertEqual(row["perturbed_actions"], 8)
        self.assertEqual(row["random_action_selections"], 0)

    def test_random_action_sweep_is_reproducible_and_logs_actual_overrides(self):
        helper = SimpleNamespace(ScannedRNN=SimpleNamespace(
            initialize_carry=lambda count, width: jnp.zeros((count, width))))
        play = evaluation.make_episode(helper, {"GRU_HIDDEN_DIM": 2}, self.env, FakeNetwork(),
                                       random_action_prob=1.)
        initial, trace, diagnostics = jax.device_get(play({}, np.uint32(1000)))
        repeat = jax.device_get(play({}, np.uint32(1000)))
        for a, b in zip(jax.tree.leaves((initial, trace, diagnostics)), jax.tree.leaves(repeat)):
            np.testing.assert_array_equal(a, b)
        row = evaluation.summarize_episode(initial, trace, diagnostics, self.env, 1000, 0)
        proposed, executed = np.asarray(row["proposed_actions"]), np.asarray(row["executed_actions"])
        np.testing.assert_array_equal(proposed, np.full((4, 2), Action.RIGHT))
        self.assertTrue(((executed >= 0) & (executed < 7)).all())
        self.assertEqual(row["random_action_selections"], 8)
        self.assertEqual(row["dropout_selections"], 0)
        self.assertEqual(row["perturbed_actions"], int(np.count_nonzero(proposed != executed)))

    def test_spawn_jitter_changes_only_valid_distinct_nearby_poses(self):
        _, original = self.env.reset(jax.random.PRNGKey(0))
        positions_seen, facings_seen = set(), set()
        for radius in (0, 1, 2):
            jitter = jax.jit(lambda seed: evaluation.perturb_initial_state(
                self.env, original, seed, spawn_jitter=radius, random_facing=True))
            unchanged_facing = jax.jit(lambda seed: evaluation.perturb_initial_state(
                self.env, original, seed, spawn_jitter=radius, random_facing=False))
            for seed in range(10):
                state, default_facing = jax.device_get((jitter(np.uint32(seed)), unchanged_facing(np.uint32(seed))))
                positions, directions = np.asarray(state.positions), np.asarray(state.directions)
                self.assertTrue((np.abs(positions - np.asarray(original.positions)).sum(axis=1) <= radius).all())
                self.assertFalse(np.array_equal(positions[0], positions[1]))
                self.assertTrue(all(int(self.env.tiles[y, x]) == Tile.FLOOR for x, y in positions))
                self.assertTrue(((directions >= 0) & (directions < 4)).all())
                np.testing.assert_array_equal(positions, default_facing.positions)
                np.testing.assert_array_equal(default_facing.directions, original.directions)
                for name in original._fields:
                    if name not in ("positions", "directions"):
                        np.testing.assert_array_equal(getattr(state, name), getattr(original, name))
                positions_seen.add(tuple(positions.reshape(-1)))
                facings_seen.add(tuple(directions))
        self.assertGreater(len(positions_seen), 1)
        self.assertGreater(len(facings_seen), 1)

    def test_robustness_parameters_and_initial_poses_are_saved(self):
        with TemporaryDirectory() as tmp:
            run, output = Path(tmp) / "run", Path(tmp) / "evaluation"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(self.config))
            (run / "params.msgpack").write_bytes(serialization.to_bytes(self.params))
            evaluation.evaluate_run(run, output, episodes=1, steps=4, replay=False,
                                    action_dropout=.25, spawn_jitter=1, random_facing=True)
            protocol = json.loads((output / "protocol.json").read_text())
            rows = json.loads((output / "episodes.json").read_text())
            self.assertEqual(protocol["evaluator_source_sha256"], evaluation.ckpt.file_hash(Path(evaluation.__file__)))
            self.assertEqual(protocol["action_dropout"], .25)
            self.assertEqual(protocol["random_action_prob"], 0.)
            self.assertEqual(protocol["spawn_jitter"], 1)
            self.assertEqual(protocol["spawn_jitter_distance"], "Manhattan")
            self.assertTrue(protocol["random_facing"])
            self.assertIn("not an unseen-layout test", protocol["seed_scope"])
            self.assertEqual(protocol["initial_conditions"][0]["positions"], rows[0]["initial_positions"])
            self.assertEqual(protocol["initial_conditions"][0]["facings"], rows[0]["initial_facings"])
            self.assertEqual(len(rows[0]["proposed_actions"]), 4)
            self.assertEqual(len(rows[0]["executed_actions"]), 4)
            self.assertFalse((output / "replay.html").exists())

    def prepared_delivery_trace(self):
        key = jax.random.PRNGKey(0)
        _, initial = self.env.reset(key)
        initial = initial._replace(
            positions=initial.positions.at[0].set(jnp.array([1, 6])),
            directions=initial.directions.at[0].set(Action.LEFT),
            location=initial.location.at[3].set(-1), contents=initial.contents.at[3].set(RECIPE))
        state, states, rewards, deltas, masks, washes = initial, [], [], [], [], []
        for _ in range(self.env.max_steps):
            _, after, reward, _, info = self.env.step_env(key, state,
                {"agent_0": Action.PICKUP_PLACE, "agent_1": Action.STAY})
            states.append(after);rewards.append(reward["agent_0"]);deltas.append(after.deliveries-state.deliveries)
            masks.append((state.kind == Item.PLATE) & (after.kind == Item.DIRTY_PLATE) & (after.location == QUEUED))
            washes.append((state.kind == Item.DIRTY_PLATE) & (after.kind == Item.PLATE))
            state = after
        batched = jax.tree.map(lambda *x: jnp.stack(x), *states)
        trace = (batched, jnp.tile(jnp.array([Action.PICKUP_PLACE, Action.STAY]), (4, 1)),
                 jnp.array(rewards), jnp.array(deltas), jnp.zeros(4), jnp.ones(4, dtype=bool))
        diagnostics = (jnp.stack(masks), jnp.stack(washes), jnp.zeros((4, 2)), jnp.zeros((4, 2)))
        return jax.device_get((initial, trace, diagnostics))

    def test_actual_delivery_and_shared_reward_are_counted_once(self):
        initial, trace, diagnostics = self.prepared_delivery_trace()
        row = evaluation.summarize_episode(initial, trace, diagnostics, self.env, 1000, 0)
        self.assertEqual(row["correct_dishes"], 1)
        self.assertEqual(row["raw_team_reward"], 20.)
        self.assertEqual(row["first_delivery_step"], 1)
        self.assertEqual(row["first_delivery_seconds"], .25)
        self.assertIsNone(row["second_delivery_step"])
        self.assertEqual(row["reused_plate_deliveries"], 0)
        self.assertTrue(row["success_at_least_one"])
        self.assertFalse(row["success_at_least_two"])

    def test_inconsistent_reward_counter_and_nonfinite_flags_are_rejected(self):
        initial, trace, diagnostics = self.prepared_delivery_trace()
        corrupt = list(trace);corrupt[2] = trace[2] * 2
        with self.assertRaisesRegex(ValueError, "Raw reward"):
            evaluation.summarize_episode(initial, tuple(corrupt), diagnostics, self.env, 1000, 0)
        corrupt = list(trace);corrupt[3] = np.zeros_like(trace[3])
        with self.assertRaisesRegex(ValueError, "Delivery counter"):
            evaluation.summarize_episode(initial, tuple(corrupt), diagnostics, self.env, 1000, 0)
        corrupt = list(trace);corrupt[5] = np.zeros_like(trace[5])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluation.summarize_episode(initial, tuple(corrupt), diagnostics, self.env, 1000, 0)

    def test_saved_policy_outputs_protocol_and_real_first_episode_replay(self):
        with TemporaryDirectory() as tmp:
            run, output = Path(tmp) / "run", Path(tmp) / "evaluation"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(self.config))
            (run / "params.msgpack").write_bytes(serialization.to_bytes(self.params))
            result_path, result = evaluation.evaluate_run(run, output, episodes=2, steps=4,
                                                         action_selection="greedy", replay=True)
            self.assertEqual(result_path, output.resolve())
            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["success_rate_at_least_one"], 0.)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertFalse(protocol["expert_or_planner_used"])
            self.assertFalse(protocol["learning_during_evaluation"])
            self.assertIn("Reset ignores seeds", protocol["seed_scope"])
            self.assertIn("identical deterministic", protocol["greedy_scope"])
            self.assertEqual(protocol["training_episode_steps"], 8)
            self.assertEqual(protocol["episode_steps"], 4)
            self.assertTrue((output / "episodes.csv").exists())
            self.assertTrue((output / "replay.html").exists())
            self.assertTrue((output / "rollout.gif").exists())
            self.assertEqual(json.loads((run / "config.json").read_text())["ENV_KWARGS"]["rules"]["max_steps"], 8)
            from evals_shared.format import parse_document
            exported = parse_document((output / "evaluation.json").read_bytes())
            self.assertEqual(len(exported["episodes"]), 2)
            self.assertEqual(len(exported["checkpoints"]), 1)
            self.assertEqual(exported["environment"]["config"]["rules"]["max_steps"], 4)
            self.assertEqual(exported["training_runs"][0]["config"]["ENV_KWARGS"]["rules"]["max_steps"], 8)
            self.assertIsNone(exported["checkpoints"][0]["progress"])
            for index, episode in enumerate(exported["episodes"]):
                self.assertEqual(episode["recording"]["metadata"]["episode"], index)
                self.assertEqual(len(episode["recording"]["steps"]), 5)
                self.assertIsNone(episode["recording"]["steps"][0]["agents"][0]["action"])
                self.assertIn("proposed_action_code", episode["recording"]["steps"][1]["agents"][0])
            self.assertNotIn(str(run), (output / "evaluation.json").read_text())
            self.assertNotIn('"frames"', (output / "evaluation.json").read_text())
            self.assertNotIn('"params"', (output / "evaluation.json").read_text())

    def test_no_replay_and_argument_bounds(self):
        with TemporaryDirectory() as tmp:
            run, output = Path(tmp) / "run", Path(tmp) / "evaluation"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(self.config))
            (run / "params.msgpack").write_bytes(serialization.to_bytes(self.params))
            with patch("scenarios.oc2_sushi_1_3.preview.save_policy_preview") as render:
                evaluation.evaluate_run(run, output, episodes=1, steps=4, replay=False)
                render.assert_not_called()
            self.assertFalse((output / "replay.html").exists())
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "evaluation.json").exists())
        for kwargs in ({"episodes": 0}, {"steps": 0}, {"eval_seed": -1}, {"action_selection": "expert"},
                       {"action_dropout": -.1}, {"action_dropout": 1.1}, {"action_dropout": float("nan")},
                       {"random_action_prob": -1}, {"random_action_prob": 2},
                       {"action_dropout": .1, "random_action_prob": .1},
                       {"spawn_jitter": -1}, {"spawn_jitter": 3}, {"spawn_jitter": 1.5}):
            with self.assertRaises(ValueError):
                evaluation.evaluate_run("nonexistent", **kwargs)


if __name__ == "__main__":
    unittest.main()
