"""Targeted numeric checks for experimental one-step critic probes.

Run with the simulator dependencies: python -m unittest tests.test_critic_probe
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jaxmarl  # noqa: E402
from critic_probe import (  # noqa: E402
    action_names,
    build_probe_payload,
    candidate_action_grid,
    make_probe_runner,
)
from evaluate import make_episode  # noqa: E402


class _ScannedRNN:
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class _Policy:
    """Minimal Distrax-like categorical policy for the fake recurrent model."""

    def __init__(self, logits):
        self.logits = logits
        self.probs = jax.nn.softmax(logits, axis=-1)

    def sample(self, seed):
        return jax.random.categorical(seed, self.logits)


class _Network:
    """Deterministic stand-in whose critic depends on recurrent memory."""

    def apply(self, params, hidden, inputs):
        observations, dones = inputs
        observations = jnp.asarray(observations)
        hidden = jnp.asarray(hidden)
        features = (
            observations.reshape(hidden.shape[0], -1)[:, : hidden.shape[-1]]
            + 1.0
        )
        resets = jnp.asarray(dones).reshape(hidden.shape[0])
        new_hidden = jnp.where(resets[:, None], features, hidden + features)
        values = jnp.sum(new_hidden, axis=-1)[None, :]
        logits = jnp.zeros((1, hidden.shape[0], len(action_names())))
        return new_hidden, _Policy(logits), values


def assert_trees_equal(test, left, right, label):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    test.assertEqual(len(left_leaves), len(right_leaves), label)
    for first, second in zip(left_leaves, right_leaves):
        if first is None and second is None:
            continue
        np.testing.assert_array_equal(
            np.asarray(first), np.asarray(second), err_msg=label
        )


class CriticProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {"GRU_HIDDEN_DIM": 4}
        cls.env = jaxmarl.make(
            "overcooked_v2",
            layout="coord_ring",
            max_steps=3,
            agent_view_size=2,
            random_agent_positions=False,
            negative_rewards=True,
        )
        cls.module = SimpleNamespace(ScannedRNN=_ScannedRNN)
        cls.network = _Network()
        cls.params = {"dummy": jnp.array(0.0)}

        plain_play = make_episode(
            cls.module, cls.config, cls.env, cls.network, probe=False
        )
        probe_play = make_episode(
            cls.module, cls.config, cls.env, cls.network, probe=True
        )
        cls.plain = jax.device_get(plain_play(cls.params, np.uint32(7)))
        cls.probed = jax.device_get(probe_play(cls.params, np.uint32(7)))
        cls.initial, cls.trace, cls.inputs = cls.probed

        runner = make_probe_runner(cls.env, cls.network)
        cls.result = jax.device_get(
            runner(
                cls.params,
                cls.inputs["states"],
                cls.inputs["observations"],
                cls.inputs["hidden"],
                cls.inputs["values"],
                cls.inputs["actions"],
                cls.inputs["rewards"],
                cls.inputs["dones"],
                cls.inputs["step_keys"],
            )
        )
        cls.payload = build_probe_payload(
            cls.inputs["values"],
            cls.inputs["actions"],
            cls.inputs["rewards"],
            cls.inputs["dones"],
            cls.result,
        )

    def test_probe_collection_does_not_change_factual_rollout(self):
        assert_trees_equal(
            self, self.plain, (self.initial, self.trace), "plain vs probe episode"
        )

    def test_probe_steps_cover_terminal_and_use_pre_action_values(self):
        steps = self.payload["steps"]
        self.assertEqual(set(self.payload), {"reward", "action_names", "steps"})
        self.assertEqual(
            self.payload["reward"], "raw environment reward; no shaping"
        )
        self.assertEqual(len(steps), self.env.max_steps + 1)
        self.assertEqual(len(self.inputs["values"]), self.env.max_steps + 1)
        np.testing.assert_allclose(
            [step["values"] for step in steps], self.inputs["values"]
        )
        self.assertEqual(self.payload["action_names"], action_names())
        self.assertEqual(
            self.payload["action_names"],
            ["right", "down", "left", "up", "stay", "interact"],
        )
        self.assertIsNone(steps[-1]["actual_actions"])
        self.assertEqual(steps[-1]["interventions"], [])
        for step in steps:
            self.assertEqual(
                set(step), {"values", "actual_actions", "interventions"}
            )
            self.assertEqual(len(step["values"]), 2)
            self.assertTrue(np.isfinite(step["values"]).all())

    def test_intervention_baseline_matches_factual_transition(self):
        actions = np.asarray(self.trace[1])
        rewards = np.asarray(self.trace[2])
        dones = np.asarray(self.inputs["dones"])
        values = np.asarray(self.inputs["values"])
        for index in range(len(actions)):
            for intervention in self.payload["steps"][index]["interventions"]:
                actor = intervention["actor"]
                target = intervention["target"]
                baseline = intervention["baseline"]
                self.assertEqual(target, 1 - actor)
                self.assertEqual(baseline["action"], int(actions[index, actor]))
                self.assertAlmostEqual(
                    baseline["next_value"], float(values[index + 1, target])
                )
                self.assertAlmostEqual(baseline["reward"], float(rewards[index]))
                self.assertEqual(baseline["done"], int(dones[index]))
                alternatives = intervention["alternatives"]
                self.assertEqual(
                    [alt["action"] for alt in alternatives],
                    list(range(len(action_names()))),
                )
                unchanged = next(
                    alt for alt in alternatives
                    if alt["action"] == baseline["action"]
                )
                for key in ("next_value", "reward", "done"):
                    self.assertEqual(unchanged[key], baseline[key])
                self.assertEqual(unchanged["value_delta"], 0.0)

    def test_successor_values_use_post_current_memory(self):
        observations = jnp.asarray(self.inputs["observations"])
        hidden = jnp.asarray(self.inputs["hidden"])
        dones = jnp.asarray(self.inputs["dones"])
        actions = np.asarray(self.inputs["actions"])
        transitions, num_agents = actions.shape
        next_observations = observations[1:].reshape(
            (1, transitions * num_agents) + observations.shape[2:]
        )
        next_dones = jnp.repeat(dones, num_agents)[None]

        def values(memory):
            _, _, result = self.network.apply(
                self.params, memory, (next_observations, next_dones)
            )
            return np.asarray(result[0]).reshape(transitions, num_agents)

        # The critic must see the memory returned *after* the current
        # observation, not the stale pre-observation carry or a reset carry.
        post = values(hidden[1:].reshape(-1, hidden.shape[-1]))
        stale = values(hidden[:-1].reshape(-1, hidden.shape[-1]))
        reset = values(jnp.zeros_like(hidden[1:]).reshape(-1, hidden.shape[-1]))
        factual = np.take_along_axis(
            np.asarray(self.result["next_values"]),
            actions[:, :, None],
            axis=2,
        )[:, :, 0]
        # An actor's row reports the other chef's value.
        np.testing.assert_allclose(
            factual, post[:, [1, 0]], rtol=0, atol=1e-6
        )
        self.assertGreater(float(np.max(np.abs(stale - post))), 1e-3)
        self.assertGreater(float(np.max(np.abs(reset - post))), 1e-3)

    def test_value_delta_sign_follows_next_value(self):
        values = np.array([[0.0, 0.0], [0.0, 0.0]])
        actions = np.zeros((1, 2), dtype=int)
        rewards = np.array([1.0])
        dones = np.array([False])
        result = {
            "next_values": np.zeros((1, 2, len(action_names()))),
            "rewards": np.ones((1, 2, len(action_names()))),
            "dones": np.zeros((1, 2, len(action_names())), dtype=bool),
            "factual_ok": np.ones((1, 2), dtype=bool),
            "max_state_difference": np.zeros(1),
            "max_observation_difference": np.zeros(1),
            "max_value_difference": np.zeros(1),
        }
        result["next_values"][0, 0, 1] = 2.0
        result["next_values"][0, 0, 2] = -3.0
        payload = build_probe_payload(values, actions, rewards, dones, result)
        alternatives = payload["steps"][0]["interventions"][0]["alternatives"]
        self.assertEqual(alternatives[1]["value_delta"], 2.0)
        self.assertEqual(alternatives[2]["value_delta"], -3.0)

    def test_terminal_state_value_is_diagnostic_and_has_no_outgoing_action(self):
        # Huge terminal values must still be reported as raw diagnostics only.
        values = np.array([[0.0, 0.5], [100.0, 200.0], [999.0, 999.0]])
        actions = np.zeros((2, 2), dtype=int)
        rewards = np.array([2.0, 3.0])
        dones = np.array([False, True])
        result = {
            "next_values": np.zeros((2, 2, len(action_names()))),
            "rewards": np.zeros((2, 2, len(action_names()))),
            "dones": np.ones((2, 2, len(action_names())), dtype=bool),
            "factual_ok": np.ones((2, 2), dtype=bool),
            "max_state_difference": np.zeros(2),
            "max_observation_difference": np.zeros(2),
            "max_value_difference": np.zeros(2),
        }
        payload = build_probe_payload(values, actions, rewards, dones, result)
        final = payload["steps"][2]
        self.assertIsNone(final["actual_actions"])
        self.assertEqual(final["interventions"], [])
        self.assertEqual(final["values"], [999.0, 999.0])

    def test_candidate_grid_changes_only_the_selected_actor(self):
        actions = jnp.array([2, 5])
        grid = np.asarray(candidate_action_grid(actions, 2, len(action_names())))
        self.assertEqual(grid.shape, (2, len(action_names()), 2))
        for actor in range(2):
            for code in range(len(action_names())):
                for agent in range(2):
                    expected = code if agent == actor else int(actions[1 - actor])
                    self.assertEqual(grid[actor, code, agent], expected)


if __name__ == "__main__":
    unittest.main()
