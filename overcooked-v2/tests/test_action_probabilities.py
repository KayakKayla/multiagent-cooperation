"""Probability recording checks with three-step toy policies, never training."""

import copy
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import distrax
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
from evals_shared.format import parse_document
from preview import draw_action_probability_panel


class ProbabilityPolicy:
    """Soft probabilities change with each chef's memory, exposing shifted rows."""

    def apply(self, weights, hidden, inputs):
        logits = weights["logits"] + hidden[:, :1] * jnp.arange(6) * 0.25
        return (hidden + weights["increment"],
                distrax.Categorical(logits=logits[None]), None)


class ActionProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            "ENV_NAME": "overcooked_v2", "SEED": 10, "GRU_HIDDEN_DIM": 2,
            "ALGORITHM": "ippo_rnn",
            "ENV_KWARGS": {"layout": "coord_ring", "max_steps": 3,
                           "agent_view_size": 2, "random_agent_positions": True},
        }
        cls.env = jaxmarl.make(cls.config["ENV_NAME"], **cls.config["ENV_KWARGS"])
        cls.module = SimpleNamespace(
            distrax=distrax,
            ScannedRNN=SimpleNamespace(
                initialize_carry=lambda batch, size: jnp.zeros((batch, size))),
        )
        cls.network = ProbabilityPolicy()
        cls.a = {"logits": jnp.array([0., 0.5, 1., -0.5, 0.25, -1.]),
                 "increment": jnp.array(1.)}
        cls.b = {"logits": jnp.array([1., -0.5, 0., 0.25, -1., 0.5]),
                 "increment": jnp.array(2.)}
        cls.legacy = staticmethod(evaluate.make_episode(
            cls.module, cls.config, cls.env, cls.network))
        cls.recorded = staticmethod(evaluate.make_episode(
            cls.module, cls.config, cls.env, cls.network, record_action_probs=True))

    def expected_probabilities(self, weights):
        return np.asarray(jax.nn.softmax(jnp.stack([
            weights["logits"] + step * weights["increment"] * jnp.arange(6) * 0.25
            for step in range(3)
        ]), axis=-1))

    def test_pre_action_probabilities_for_shared_and_separate_policies(self):
        for partner in (None, self.a, self.b):
            with self.subTest(partner="shared" if partner is None else "separate"):
                _, trace, probabilities = jax.device_get(
                    self.recorded(self.a, 1000, partner))
                self.assertEqual(probabilities.shape, (3, 2, 6))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=-1), 1., atol=1e-6)
                for chef, weights in enumerate((self.a, self.a if partner is None else partner)):
                    np.testing.assert_allclose(
                        probabilities[:, chef], self.expected_probabilities(weights),
                        rtol=1e-6, atol=1e-7)
                self.assertTrue(np.asarray(trace[-1]).all())
                self.assertTrue(bool(trace[0].done[-1]))

    def test_enabling_recording_preserves_sampled_actions_and_all_states(self):
        for partner in (None, self.a, self.b):
            with self.subTest(partner="shared" if partner is None else "separate"):
                initial, trace, _ = self.recorded(self.a, 1000, partner)
                legacy_initial, legacy_trace = self.legacy(self.a, 1000, partner)
                self.assertEqual(len(trace), 6)
                for actual, expected in zip(
                    jax.tree.leaves((initial, trace)),
                    jax.tree.leaves((legacy_initial, legacy_trace)), strict=True,
                ):
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_each_episode_is_archived_and_first_episode_is_previewed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "UPSTREAM_REVISION").write_text("test-revision\n")
            with (
                patch.object(evaluate, "SCRIPT_DIR", root),
                patch.object(evaluate, "load_baseline",
                             return_value=(self.module, copy.deepcopy(self.config))),
                patch.object(evaluate, "make_env", return_value=self.env),
                patch.object(evaluate, "initialize_policy", return_value=(self.network, self.a)),
                patch.object(evaluate, "save_preview") as save_preview,
                patch.object(sys, "argv", ["evaluate.py", "--untrained", "--episodes", "2"]),
                redirect_stdout(StringIO()),
            ):
                evaluate.main()

            output, = (root / "outputs/evaluate").iterdir()
            self.assertEqual(len(list(output.glob("decisions-*.npz"))), 2)
            exported = parse_document((output / "evaluation.json").read_bytes())
            self.assertEqual(len(exported["episodes"]), 2)
            self.assertEqual(len(exported["checkpoints"]), 1)
            self.assertEqual(exported["checkpoints"][0]["progress"], 0)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["gif_stride"], 1)
            self.assertEqual(protocol["gif_frame_ms"], 100)
            first_probabilities = None
            first_actions = None
            for episode, seed in enumerate((1000, 1001)):
                _, expected_trace, expected_probabilities = jax.device_get(
                    self.recorded(self.a, seed))
                with np.load(output / f"decisions-{episode:03d}.npz") as saved:
                    self.assertEqual(set(saved.files), {
                        "action_probs", "actions", "agent_ids", "evaluation_seed"})
                    np.testing.assert_array_equal(saved["action_probs"], expected_probabilities)
                    np.testing.assert_array_equal(saved["actions"], expected_trace[1])
                    np.testing.assert_array_equal(saved["agent_ids"], self.env.agents)
                    self.assertEqual(int(saved["evaluation_seed"]), seed)
                    evidence = exported["episodes"][episode]
                    self.assertEqual(evidence["seed"], seed)
                    steps = evidence["recording"]["steps"]
                    self.assertEqual(len(steps), 4)
                    self.assertTrue(all(agent["action_code"] is None for agent in steps[0]["agents"]))
                    np.testing.assert_array_equal(
                        [[agent["action_code"] for agent in step["agents"]]
                         for step in steps[1:]], saved["actions"])
                    np.testing.assert_array_equal(
                        [[agent["position"] for agent in step["agents"]]
                         for step in steps[1:]],
                        np.stack((expected_trace[0].agents.pos.x,
                                  expected_trace[0].agents.pos.y), axis=-1))
                    if episode == 0:
                        first_probabilities = saved["action_probs"].copy()
                        first_actions = saved["actions"].copy()

            save_preview.assert_called_once()
            call = save_preview.call_args
            self.assertEqual(call.args[5], 1000)
            np.testing.assert_array_equal(call.kwargs["action_probs"], first_probabilities)
            np.testing.assert_array_equal(call.args[1][1], first_actions)


class ProbabilityPanelTests(unittest.TestCase):
    def test_selected_action_is_highlighted_even_when_it_is_not_most_probable(self):
        draw = Mock()
        draw_action_probability_panel(
            draw, x=0, y=0, title="Chef 1", probabilities=[0.75, 0.05, 0.05, 0.05, 0.05, 0.05],
            selected_action=5, font=None, small_font=None, color="#ff6b6b",
        )
        labels = {call.args[1]: call.kwargs["fill"] for call in draw.text.call_args_list}
        self.assertEqual(labels["> interact"], "#ff6b6b")
        self.assertEqual(labels["  right"], "#cbd5e1")
        self.assertNotIn("> right", labels)


if __name__ == "__main__":
    unittest.main()
