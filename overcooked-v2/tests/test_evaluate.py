"""Short routing/memory checks with a toy policy, not research evaluations."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import jaxmarl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import check_partner_config, check_run_revision, load_checkpoint, make_episode


class MemoryPolicy:
    """Distinct preferred actions and memory increments make chef mix-ups visible."""
    def apply(self, weights, hidden, inputs):
        action = jnp.where(hidden[:, 0] >= 2, 3, weights["action"])
        logits = jnp.where(jnp.arange(6) == action[:, None], 100.0, -100.0)[None]
        return hidden + weights["increment"], distrax.Categorical(logits=logits), None


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            "ENV_NAME": "overcooked_v2",
            "ENV_KWARGS": {"layout": "coord_ring", "max_steps": 3,
                           "agent_view_size": 2, "random_agent_positions": True},
            "ACTIVATION": "relu", "FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 2,
            "SEED": 10,
        }
        env = jaxmarl.make(cls.config["ENV_NAME"], **cls.config["ENV_KWARGS"])
        module = SimpleNamespace(
            distrax=distrax,
            ScannedRNN=SimpleNamespace(initialize_carry=lambda batch, size: jnp.zeros((batch, size))),
        )
        cls.play = staticmethod(make_episode(module, cls.config, env, MemoryPolicy()))
        cls.a = {"action": jnp.array(4), "increment": jnp.array(1.0)}
        cls.b = {"action": jnp.array(5), "increment": jnp.array(3.0)}

    def assert_same_trace(self, left, right):
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_same_weights_preserve_shared_playback(self):
        self.assert_same_trace(self.play(self.a, 1000), self.play(self.a, 1000, self.a))

    def test_assignment_and_separate_memories(self):
        _, trace = self.play(self.a, 1000, self.b)
        np.testing.assert_array_equal(np.asarray(trace[1]), [[4, 5], [4, 3], [3, 3]])
        _, reverse = self.play(self.b, 1000, self.a)
        np.testing.assert_array_equal(np.asarray(reverse[1]), [[5, 4], [3, 4], [3, 3]])
        self.assertTrue(np.asarray(trace[-1]).all())
        self.assertTrue(bool(trace[0].done[-1]))

    def test_memory_resets_and_seed_reproduces(self):
        first = self.play(self.a, 1000, self.b)
        self.play(self.a, 1001, self.b)
        self.assert_same_trace(first, self.play(self.a, 1000, self.b))

    def test_compatibility_allows_independent_training(self):
        other = copy.deepcopy(self.config)
        other.update(SEED=20, LR=0.0001)
        check_partner_config(self.config, other)
        other["ENV_KWARGS"]["agent_view_size"] = 3
        with self.assertRaisesRegex(ValueError, "ENV_KWARGS"):
            check_partner_config(self.config, other)
        other = copy.deepcopy(self.config)
        other["GRU_HIDDEN_DIM"] = 4
        with self.assertRaisesRegex(ValueError, "GRU_HIDDEN_DIM"):
            check_partner_config(self.config, other)

    def test_compatibility_checks_algorithm_and_accepts_old_configs(self):
        other = copy.deepcopy(self.config)
        other["ALGORITHM"] = "ippo_rnn"
        check_partner_config(self.config, other)
        other["ALGORITHM"] = "different"
        with self.assertRaisesRegex(ValueError, "ALGORITHM"):
            check_partner_config(self.config, other)

    def test_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            template = {"weights": np.zeros(2, dtype=np.float32)}
            for bad in (np.zeros(3, dtype=np.float32), np.array([0, np.nan], dtype=np.float32)):
                (run / "params.msgpack").write_bytes(serialization.to_bytes({"weights": bad}))
                with self.assertRaisesRegex(ValueError, "shapes or non-finite"):
                    load_checkpoint(run, template)
            (run / "params.msgpack").write_bytes(serialization.to_bytes(template))
            np.testing.assert_array_equal(load_checkpoint(run, template)["weights"], template["weights"])
            (run / "metadata.json").write_text(json.dumps({"upstream_revision": "one"}))
            check_run_revision(run, "one")
            with self.assertRaisesRegex(ValueError, "revisions differ"):
                check_run_revision(run, "two")


if __name__ == "__main__":
    unittest.main()
