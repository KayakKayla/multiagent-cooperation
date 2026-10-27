"""Optional real-GPU checks: JAX_PLATFORMS=METAL,cpu python -m unittest
tests.test_metal_compatibility. Requires the isolated Metal/PPO dependencies.
"""
import copy
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "JaxMARL")]
import checkpoint as ckpt
from train import ALGORITHMS, load_baseline


@unittest.skipUnless("metal" in os.environ.get("JAX_PLATFORMS", "").lower(),
                     "explicit Metal test environment required")
class MetalCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert jax.default_backend().lower() == "metal", "CPU fallback is not a GPU test"
        cls.gpu, cls.cpu = jax.devices("METAL")[0], jax.devices("cpu")[0]
        module, config = load_baseline(ROOT / "configs/oc2_sushi_check.yaml", None)
        # Exercise a reset inside each rollout, as well as at its final step.
        config["ENV_KWARGS"]["rules"]["max_steps"] = 4
        logger = patch.object(module.wandb, "log")
        logger.start()
        cls.addClassCleanup(logger.stop)
        cpu_config = copy.deepcopy(config)
        config["RUNTIME"] = {"METAL_COMPATIBILITY": True}
        cls.config = config
        init, updates = module.make_trainer(config)
        _, reference_updates = module.make_trainer(cpu_config)
        key = jax.device_put(jax.random.PRNGKey(0), cls.cpu)
        cls.initial = jax.jit(init, backend="cpu")(key)
        cls.gpu_initial = jax.device_put(cls.initial, cls.gpu)
        # Store functions on instances so unittest does not bind them as methods.
        from types import SimpleNamespace
        cls.fns = SimpleNamespace(one=jax.jit(lambda s: updates(s, 1), backend="METAL"))
        cls.first, cls.first_metrics = cls.fns.one(cls.gpu_initial)
        cls.second, cls.second_metrics = cls.fns.one(cls.first)
        cls.reference, cls.reference_metrics = jax.jit(
            lambda s: reference_updates(s, 2), backend="cpu")(cls.initial)
        for leaf in jax.tree.leaves((cls.second, cls.reference)):
            leaf.block_until_ready()
        jax.effects_barrier()

    def assert_trees_match(self, actual, expected, exact=False):
        left, right = jax.tree.leaves(jax.device_get(actual)), jax.tree.leaves(jax.device_get(expected))
        self.assertEqual(len(left), len(right))
        for i, (a, b) in enumerate(zip(left, right)):
            self.assertEqual(a.dtype, b.dtype, f"leaf {i} dtype")
            if exact or not np.issubdtype(a.dtype, np.inexact):
                np.testing.assert_array_equal(a, b, err_msg=f"leaf {i}")
            else:
                np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5, err_msg=f"leaf {i}")

    def test_gpu_updates_match_unmodified_cpu_math(self):
        self.assertTrue(all(self.gpu in x.devices() for x in jax.tree.leaves(self.second)))
        self.assert_trees_match(self.second, self.reference)
        joined = ckpt.concat_metrics([self.first_metrics, self.second_metrics])
        self.assert_trees_match(joined, ckpt.concat_metrics([self.reference_metrics]))
        before, after = jax.tree.leaves(self.initial[0].params), jax.tree.leaves(self.second[0].params)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, after)))
        self.assertEqual(int(self.second[0].step), 4)

    def test_episode_boundaries_and_true_masks_are_preserved(self):
        np.testing.assert_array_equal(self.first[3], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(self.second[3], np.ones(4, dtype=bool))
        np.testing.assert_array_equal(self.second[1].env_state.time, [0, 0])
        np.testing.assert_array_equal(self.second[1].returned_episode_lengths, [[4, 4], [4, 4]])
        np.testing.assert_allclose(self.first_metrics["returned_episode"], [1 / 4])

    def test_checkpoint_resume_matches_uninterrupted_gpu(self):
        with TemporaryDirectory() as tmp:
            path = ckpt.save_checkpoint(tmp, self.first, [self.first_metrics], 1,
                                        self.config, ALGORITHMS["ippo_rnn"])
            restored, saved, update = ckpt.load_checkpoint(
                path, self.gpu_initial, self.config, ALGORITHMS["ippo_rnn"])
            self.assertEqual(update, 1)
            self.assert_trees_match(restored, self.first, exact=True)
            resumed, metrics = self.fns.one(restored)
            self.assert_trees_match(resumed, self.second, exact=True)
            self.assert_trees_match(metrics, self.second_metrics, exact=True)
            self.assert_trees_match(saved, ckpt.concat_metrics([self.first_metrics]), exact=True)


if __name__ == "__main__":
    unittest.main()
