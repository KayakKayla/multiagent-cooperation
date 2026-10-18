"""Run with the simulator dependencies: python -m unittest tests.test_checkpoint_resume

Uses a tiny budget (8 PPO updates, 2 kitchens) so it finishes on a CPU.
Checks, in order:
  1. the forked algorithm file reproduces upstream's make_train exactly;
  2. a saved runner state reloads bit-for-bit;
  3. training 4 updates, checkpointing, restoring, then training 4 more
     matches 8 uninterrupted updates, including the metrics;
  4. a checkpoint refuses to resume under a different config.
"""

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax
import numpy as np
import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import checkpoint  # noqa: E402

UPSTREAM = ROOT / "JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v2.py"
FORK = ROOT / "algorithms/ippo_rnn.py"
CONFIG = ROOT / "configs/coord_ring.yaml"
TOTAL_UPDATES = 8


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config():
    config = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    config.update(
        NUM_ENVS=2, NUM_STEPS=8, NUM_MINIBATCHES=2, UPDATE_EPOCHS=1,
        LR_WARMUP=0.25, REW_SHAPING_HORIZON=64,
        TOTAL_TIMESTEPS=2 * 8 * TOTAL_UPDATES,
        ALGORITHM="ippo_rnn",
    )
    return config


def rows(n):
    """Fake per-update metrics for a checkpoint taken after n updates."""
    return [{"x": np.zeros(n)}]


def assert_trees_equal(test, a, b, what):
    # Compare leaves, not tree structure: the structures embed the optimizer's
    # function objects, which differ between separately loaded modules.
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    test.assertEqual(len(leaves_a), len(leaves_b), f"{what}: leaf count differs")
    for i, (x, y) in enumerate(zip(leaves_a, leaves_b)):
        x, y = np.asarray(x), np.asarray(y)
        test.assertEqual(x.dtype, y.dtype, f"{what}: leaf {i} dtype")
        np.testing.assert_array_equal(x, y, err_msg=f"{what}: leaf {i}")


class CheckpointResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wandb.init(mode="disabled", project="test")
        cls.config = tiny_config()
        cls.key = jax.random.split(jax.random.PRNGKey(cls.config["SEED"]), 1)[0]
        cls.upstream = load_module("upstream_ippo", UPSTREAM)
        cls.fork = load_module("fork_ippo", FORK)

        # Reference: upstream's unmodified train(), all updates in one scan.
        cls.reference = jax.jit(cls.upstream.make_train(dict(cls.config)))(cls.key)

        # Fork: init once, then run chunks.
        # Held in a namespace: plain functions stored on the class would be
        # bound as methods, and `run` would shadow TestCase.run.
        init, run = cls.fork.make_trainer(dict(cls.config))
        cls.trainer = SimpleNamespace(init=init, run=run)
        cls.init_state = jax.jit(init)(cls.key)
        cls.full_state, cls.full_metrics = jax.jit(
            lambda s: run(s, TOTAL_UPDATES))(cls.init_state)

    @classmethod
    def tearDownClass(cls):
        wandb.finish()

    def test_fork_matches_upstream(self):
        assert_trees_equal(
            self, self.full_state, self.reference["runner_state"], "final state")
        assert_trees_equal(
            self, self.full_metrics, self.reference["metrics"], "metrics")

    def test_comparisons_are_not_vacuous(self):
        # Equality checks mean little if training left the state unchanged.
        before = jax.tree.leaves(self.init_state[0].params)
        after = jax.tree.leaves(self.full_state[0].params)
        self.assertTrue(any(
            not np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(before, after)))
        self.assertEqual(int(self.init_state[4]), 0)
        self.assertEqual(int(self.full_state[4]), TOTAL_UPDATES)

    def test_state_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = checkpoint.save_checkpoint(
                tmp, self.init_state, [], 0, self.config, FORK)
            restored, metrics, update = checkpoint.load_checkpoint(
                path, self.init_state, self.config, FORK)
        self.assertEqual(update, 0)
        self.assertEqual(metrics, {})
        assert_trees_equal(self, restored, self.init_state, "restored state")

    def test_resume_matches_uninterrupted(self):
        half = TOTAL_UPDATES // 2
        run_half = jax.jit(lambda s: self.trainer.run(s, half))
        state, first = run_half(self.init_state)
        with TemporaryDirectory() as tmp:
            path = checkpoint.save_checkpoint(
                tmp, state, [first], half, self.config, FORK)
            restored, saved_metrics, update = checkpoint.load_checkpoint(
                path, self.init_state, self.config, FORK)
        self.assertEqual(update, half)
        state, second = run_half(restored)
        assert_trees_equal(self, state, self.full_state, "resumed final state")

        joined = checkpoint.concat_metrics([saved_metrics, second])
        expected = checkpoint.concat_metrics([self.full_metrics])
        self.assertEqual(joined.keys(), expected.keys())
        for name in expected:
            np.testing.assert_array_equal(joined[name], expected[name], err_msg=name)

    def test_resume_refuses_changed_config(self):
        changed = dict(self.config, LR=self.config["LR"] * 2)
        with TemporaryDirectory() as tmp:
            path = checkpoint.save_checkpoint(
                tmp, self.init_state, [], 0, self.config, FORK)
            with self.assertRaises(ValueError):
                checkpoint.load_checkpoint(path, self.init_state, changed, FORK)
            # Interval/eval keys are allowed to differ.
            relaxed = dict(self.config, CHECKPOINT_INTERVAL_UPDATES=99)
            checkpoint.load_checkpoint(path, self.init_state, relaxed, FORK)

    def test_retention_and_atomic_names(self):
        with TemporaryDirectory() as tmp:
            for n in (2, 4, 6):
                checkpoint.save_checkpoint(
                    tmp, self.init_state, rows(n), n, self.config, FORK, keep_last=2)
            names = [p.name for p in checkpoint.list_checkpoints(tmp)]
            self.assertEqual(names, ["update-000004", "update-000006"])
            self.assertFalse(any(
                p.name.endswith(".tmp")
                for p in (Path(tmp) / checkpoint.CHECKPOINT_DIRNAME).iterdir()))
            self.assertEqual(
                checkpoint.latest_checkpoint(tmp).name, "update-000006")

    def test_concat_metrics_skips_empty_parts(self):
        chunk = {"a": np.arange(3), "nested": {"b": np.ones((3, 2))}}
        joined = checkpoint.concat_metrics([{}, chunk, None, chunk])
        self.assertEqual(sorted(joined), ["a", "nested/b"])
        self.assertEqual(joined["a"].shape, (6,))
        self.assertEqual(joined["nested/b"].shape, (6, 2))
        self.assertEqual(checkpoint.concat_metrics([{}]), {})

    def test_metrics_must_cover_every_update(self):
        with TemporaryDirectory() as tmp:
            # Saving: empty or short metrics are refused, and nothing is written.
            for bad in ([], rows(3)):
                with self.assertRaises(ValueError):
                    checkpoint.save_checkpoint(
                        tmp, self.init_state, bad, 4, self.config, FORK)
            self.assertEqual(checkpoint.list_checkpoints(tmp), [])
            self.assertFalse((Path(tmp) / checkpoint.CHECKPOINT_DIRNAME).exists())

            # Loading: a checkpoint written by the old buggy version (state
            # present, metrics empty) gets a clear error, not a late KeyError.
            good = checkpoint.save_checkpoint(
                tmp, self.init_state, rows(4), 4, self.config, FORK)
            np.savez_compressed(good / checkpoint.METRICS_FILE)   # empty, like the bug
            with self.assertRaisesRegex(ValueError, "do not cover all 4 updates"):
                checkpoint.load_checkpoint(good, self.init_state, self.config, FORK)

    def test_incomplete_checkpoints_are_ignored(self):
        # A crash mid-save leaves a .tmp folder, or a folder with no meta.json.
        with TemporaryDirectory() as tmp:
            for n in (4, 8):
                checkpoint.save_checkpoint(
                    tmp, self.init_state, rows(n), n, self.config, FORK)
            root = Path(tmp) / checkpoint.CHECKPOINT_DIRNAME
            (root / "update-000012.tmp").mkdir()
            (root / "update-000012.tmp" / checkpoint.STATE_FILE).write_bytes(b"partial")
            (root / "update-000016").mkdir()          # no meta.json written yet
            (root / "update-000016" / checkpoint.STATE_FILE).write_bytes(b"partial")
            names = [p.name for p in checkpoint.list_checkpoints(tmp)]
            self.assertEqual(names, ["update-000004", "update-000008"])
            self.assertEqual(checkpoint.latest_checkpoint(tmp).name, "update-000008")
            # And the latest complete one still loads.
            _, _, update = checkpoint.load_checkpoint(
                checkpoint.latest_checkpoint(tmp), self.init_state, self.config, FORK)
            self.assertEqual(update, 8)


if __name__ == "__main__":
    unittest.main()
