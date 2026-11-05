"""Check opt-in recurrent unrolling and full-state checkpoint compatibility."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import checkpoint
from train import load_algorithm, load_config
from tests.test_checkpoint import assert_trees_equal, tiny_config


class GruUnrollTests(unittest.TestCase):
    @jax.default_matmul_precision('float32')
    def test_shared_weights_outputs_and_gradients_across_resets_and_tail(self):
        module = load_algorithm('ippo_rnn')
        original = module.ScannedRNN()
        unrolled = module.ScannedRNN(unroll=16)
        x = jax.random.normal(jax.random.PRNGKey(1), (19, 3, 8))
        resets = jnp.zeros((19, 3), dtype=bool).at[0].set(True).at[5, 1].set(True).at[17, 2].set(True)
        hidden = jnp.ones((3, 8))
        params = original.init(jax.random.PRNGKey(2), hidden, (x, resets))
        candidate_params = unrolled.init(jax.random.PRNGKey(2), hidden, (x, resets))
        assert_trees_equal(self, params, candidate_params, 'initial parameters')
        def output(network, p):
            return network.apply(p, hidden, (x, resets))
        for a, b in zip(jax.tree.leaves(jax.jit(lambda p: output(original, p))(params)),
                        jax.tree.leaves(jax.jit(lambda p: output(unrolled, p))(params))):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
        def loss(network, p):
            carry, values = output(network, p)
            return jnp.mean(carry**2) + jnp.mean(values**2)
        for a, b in zip(jax.tree.leaves(jax.jit(jax.grad(lambda p: loss(original, p)))(params)),
                        jax.tree.leaves(jax.jit(jax.grad(lambda p: loss(unrolled, p)))(params))):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)

    def test_unrolled_training_resume_and_config_guard(self):
        config = tiny_config()
        config.update(GRU_UNROLL=16, NUM_STEPS=16, TOTAL_TIMESTEPS=2*16*4,
                      GRU_HIDDEN_DIM=8, FC_DIM_SIZE=8)
        module = load_algorithm('ippo_rnn')
        source = ROOT / 'algorithms/ippo_rnn.py'
        with patch.object(module.wandb, 'log', lambda metrics: None):
            init, run = module.make_trainer(config)
            initial = jax.jit(init)(jax.random.PRNGKey(42))
            advance = jax.jit(lambda state: run(state, 2))
            state, first = advance(initial)
            expected, second = advance(state)
            with TemporaryDirectory() as directory:
                saved = Path(directory) / 'config.json'
                saved.write_text(json.dumps(config))
                restored_config = load_config(saved)
                self.assertEqual(restored_config['GRU_UNROLL'], 16)
                path = checkpoint.save_checkpoint(directory, state, [first], 2, config, source)
                restored, metrics, update = checkpoint.load_checkpoint(path, initial, restored_config, source)
                changed = copy.deepcopy(config)
                changed['GRU_UNROLL'] = 1
                with self.assertRaisesRegex(ValueError, 'config|settings'):
                    checkpoint.load_checkpoint(path, initial, changed, source)
            actual, resumed_metrics = advance(restored)
            self.assertEqual(update, 2)
            assert_trees_equal(self, actual, expected, 'resumed unrolled state')
            assert_trees_equal(self, resumed_metrics, second, 'resumed metrics')
            self.assertTrue(metrics)

    def test_unroll_config_validation_and_legacy_default(self):
        config = load_config(ROOT / 'configs/coord_ring.yaml')
        self.assertEqual(config.get('GRU_UNROLL', 1), 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for value in (0, -1, True, 1.5, '16'):
                with self.subTest(value=value):
                    path.write_text(json.dumps(dict(config, GRU_UNROLL=value)))
                    with self.assertRaisesRegex(ValueError, 'GRU_UNROLL'):
                        load_config(path)


if __name__ == '__main__':
    unittest.main()
