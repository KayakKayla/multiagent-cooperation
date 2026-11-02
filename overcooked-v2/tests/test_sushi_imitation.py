"""Fast CPU tests for demonstration packing and the imitation objective.

Toy recurrent policies make time/episode/chef ordering independently observable,
without training the large CNN or launching a GPU job.
"""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import checkpoint as ckpt
import sushi_imitation as imitation


class _HistoryPolicy:
    """Predict the modulo-seven cumulative sum for each chef independently."""
    def apply(self, params, hidden, inputs):
        observations, resets = inputs
        values = observations[..., 0, 0, 0]

        def step(carry, pair):
            value, reset = pair
            after = carry * (1 - reset[:, None]) + value[:, None]
            return after, after[:, 0]

        hidden, accumulated = jax.lax.scan(step, hidden, (values, resets))
        targets = jnp.mod(accumulated.astype(jnp.int32), 7)
        logits = jax.nn.one_hot(targets, 7) * params['scale']
        return hidden, SimpleNamespace(logits_parameter=lambda: logits), jnp.zeros_like(accumulated)


class _ConstantPolicy:
    def apply(self, params, hidden, inputs):
        observations, _ = inputs
        logits = jnp.broadcast_to(params['logits'], (*observations.shape[:2], 7))
        return hidden, SimpleNamespace(logits_parameter=lambda: logits), jnp.zeros(observations.shape[:2])


def _history_data():
    # Each episode and chef has a different history. Flattening either axis into
    # time, or swapping episode and chef batches, changes the correct labels.
    values = np.array([[[1, 2], [2, 1], [1, 3], [3, 1]],
                       [[3, 1], [1, 1], [2, 2], [1, 3]]], dtype=np.float32)
    observations = values[..., None, None, None]
    labels = np.cumsum(values, axis=1).astype(np.int32) % 7
    return jnp.asarray(observations), jnp.asarray(labels), jnp.ones_like(jnp.asarray(labels), dtype=jnp.float32)


class SushiImitationTests(unittest.TestCase):
    def test_packing_preserves_time_chefs_masks_and_padding(self):
        observations = np.zeros((3, 2, 1, 1, 48), dtype=np.float32)
        observations[:, :, 0, 0, 0] = [[1, 2], [3, 4], [5, 6]]
        actions = np.array([[0, 1], [2, 3], [5, 6]], dtype=np.int32)
        mask = np.array([[1, 1], [0, 0], [1, 1]], dtype=np.float32)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'demo.npz'
            np.savez(path, observations=observations, actions=actions, supervision_mask=mask)
            packed = imitation.load_demonstrations(path, padded_steps=5)
            self.assertEqual(packed['observations'].shape, (1, 5, 2, 1, 1, 48))
            np.testing.assert_array_equal(packed['observations'][0, :3], observations)
            np.testing.assert_array_equal(packed['actions'][0, :3], actions)
            np.testing.assert_array_equal(packed['mask'][0, :3], mask)
            np.testing.assert_array_equal(packed['mask'][0, 3:], 0)
            self.assertEqual(packed['lengths'], [3])
            self.assertEqual(len(packed['hashes']['demo.npz']), 64)
            with self.assertRaises(ValueError):
                imitation.load_demonstrations(path, padded_steps=2)

    def test_full_trajectory_objective_preserves_every_chef_history(self):
        network = _HistoryPolicy()
        _, loss = imitation.make_updates(network, {'GRU_HIDDEN_DIM': 1}, jnp.ones(7))
        obs, labels, mask = _history_data()
        value, metrics = loss({'scale': jnp.float32(8)}, obs, labels, mask)
        self.assertEqual(float(metrics['accuracy']), 1.)
        self.assertLess(float(value), .003)
        # Labels from the wrong temporal ordering should decisively fail.
        _, wrong = loss({'scale': jnp.float32(8)}, obs, jnp.flip(labels, axis=1), mask)
        self.assertLess(float(wrong['accuracy']), .5)

    def test_unsupervised_waits_preserve_history_and_padding_has_no_loss(self):
        network = _HistoryPolicy()
        _, loss = imitation.make_updates(network, {'GRU_HIDDEN_DIM': 1}, jnp.ones(7))
        obs, labels, mask = _history_data()
        # Keep the middle observation in recurrent history, but ignore its label.
        mask = mask.at[:, 1].set(0).at[:, -1].set(0)
        expected, metrics = loss({'scale': jnp.float32(8)}, obs, labels, mask)
        self.assertEqual(float(metrics['accuracy']), 1.)
        arbitrary = labels.at[:, 1].set((labels[:, 1] + 3) % 7).at[:, -1].set(6)
        actual, _ = loss({'scale': jnp.float32(8)}, obs, arbitrary, mask)
        np.testing.assert_array_equal(actual, expected)
        # Altering only trailing padded observations cannot affect earlier loss.
        actual, _ = loss({'scale': jnp.float32(8)}, obs.at[:, -1].set(99), arbitrary, mask)
        np.testing.assert_array_equal(actual, expected)
        # Removing the unsupervised middle input would corrupt later predictions.
        _, dropped = loss({'scale': jnp.float32(8)}, obs.at[:, 1].set(0), labels, mask)
        self.assertLess(float(dropped['accuracy']), 1.)

    def test_class_weighted_loss_uses_weighted_denominator_and_masked_accuracy(self):
        network = _ConstantPolicy()
        weights = jnp.array([1., 3., 1., 1., 1., 1., 1.])
        _, loss = imitation.make_updates(network, {'GRU_HIDDEN_DIM': 1}, weights)
        obs = jnp.zeros((1, 2, 2, 1, 1, 1))
        labels = jnp.array([[[0, 1], [2, 3]]])
        mask = jnp.array([[[1., 1.], [0., 0.]]])
        logits = jnp.array([2., 0., 0., 0., 0., 0., 0.])
        value, metrics = loss({'logits': logits}, obs, labels, mask)
        nll = -jax.nn.log_softmax(logits)
        np.testing.assert_allclose(value, (nll[0] + 3 * nll[1]) / 4, rtol=1e-6)
        self.assertEqual(float(metrics['accuracy']), .5)
        np.testing.assert_allclose(metrics['unweighted_nll'], (nll[0] + nll[1]) / 2, rtol=1e-6)

    def test_saved_optimizer_and_key_resume_exactly(self):
        network = _HistoryPolicy()
        update, loss = imitation.make_updates(network, {'GRU_HIDDEN_DIM': 1}, jnp.ones(7))
        obs, labels, mask = _history_data()
        initial = TrainState.create(apply_fn=network.apply, params={'scale': jnp.float32(2)}, tx=optax.adam(.01))
        state, metrics1 = update(initial, obs, labels, mask)
        state, metrics2 = update(state, obs, labels, mask)
        before_loss = float(loss(initial.params, obs, labels, mask)[0])
        self.assertLess(float(loss(state.params, obs, labels, mask)[0]), before_loss)
        key = jax.random.PRNGKey(19)
        history = [{k: float(v) for k, v in metrics.items()} for metrics in (metrics1, metrics2)]
        config = {'ALGORITHM': 'ippo_rnn', 'IMITATION': {'purpose': 'unit test'}}
        with TemporaryDirectory() as folder:
            output = Path(folder)
            path = imitation.save_artifacts(output, state, key, history, config)
            (restored, restored_key), metrics, done = ckpt.load_checkpoint(
                path, (initial, jax.random.PRNGKey(0)), config, Path(imitation.__file__))
            self.assertEqual(done, 2)
            self.assertEqual(len(metrics['loss']), 2)
            np.testing.assert_array_equal(restored_key, key)
            saved_params = serialization.from_bytes(initial.params, (output / 'params.msgpack').read_bytes())
            np.testing.assert_array_equal(saved_params['scale'], state.params['scale'])
            resumed, resumed_metrics = update(restored, obs, labels, mask)
            uninterrupted, full_metrics = update(state, obs, labels, mask)
            for a, b in zip(jax.tree.leaves((resumed, resumed_metrics)), jax.tree.leaves((uninterrupted, full_metrics))):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
