"""A policy warm start changes model weights, never optimizer/env/RNG state."""
from pathlib import Path
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import warm_start_runner

class PolicyWarmStartTests(unittest.TestCase):
    def initial(self):
        state=TrainState.create(apply_fn=lambda *_:None,params={'params':{'w':jnp.zeros((2,3))}},tx=optax.adam(.001))
        return (state,{'kitchen':jnp.array([1,2])},jax.random.PRNGKey(3))
    def test_weights_replace_only_params(self):
        before=self.initial();params={'params':{'w':np.ones((2,3),np.float32)}}
        after=warm_start_runner(before,serialization.to_bytes(params))
        np.testing.assert_array_equal(after[0].params['params']['w'],1)
        for a,b in zip(jax.tree.leaves((before[0].opt_state,*before[1:])),jax.tree.leaves((after[0].opt_state,*after[1:]))):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(int(after[0].step),0)
    def test_rejects_wrong_dimensions_and_nonfinite_weights(self):
        for value in [np.ones((3,2),np.float32),np.full((2,3),np.nan,np.float32)]:
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):
                warm_start_runner(self.initial(),serialization.to_bytes({'params':{'w':value}}))
if __name__=='__main__':unittest.main()
