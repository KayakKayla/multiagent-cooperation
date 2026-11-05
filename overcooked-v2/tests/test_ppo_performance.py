"""Correctness gates for opt-in PPO performance experiments."""
import copy
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.kernels import conditional_log_step
from benchmarks.variants import load_variant
from scenarios import make_env,wrap_training
from train import load_config,load_algorithm


class PerformanceCorrectnessTests(unittest.TestCase):
    def test_cuda_config_preserves_training_and_environment(self):
        baseline=load_config(ROOT/'configs/coord_ring.yaml',42)
        base_runtime=baseline.pop('RUNTIME')
        for filename,commands in [('coord_ring_fusion.yaml','FUSION'),
                                  ('coord_ring_cuda_graphs.yaml','FUSION,CUBLAS,CUSTOM_CALL,CUDNN')]:
            candidate=load_config(ROOT/'configs'/filename,42)
            candidate_runtime=candidate.pop('RUNTIME')
            self.assertEqual(baseline,candidate)
            self.assertEqual(candidate_runtime['XLA_FLAGS'],base_runtime['XLA_FLAGS']+
                             ['--xla_gpu_enable_command_buffer='+commands])

    def assert_tree_equal(self, a, b):
        left,right=jax.tree.leaves(jax.device_get(a)),jax.tree.leaves(jax.device_get(b))
        self.assertEqual(len(left),len(right))
        for i,(x,y) in enumerate(zip(left,right)):
            self.assertEqual(x.dtype,y.dtype,f'leaf {i}')
            np.testing.assert_array_equal(x,y,err_msg=f'leaf {i}')

    def test_conditional_reset_matches_mixed_terminal_batch(self):
        config=load_config(ROOT/'configs/coord_ring.yaml',42)
        env=wrap_training(make_env(env_name=config['ENV_NAME'],**config['ENV_KWARGS']))
        keys=jax.random.split(jax.random.PRNGKey(42),8)
        _,state=jax.jit(jax.vmap(env.reset))(keys)
        state=state.replace(env_state=state.env_state.replace(step=jnp.array([0,399,398,399,100,50,1,398])),
                            episode_returns=jnp.arange(16,dtype=jnp.float32).reshape(8,2))
        acts={a:jax.random.randint(jax.random.fold_in(keys[0],i),(8,),0,6) for i,a in enumerate(env.agents)}
        original=jax.jit(jax.vmap(env.step))
        candidate=jax.jit(lambda k,s,a:conditional_log_step(env,k,s,a))
        for _ in range(5):
            expected=original(keys,state,acts)
            actual=candidate(keys,state,acts)
            self.assert_tree_equal(actual,expected)
            _,state,*_=expected
            keys=jax.vmap(lambda k:jax.random.split(k)[0])(keys)

    @jax.default_matmul_precision('float32')
    def test_fused_gru_forward_and_gradient_match(self):
        config=load_config(ROOT/'configs/coord_ring.yaml',42)
        base=load_algorithm('ippo_rnn')
        variant=load_variant({'fused_gru':True})
        original=base.ActorCriticRNN(6,config)
        candidate=variant.ActorCriticRNN(6,config)
        obs=jax.random.normal(jax.random.PRNGKey(3),(12,4,5,5,30))
        resets=jnp.zeros((12,4),dtype=jnp.bool_).at[0].set(True).at[5,1].set(True).at[9,2].set(True)
        hidden=jax.random.normal(jax.random.PRNGKey(4),(4,128))
        params=original.init(jax.random.PRNGKey(5),hidden,(obs,resets))
        def loss(network,p):
            carry,policy,value=network.apply(p,hidden,(obs,resets))
            return jnp.mean(carry**2)+jnp.mean(policy.logits**2)+jnp.mean(value**2)
        for x,y in zip(jax.tree.leaves(jax.jit(lambda p:original.apply(p,hidden,(obs,resets)))(params)),
                       jax.tree.leaves(jax.jit(lambda p:candidate.apply(p,hidden,(obs,resets)))(params))):
            np.testing.assert_allclose(x,y,rtol=1e-4,atol=2e-5)
        a=jax.jit(jax.grad(lambda p:loss(original,p)))(params)
        b=jax.jit(jax.grad(lambda p:loss(candidate,p)))(params)
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(x,y,rtol=5e-4,atol=3e-5)

    def test_compact_observations_are_lossless_for_current_schema(self):
        from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME
        # Encoded ingredient counts <=3, binary channels <=1, pot timers <=20.
        self.assertLessEqual(POT_COOK_TIME,255)
        config=load_config(ROOT/'configs/coord_ring.yaml',42)
        env=make_env(env_name=config['ENV_NAME'],**config['ENV_KWARGS'])
        obs,state=env.reset(jax.random.PRNGKey(42))
        grid=state.grid.at[:,:,2].set(POT_COOK_TIME)
        obs=env.get_obs(state.replace(grid=grid))
        for x in obs.values():
            np.testing.assert_array_equal(x,x.astype(jnp.uint8).astype(x.dtype))


if __name__=='__main__':unittest.main()
