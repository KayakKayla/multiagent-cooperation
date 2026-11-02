"""Fit only the critic head on the frozen policy's actual discounted returns.

Actor, CNN, recurrent weights and action probabilities must remain unchanged.
This prepares a behavior-cloned policy for an explicitly separate PPO trial.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from evaluate import initialize_policy
from evaluate_sushi_policy import make_episode
from scenarios import make_env
from train import load_baseline


class FrozenFeatures(nn.Module):
    config: dict
    algorithm: object

    @nn.compact
    def __call__(self, hidden, inputs):
        observations, resets = inputs
        activation = nn.relu if self.config['ACTIVATION'] == 'relu' else nn.tanh
        cnn = self.algorithm.CNN(output_size=self.config['GRU_HIDDEN_DIM'], activation=activation, name='CNN_0')
        embeddings = jax.vmap(cnn)(observations)
        embeddings = nn.LayerNorm(name='LayerNorm_0')(embeddings)
        _, features = self.algorithm.ScannedRNN(
            metal_compat=self.config.get('RUNTIME', {}).get('METAL_COMPATIBILITY', False),
            name='ScannedRNN_0')(hidden, (embeddings, resets))
        return features


def head_values(head, features):
    hidden = jax.nn.relu(features @ head['Dense_2']['kernel'] + head['Dense_2']['bias'])
    return (hidden @ head['Dense_3']['kernel'] + head['Dense_3']['bias'])[..., 0]


def discounted_returns(rewards, gamma):
    result = np.zeros_like(rewards, dtype=np.float32)
    carry = 0.
    for index in range(len(rewards)-1, -1, -1):
        carry = float(rewards[index]) + gamma * carry
        result[index] = carry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=6000)
    parser.add_argument('--lr', type=float, default=.001)
    args = parser.parse_args()
    if jax.default_backend().lower() != 'cpu':
        raise ValueError('Critic preparation deliberately uses CPU')
    module, config = load_baseline(args.run/'config.json', None)
    env = make_env(**config['ENV_KWARGS'])
    network, template = initialize_policy(module, config, env)
    original_bytes = (args.run/'params.msgpack').read_bytes()
    params = serialization.from_bytes(template, original_bytes)
    extractor = FrozenFeatures(config, module)
    extract = jax.jit(extractor.apply)
    play = make_episode(module, config, env, network)
    features, targets, dish_counts = [], [], []
    actor_checks = []
    for episode in range(args.episodes):
        initial, trace, _ = play(params, np.uint32(args.seed+episode))
        states, actions, rewards, deliveries, _, _ = trace
        before = jax.tree.map(lambda first, after: jnp.concatenate([first[None], after[:-1]], axis=0), initial, states)
        obs = jax.jit(jax.vmap(env.get_obs))(before)
        obs = jnp.stack([obs[a] for a in env.agents], axis=1)
        hidden = jnp.zeros((env.num_agents, config['GRU_HIDDEN_DIM']), jnp.float32)
        reset = jnp.zeros((env.max_steps, env.num_agents), jnp.int32).at[0].set(1)
        x = extract(params, hidden, (obs, reset))
        head = {name: params['params'][name] for name in ['Dense_2','Dense_3']}
        _, policy, predicted = jax.jit(network.apply)(params, hidden, (obs, reset))
        np.testing.assert_allclose(head_values(head, x), predicted, rtol=1e-5, atol=1e-5)
        features.append(np.asarray(x).reshape(-1, config['GRU_HIDDEN_DIM']))
        returns = discounted_returns(np.asarray(rewards), config.get('GAMMA',.99))
        targets.append(np.repeat(returns[:,None], env.num_agents, axis=1).reshape(-1))
        dish_counts.append(int(np.asarray(deliveries).sum()))
        if episode == 0:
            actor_checks.append((hidden, obs, reset, np.asarray(policy.probs)))
    x, y = jnp.asarray(np.concatenate(features)), jnp.asarray(np.concatenate(targets))
    head = {name: params['params'][name] for name in ['Dense_2','Dense_3']}
    tx = optax.adam(args.lr); optimizer = tx.init(head)
    def loss(h, x, y):return jnp.mean(jnp.square(head_values(h,x)-y))
    @jax.jit
    def update(h, o, x, y):
        value, gradient = jax.value_and_grad(loss)(h,x,y)
        updates,o = tx.update(gradient,o,h)
        return optax.apply_updates(h,updates),o,value
    before_mse = float(loss(head,x,y))
    rng = np.random.default_rng(args.seed)
    for step in range(args.updates):
        ids = rng.integers(0,len(y),size=min(1024,len(y)))
        head,optimizer,value = update(head,optimizer,x[ids],y[ids])
        if (step+1)%500==0:print('Critic update',step+1,'MSE',float(value),flush=True)
    updated = copy.deepcopy(params)
    for name in head:updated['params'][name] = head[name]
    for name in params['params']:
        if name not in head:
            for a,b in zip(jax.tree.leaves(params['params'][name]),jax.tree.leaves(updated['params'][name])):
                np.testing.assert_array_equal(a,b)
    for hidden,obs,reset,probs in actor_checks:
        _,policy,_ = jax.jit(network.apply)(updated,hidden,(obs,reset))
        np.testing.assert_array_equal(np.asarray(policy.probs),probs)
    if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(updated)):
        raise ValueError('Non-finite prepared weights')
    args.output.mkdir(parents=True,exist_ok=False)
    config['TRAINING_METHOD']='critic_head_only_preparation'
    config['VALUE_WARMUP']={'parent_sha256':hashlib.sha256(original_bytes).hexdigest(),
                           'episode_seeds':list(range(args.seed,args.seed+args.episodes)),
                           'updates':args.updates,'lr':args.lr,'gamma':config.get('GAMMA',.99)}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'params.msgpack').write_bytes(serialization.to_bytes(jax.device_get(updated)))
    result={'parent_policy':str(args.run),'self_rollout_dish_counts':dish_counts,
            'critic_mse_before':before_mse,'critic_mse_after':float(loss(head,x,y)),
            'all_noncritic_parameters_identical':True,'checked_action_probabilities_bit_identical':True,
            'critic_updates':args.updates,'method':'critic head only; no actor learning'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
