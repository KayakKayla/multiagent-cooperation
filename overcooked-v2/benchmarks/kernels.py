"""Experimental equivalent kernels, restricted to the pinned Coordination Ring."""
import jax
import jax.numpy as jnp
import flax.linen as nn


def conditional_log_step(wrapper, keys, state, actions):
    """Move reset outside vmap; preserve upstream key splitting and log fields."""
    from jaxmarl.wrappers.baselines import OvercookedV2LogEnvState
    env = wrapper._env
    split = jax.vmap(jax.random.split)(keys)
    obs, next_env, reward, done, info = jax.vmap(env.step_env)(split[:, 0], state.env_state, actions)
    finished = done['__all__']

    def reset_if_needed(values):
        old_obs, old_state = values
        reset_obs, reset_state = jax.vmap(env.reset)(split[:, 1])
        def select(new, old):
            mask = finished.reshape((len(finished),)+(1,)*(old.ndim-1))
            return jnp.where(mask, new, old)
        return jax.tree.map(select, reset_obs, old_obs), jax.tree.map(select, reset_state, old_state)

    obs, next_env = jax.lax.cond(jnp.any(finished), reset_if_needed, lambda x:x, (obs, next_env))
    episode_done = finished[:, None]
    returns = state.episode_returns+jnp.stack([reward[a] for a in env.agents], axis=1)
    lengths = state.episode_lengths+1
    recipe_returns = {
        recipe: jnp.where(((state.env_state.recipe == encoding)&finished)[:,None],
                          returns, state.returned_episode_recipe_returns[recipe])
        for recipe, encoding in wrapper.recipe_dict.items()
    }
    logged = OvercookedV2LogEnvState(next_env, returns*(1-episode_done), lengths*(1-episode_done),
        jnp.where(episode_done,returns,state.returned_episode_returns),
        jnp.where(episode_done,lengths,state.returned_episode_lengths), recipe_returns)
    info['returned_episode_returns'] = logged.returned_episode_returns
    info['returned_episode_lengths'] = logged.returned_episode_lengths
    info['returned_episode'] = jnp.broadcast_to(episode_done, (len(keys),env.num_agents))
    info['returned_episode_recipe_returns'] = logged.returned_episode_recipe_returns
    return obs, logged, reward, done, info


class FusedScannedRNN(nn.Module):
    """Same GRU parameters/equations with sequence-wide input projection.

    Keep parameter names compatible with the original nn.scan GRU. Initialization
    is delegated to that same cell; experiments nevertheless share baseline init.
    """
    metal_compat: bool = False
    unroll: int = 1

    @nn.compact
    def __call__(self, hidden, inputs):
        sequence, resets = inputs
        width = sequence.shape[-1]
        if self.is_initializing():
            # Original carry initialization consumes the automatic GRUCell_0 name.
            nn.GRUCell(features=width, name='GRUCell_1')(hidden,sequence[0])
        parameters = self.variables['params']['GRUCell_1']
        input_kernel = jnp.concatenate([parameters[k]['kernel'] for k in ('ir','iz','in')], axis=1)
        input_bias = jnp.concatenate([parameters[k]['bias'] for k in ('ir','iz','in')])
        hidden_kernel = jnp.concatenate([parameters[k]['kernel'] for k in ('hr','hz','hn')], axis=1)
        projected = sequence@input_kernel+input_bias
        def step(h, xs):
            x, reset = xs
            h = jnp.where(reset[:,None], jnp.zeros_like(h), h)
            ir, iz, inn = jnp.split(x,3,axis=-1)
            hr, hz, hn = jnp.split(h@hidden_kernel,3,axis=-1)
            r, z = nn.sigmoid(ir+hr), nn.sigmoid(iz+hz)
            n = jnp.tanh(inn+r*(hn+parameters['hn']['bias']))
            h = (1-z)*n+z*h
            return h,h
        return jax.lax.scan(step,hidden,(projected,resets),unroll=self.unroll)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size,hidden_size),jnp.float32)
