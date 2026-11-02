"""Adapt logging to the existing IPPO trainer without adding reward shaping."""
import jax.numpy as jnp
from jaxmarl.wrappers.baselines import LogWrapper


class SushiLogWrapper(LogWrapper):
    def __init__(self, env):
        # Native diagnostics include scalars; PPO expects per-agent logging arrays.
        super().__init__(env, replace_info=True)

    def step(self, key, state, actions):
        obs, state, rewards, dones, info = super().step(key, state, actions)
        info["shaped_reward"] = {agent: jnp.float32(0) for agent in self.agents}
        return obs, state, rewards, dones, info
