"""Experimental one-step counterfactual critic diagnostics for recorded episodes.

This module never selects actions or trains anything.  It replays the recorded
simulator transitions on immutable copies to ask what the shared critic predicts
for the other chef after one chef deviates for a single step.  The recorded
rollout, its random-number stream, and the policy's recurrent state are not
modified.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked_v2.common import OvercookedActionsEnum


REWARD_DESCRIPTION = "raw environment reward; no shaping"
VALUE_WARNING = (
    "Experimental raw value-head differences, not a return or cooperation "
    "metric. The critic may have been trained with shaping bonuses or may be "
    "untrained, and the terminal value is diagnostic only: it has no "
    "future-return interpretation and is never bootstrapped."
)


def action_names():
    """Simulator action codes in integer order (0..5)."""
    return [action.name for action in OvercookedActionsEnum]


def candidate_action_grid(actions, num_agents, num_actions):
    """Build [actor, candidate action, agent] joint actions.

    Only ``actions[actor]`` changes for an ``actor`` row; the teammate's action
    stays fixed at its recorded value.
    """
    actions = jnp.asarray(actions)
    actor = jnp.arange(num_agents)
    candidates = jnp.broadcast_to(
        actions[None, None, :], (num_agents, num_actions, num_agents)
    )
    return candidates.at[actor, :, actor].set(
        jnp.broadcast_to(jnp.arange(num_actions)[None, :], (num_agents, num_actions))
    )


def _max_tree_difference(left, right):
    """Largest numeric/bool difference across two equally structured trees."""
    differences = []
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if first is None and second is None:
            continue
        first = jnp.asarray(first)
        second = jnp.asarray(second)
        if jnp.issubdtype(first.dtype, jnp.floating):
            difference = jnp.max(jnp.abs(first - second))
        else:
            difference = jnp.max((first != second).astype(jnp.float32))
        differences.append(difference)
    if not differences:
        return jnp.array(0.0)
    return jnp.max(jnp.stack(differences))


def _tree_slice(tree, index):
    """Slice every array leaf, preserving None fields."""
    return jax.tree.map(
        lambda leaf: None if leaf is None else leaf[index], tree
    )


def _tree_map_arrays(tree, function):
    """Map over array leaves, preserving None fields."""
    return jax.tree.map(
        lambda leaf: None if leaf is None else function(leaf), tree
    )


def make_probe_runner(env, network):
    """Return a jitted function that evaluates every one-step intervention.

    Expected episode inputs are the recorded state/observation/value timelines
    (length T+1), recurrent memory before each observation, and the T actions,
    raw rewards, terminal flags, and simulator keys used by the factual rollout.
    """
    if env.num_agents != 2:
        raise ValueError("Critic probes currently assume exactly two chefs.")
    num_agents = env.num_agents
    num_actions = env.action_space(env.agents[0]).n
    batch = num_agents * num_actions
    actors = jnp.repeat(jnp.arange(num_agents), num_actions)
    targets = 1 - actors

    @jax.jit
    def run(
        params,
        states,
        observations,
        hidden,
        values,
        actions,
        rewards,
        dones,
        step_keys,
    ):
        def step(_, inputs):
            (
                state,
                hidden_after,
                action,
                reward,
                done,
                step_key,
                next_state_recorded,
                next_observation_recorded,
                next_value_recorded,
            ) = inputs

            # Every row is one intervention: one actor's candidate action and
            # the teammate's recorded action.  The same simulator key as the
            # factual transition is reused, so no extra randomness is consumed.
            candidates = candidate_action_grid(action, num_agents, num_actions)
            flat_actions = candidates.reshape(batch, num_agents)
            flat_states = _tree_map_arrays(
                state,
                lambda leaf: jnp.broadcast_to(leaf, (batch,) + leaf.shape),
            )
            flat_keys = jnp.broadcast_to(step_key[None, :], (batch, 2))
            flat_action_dict = {
                f"agent_{i}": flat_actions[:, i] for i in range(num_agents)
            }
            next_observations, next_states, next_rewards, _, _ = jax.vmap(
                env.step_env, in_axes=(0, 0, 0)
            )(flat_keys, flat_states, flat_action_dict)

            next_observations = jnp.stack(
                [next_observations[agent] for agent in env.agents], axis=1
            )
            next_reward = next_rewards[env.agents[0]]
            next_done = next_states.done

            # The hidden state after processing the current observation is the
            # next recorded state's pre-observation memory.  It is independent
            # of the intervention; the candidate's observation and done flag
            # decide the successor value.
            flat_hidden = jnp.broadcast_to(
                hidden_after[None], (batch, num_agents, hidden_after.shape[-1])
            ).reshape(batch * num_agents, hidden_after.shape[-1])
            flat_observations = next_observations.reshape(
                (1, batch * num_agents) + next_observations.shape[2:]
            )
            flat_dones = jnp.repeat(next_done, num_agents)[None]
            _, _, next_values = network.apply(
                params, flat_hidden, (flat_observations, flat_dones)
            )
            next_values = next_values[0].reshape(batch, num_agents)
            target_values = jnp.take_along_axis(
                next_values, targets[:, None], axis=1
            )[:, 0].reshape(num_agents, num_actions)

            # The no-change candidate for each actor must reproduce the factual
            # transition.  Check the simulator result, not just our arithmetic.
            factual_next_values = jnp.take_along_axis(
                target_values, action[:, None], axis=1
            )[:, 0]
            factual_rewards = jnp.take_along_axis(
                next_reward.reshape(num_agents, num_actions), action[:, None], axis=1
            )[:, 0]
            factual_dones = jnp.take_along_axis(
                next_done.reshape(num_agents, num_actions), action[:, None], axis=1
            )[:, 0]
            recorded_target_values = next_value_recorded[1 - jnp.arange(num_agents)]
            factual_indices = jnp.arange(num_agents) * num_actions + action
            factual_states = _tree_map_arrays(
                next_states, lambda leaf: leaf[factual_indices]
            )
            factual_observations = next_observations[factual_indices]
            broadcast_recorded_observations = jnp.broadcast_to(
                next_observation_recorded[None],
                (num_agents,) + next_observation_recorded.shape,
            )
            max_state_difference = _max_tree_difference(
                factual_states, next_state_recorded
            )
            max_observation_difference = _max_tree_difference(
                factual_observations, broadcast_recorded_observations
            )
            max_value_difference = jnp.max(
                jnp.abs(factual_next_values - recorded_target_values)
            )
            factual_ok = (
                (factual_rewards == reward)
                & (factual_dones == done)
                & (max_state_difference <= 1e-3)
                & (max_observation_difference <= 1e-3)
                & (max_value_difference <= 1e-3)
            )

            return (), {
                "next_values": target_values,
                "rewards": next_reward.reshape(num_agents, num_actions),
                "dones": next_done.reshape(num_agents, num_actions),
                "factual_ok": factual_ok,
                "max_state_difference": max_state_difference,
                "max_observation_difference": max_observation_difference,
                "max_value_difference": max_value_difference,
            }

        inputs = (
            _tree_slice(states, slice(None, -1)),
            hidden[1:],
            actions,
            rewards,
            dones,
            step_keys,
            _tree_slice(states, slice(1, None)),
            observations[1:],
            values[1:],
        )
        _, result = jax.lax.scan(step, (), inputs)
        return result

    return run


def _finite(array, label):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Critic probe {label} contains non-finite values.")


def build_probe_payload(values, actions, rewards, dones, result):
    """Convert recorded arrays and intervention results into JSON-ready data."""
    values = np.asarray(values, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    next_values = np.asarray(result["next_values"], dtype=np.float64)
    candidate_rewards = np.asarray(result["rewards"], dtype=np.float64)
    candidate_dones = np.asarray(result["dones"], dtype=bool)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Critic probe values must have shape [T+1, 2].")
    num_states, num_agents = values.shape
    num_transitions = num_states - 1
    if actions.shape != (num_transitions, num_agents):
        raise ValueError("Critic probe actions must have shape [T, 2].")
    if rewards.shape != (num_transitions,) or dones.shape != (num_transitions,):
        raise ValueError("Critic probe rewards and done flags must have length T.")
    names = action_names()
    expected = (num_transitions, num_agents, len(names))
    if next_values.shape != expected:
        raise ValueError("Critic probe interventions have inconsistent shapes.")
    if candidate_rewards.shape != expected or candidate_dones.shape != expected:
        raise ValueError("Critic probe intervention events have inconsistent shapes.")
    if not np.all(result["factual_ok"]):
        raise ValueError(
            "A no-change intervention did not reproduce the factual transition; "
            "the probe would not be interpretable."
        )
    for name in (
        "max_state_difference",
        "max_observation_difference",
        "max_value_difference",
    ):
        differences = np.asarray(result[name])
        if differences.size and np.max(differences) > 1e-3:
            raise ValueError(f"Critic probe factual replay mismatch in {name}.")
    _finite(values, "values")
    _finite(rewards, "rewards")
    _finite(next_values, "successor values")
    _finite(candidate_rewards, "intervention rewards")

    steps = []
    for index in range(num_states):
        step_values = [float(value) for value in values[index]]
        if index == num_transitions:
            # There is no outgoing action here.  The raw value is diagnostic
            # only, not a future-return estimate or a bootstrap target.
            steps.append(
                {
                    "values": step_values,
                    "actual_actions": None,
                    "interventions": [],
                }
            )
            continue

        actual = [int(action) for action in actions[index]]
        interventions = []
        for actor in range(num_agents):
            target = 1 - actor
            baseline_next_value = float(values[index + 1, target])
            baseline = {
                "action": actual[actor],
                "next_value": baseline_next_value,
                "reward": float(rewards[index]),
                "done": int(dones[index]),
            }

            alternatives = []
            for code in range(len(names)):
                if code == actual[actor]:
                    # This is exactly the factual transition; canonicalize it
                    # to the recorded metrics so its delta is exactly zero.
                    next_value = baseline_next_value
                    reward = baseline["reward"]
                    done = baseline["done"]
                    value_delta = 0.0
                else:
                    next_value = float(next_values[index, actor, code])
                    reward = float(candidate_rewards[index, actor, code])
                    done = int(candidate_dones[index, actor, code])
                    value_delta = next_value - baseline_next_value
                alternatives.append(
                    {
                        "action": code,
                        "next_value": next_value,
                        "reward": reward,
                        "done": done,
                        "value_delta": value_delta,
                    }
                )

            interventions.append(
                {
                    "actor": actor,
                    "target": target,
                    "baseline": baseline,
                    "alternatives": alternatives,
                }
            )

        steps.append(
            {
                "values": step_values,
                "actual_actions": actual,
                "interventions": interventions,
            }
        )

    return {
        "reward": REWARD_DESCRIPTION,
        "action_names": names,
        "steps": steps,
    }


def protocol_metadata(files):
    """Describe the probe contract in protocol.json."""
    return {
        "enabled": True,
        "reward": REWARD_DESCRIPTION,
        "action_names": action_names(),
        "files": list(files),
        "embedded_episode": 0,
        "value": (
            "current critic value before the action, using the recurrent memory "
            "for that observation; the final value is diagnostic only"
        ),
        "step_semantics": (
            "step t values are before action t; interventions describe the "
            "outgoing transition t -> t+1"
        ),
        "intervention": (
            "replace one chef's action on an immutable copy of the recorded "
            "state; keep the other chef's recorded action and the factual "
            "simulator key; do not touch the recorded rollout or its RNG"
        ),
        "successor_value": (
            "target chef's critic value from memory returned after the current "
            "observation, using the candidate next state's done flag"
        ),
        "value_delta": (
            "alternative next_value minus the recorded-action baseline "
            "next_value; raw reward and done are context only"
        ),
        "final_state": (
            "raw current value at the terminal state; diagnostic only, with no "
            "outgoing action, no bootstrap, and no future-return interpretation"
        ),
        "final_transition": (
            "the last intervention's next_value is that diagnostic terminal "
            "value; no future return is added to it"
        ),
        "warning": VALUE_WARNING,
    }
