"""Play a frozen policy and save scores plus one interactive replay (no learning)."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from jaxmarl.environments.overcooked_v2.common import (
    DynamicObject,
    OvercookedActionsEnum,
    StaticObject,
)
from preview import save_preview
from train import DEFAULT_CONFIG, SCRIPT_DIR, load_baseline
from scenarios import make_env


def initialize_policy(module, config, env):
    """Recreate the upstream trainer's exact seed splitting and initial network."""
    network = module.ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    key = jax.random.split(jax.random.PRNGKey(config["SEED"]), 1)[0]
    _, parameter_key = jax.random.split(key)
    # Use the trainer's dummy batch too, so the before policy matches its start.
    batch = config["NUM_ENVS"]
    hidden = module.ScannedRNN.initialize_carry(batch, config["GRU_HIDDEN_DIM"])
    inputs = (
        jnp.zeros((1, batch, *env.observation_space().shape)),
        jnp.zeros((1, batch)),
    )
    params = jax.jit(lambda key: network.init(key, hidden, inputs))(parameter_key)
    return network, params


def check_run_revision(run, revision):
    """Reject saved training results that record a different JaxMARL code version."""
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["upstream_revision"] != revision:
        raise ValueError(f"Saved policy and local upstream revisions differ: {run}")


def check_partner_config(config, partner_config):
    """Check compatibility with the evaluator's shared network and observation setup.

    Refactor or remove these restrictions when each policy can use its own network
    design and observation settings. Training seeds and hyperparameters may differ.
    """
    fields = ("ENV_NAME", "ENV_KWARGS", "ACTIVATION", "FC_DIM_SIZE", "GRU_HIDDEN_DIM")
    differences = [field for field in fields if config[field] != partner_config[field]]
    if config.get("ALGORITHM", "ippo_rnn") != partner_config.get("ALGORITHM", "ippo_rnn"):
        differences.append("ALGORITHM")
    if differences:
        raise ValueError("Policies have incompatible settings: " + ", ".join(differences))


def load_checkpoint(run, template):
    params = serialization.from_bytes(template, (run / "params.msgpack").read_bytes())
    for value, expected in zip(jax.tree.leaves(params), jax.tree.leaves(template), strict=True):
        if np.shape(value) != np.shape(expected) or not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Checkpoint has incompatible shapes or non-finite weights: {run}")
    return params


def delivery_counts(before, after, actions):
    """Count actual dish removals at serving stations, including simultaneous ones."""
    forward = jax.vmap(lambda agent: agent.get_fwd_pos())(before.agents)
    at_goal = before.grid[forward.y, forward.x, 0] == StaticObject.GOAL
    inventory = before.agents.inventory
    delivered = (
        (actions == OvercookedActionsEnum.interact)
        & at_goal
        & ((inventory & DynamicObject.COOKED) != 0)
        & (after.agents.inventory == DynamicObject.EMPTY)
    )
    correct_recipe = inventory == (
        before.recipe | DynamicObject.PLATE | DynamicObject.COOKED
    )
    return jnp.sum(delivered & correct_recipe), jnp.sum(delivered & ~correct_recipe)


def make_episode(module, config, env, network):
    """Compile playback only: sample actions, advance the kitchen, record events."""
    @jax.jit
    def play(params, seed, partner_params=None):
        key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        observations, initial = env.reset(reset_key)
        hidden = module.ScannedRNN.initialize_carry(
            env.num_agents, config["GRU_HIDDEN_DIM"]
        )

        def step(carry, _):
            observations, state, hidden, key = carry
            key, action_key, step_key = jax.random.split(key, 3)
            # Each chef keeps its own memory, whether weights are shared or different.
            obs_batch = jnp.stack([observations[agent] for agent in env.agents])
            dones = jnp.full((1, env.num_agents), state.done)
            if partner_params is None:
                # Preserve the original shared-policy playback and sampling behavior.
                hidden, policy, _ = network.apply(params, hidden, (obs_batch[None], dones))
            else:
                memories, logits = [], []
                # This two-chef loop is compiled by JAX, not repeated in Python at runtime.
                for chef, weights in enumerate((params, partner_params)):
                    inputs = (obs_batch[None, chef:chef + 1], dones[:, chef:chef + 1])
                    memory, chef_policy, _ = network.apply(
                        weights, hidden[chef:chef + 1], inputs
                    )
                    memories.append(memory)
                    logits.append(chef_policy.logits)
                hidden = jnp.concatenate(memories, axis=0)
                policy = module.distrax.Categorical(logits=jnp.concatenate(logits, axis=1))
            # Sample both actions with the same random-number layout for every pairing.
            actions = policy.sample(seed=action_key)[0]
            action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            # step_env retains the final kitchen, whereas env.step auto-resets it.
            observations, after, rewards, _, _ = env.step_env(
                step_key, state, action_dict
            )
            if env.name == "oc2_sushi_1_3":
                correct, wrong = after.deliveries - state.deliveries, jnp.int32(0)
            else:
                correct, wrong = delivery_counts(state, after, actions)
            # Reward is shared: read it once. Do not add training's shaping bonuses.
            reward = rewards[env.agents[0]]
            finite = jnp.all(jnp.isfinite(policy.probs))
            trace = (after, actions, reward, correct, wrong, finite)
            return (observations, after, hidden, key), trace

        _, trace = jax.lax.scan(
            step, (observations, initial, hidden, key), None, env.max_steps
        )
        return initial, trace

    return play


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--untrained", action="store_true", help="Use fresh baseline weights.")
    source.add_argument(
        "--run", type=Path,
        help="Training directory containing config.json and params.msgpack.",
    )
    parser.add_argument(
        "--partner-run", type=Path,
        help="Optional saved policy for chef 2 (blue); --run controls chef 1 (red).",
    )
    parser.add_argument("--seed", type=int, help="Policy initialization seed (untrained only).")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--eval-seed", type=int, default=1000,
        help="First test seed; independent of training.",
    )
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.eval_seed <= 2**32 - args.episodes:
        parser.error("Use a positive episode count and unsigned 32-bit test seeds.")
    if args.run and args.seed is not None:
        parser.error("A saved policy uses its saved training seed; omit --seed.")

    if args.partner_run and not args.run:
        parser.error("--partner-run requires --run; it cannot accompany --untrained.")
    run = args.run.resolve() if args.run else None
    partner_run = args.partner_run.resolve() if args.partner_run else None
    revision = (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip()
    for source_run in (run, partner_run):
        if source_run:
            check_run_revision(source_run, revision)
    module, config = load_baseline(
        run / "config.json" if run else DEFAULT_CONFIG, args.seed
    )
    partner_config = config
    if partner_run:
        partner_config = json.loads((partner_run / "config.json").read_text())
        check_partner_config(config, partner_config)
    env = make_env(env_name=config["ENV_NAME"], **config["ENV_KWARGS"])
    if partner_run and env.num_agents != 2:
        raise ValueError("Cross-policy evaluation requires exactly two chefs.")
    network, params = initialize_policy(module, config, env)
    partner_params = load_checkpoint(partner_run, params) if partner_run else None
    if run:
        params = load_checkpoint(run, params)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Policy contains non-finite weights.")
    checkpoint_bytes = serialization.to_bytes(jax.device_get(params))

    mode = "cross-policy" if partner_run else ("trained" if run else "untrained")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{config['ENV_KWARGS']['layout']}-{mode}-{timestamp}"
    output = SCRIPT_DIR / "outputs/evaluate" / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not run:
        # Keep the actual before weights, not just an initialization seed.
        (output / "params.msgpack").write_bytes(checkpoint_bytes)
    if partner_run:
        (output / "partner-config.json").write_text(json.dumps(partner_config, indent=2) + "\n")
    policy_seeds = [config["SEED"], partner_config["SEED"]]
    partner_bytes = (
        serialization.to_bytes(jax.device_get(partner_params))
        if partner_run else checkpoint_bytes
    )
    # Record both checkpoint identities and their fixed assignment to chef colors.
    policies = [
        {"agent": agent, "chef": chef + 1, "color": ("red", "blue")[chef],
         "source_run": str(source_run) if source_run else None,
         "training_seed": policy_seeds[chef],
         "policy_sha256": hashlib.sha256(data).hexdigest()}
        for chef, (agent, source_run, data) in enumerate(zip(
            env.agents, (run, partner_run or run), (checkpoint_bytes, partner_bytes)
        ))
    ]
    seeds = list(range(args.eval_seed, args.eval_seed + args.episodes))
    frame_ms = round(env.rules.step_seconds * 1000) if env.name == "oc2_sushi_1_3" else 100
    protocol = {
        "policy": mode, "source_run": str(run) if run else None,
        "partner_run": str(partner_run) if partner_run else None, "policies": policies,
        "policy_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "upstream_revision": revision, "backend": jax.default_backend(),
        "episode_seeds": seeds, "action_selection": "sample", "recorded_episode": 0,
        "reward": "shared raw environment reward counted once; no shaping",
        "gif_stride": 2, "gif_frame_ms": 2 * frame_ms,
        "replay_file": "replay.html", "replay_schema_version": 1,
        "replay_stride": 1, "replay_frame_ms": frame_ms,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        f"Evaluating {mode} policy, {len(seeds)} episodes; no training. Results: {output}",
        flush=True,
    )
    print(f"Chef 1 (red): seed {policy_seeds[0]}; chef 2 (blue): seed {policy_seeds[1]}", flush=True)
    play = make_episode(module, config, env, network)
    rows = []
    recorded = None
    for episode, seed in enumerate(seeds):
        initial, trace = jax.device_get(play(params, np.uint32(seed), partner_params))
        states, actions, rewards, correct, wrong, finite = trace
        if not finite.all() or not np.isfinite(rewards).all() or not bool(states.done[-1]):
            raise ValueError("Invalid policy predictions, rewards, or unfinished episode.")
        row = {
            "episode": episode, "seed": seed, "steps": len(actions),
            "team_reward": float(rewards.sum()),
            "correct_dishes": int(correct.sum()), "wrong_dishes": int(wrong.sum()),
        }
        rows.append(row)
        print(f"Episode {episode + 1}/{len(seeds)}: {row}", flush=True)
        if episode == 0:
            recorded = (initial, trace)

    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Always record the first episode; do not pick whichever looks most successful.
    print("Rendering the first episode with spectator counters.", flush=True)
    if env.name == "oc2_sushi_1_3":
        from scenarios.oc2_sushi_1_3.preview import save_policy_preview
        save_policy_preview(env, *recorded, output, mode, seeds[0])
    else:
        save_preview(*recorded, output, config, mode, seeds[0], policy_seeds=policy_seeds)
    summary = {
        "status": "complete", "policy": mode, "episodes": len(rows),
        "mean_team_reward": float(np.mean([row["team_reward"] for row in rows])),
        "mean_correct_dishes": float(np.mean([row["correct_dishes"] for row in rows])),
        "total_correct_dishes": sum(row["correct_dishes"] for row in rows),
        "total_wrong_dishes": sum(row["wrong_dishes"] for row in rows),
        "episodes_with_a_correct_delivery": sum(row["correct_dishes"] > 0 for row in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Replay (open in a browser): {output / 'replay.html'}", flush=True)
    print(f"GIF: {output / 'rollout.gif'}", flush=True)


if __name__ == "__main__":
    main()
