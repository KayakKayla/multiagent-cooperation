"""Check the simulator with one random-action episode and save a preview."""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer

SCRIPT_DIR = Path(__file__).resolve().parent
EPISODE_LIMIT = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="grounded_coord_simple")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=EPISODE_LIMIT)
    args = parser.parse_args()

    if not 1 <= args.steps <= EPISODE_LIMIT:
        parser.error(f"--steps must be between 1 and {EPISODE_LIMIT} (one episode)")

    return args


def save_preview(states, output: Path, view_radius: int) -> None:
    """Render real simulator states as a spectator, showing agent view limits."""
    # Sampling affects only the GIF; the simulation still advances every step.
    sampled_states = states[::4]
    if (len(states) - 1) % 4:
        sampled_states.append(states[-1])

    batched_states = jax.tree.map(lambda *xs: jnp.stack(xs), *sampled_states)
    frames = np.asarray(
        OvercookedV2Visualizer().render_sequence(
            batched_states, agent_view_size=view_radius
        )
    )

    imageio.imwrite(output / "initial.png", frames[0])
    imageio.mimsave(output / "random-rollout.gif", frames, duration=160, loop=0)


def main() -> None:
    args = parse_args()
    output = SCRIPT_DIR / "outputs/smoke"
    output.mkdir(parents=True, exist_ok=True)

    settings = {
        "layout": args.layout,
        "agent_view_size": 2,
        "negative_rewards": True,
        "sample_recipe_on_delivery": True,
        "random_agent_positions": True,
    }
    devices = [str(device) for device in jax.devices()]
    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; devices={devices}",
        flush=True,
    )

    # Each agent gets a limited numerical observation of the same kitchen.
    env = jaxmarl.make("overcooked_v2", **settings)
    step_env = jax.jit(env.step_env)
    key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    observations, state = env.reset(reset_key)
    states = [state]
    total_reward = 0.0

    started = time.perf_counter()
    print(
        f"Running {args.steps} random-action steps in {args.layout}; "
        "first call compiles.",
        flush=True,
    )

    for _ in range(args.steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action_values = jax.random.randint(
            action_key, (env.num_agents,), 0, env.num_actions
        )
        actions = {
            agent: action_values[index] for index, agent in enumerate(env.agents)
        }

        # step_env preserves the terminal state instead of automatically resetting.
        observations, state, rewards, dones, _info = step_env(step_key, state, actions)
        total_reward += float(rewards[env.agents[0]])  # Count shared reward once.
        states.append(state)

        if bool(dones["__all__"]):
            break

    for observation in observations.values():
        assert np.isfinite(np.asarray(observation)).all()
    simulation_seconds = time.perf_counter() - started

    print("Rendering the real simulator state (spectator view).", flush=True)
    save_preview(states, output, settings["agent_view_size"])

    summary = {
        "purpose": (
            "Installation check; random actions, "
            "not a trained agent or research result."
        ),
        "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
        "settings": settings,
        "seed": args.seed,
        "steps": len(states) - 1,
        "observation_shapes": {
            agent: list(observation.shape)
            for agent, observation in observations.items()
        },
        "shared_team_reward": total_reward,
        "terminal": bool(dones["__all__"]),
        "simulation_seconds_including_compilation": simulation_seconds,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Preview saved to {output / 'random-rollout.gif'}")


if __name__ == "__main__":
    main()
