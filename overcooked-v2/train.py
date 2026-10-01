"""Launch the upstream recurrent-IPPO baseline and save its results locally."""

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
import wandb
from flax import serialization
from flax.traverse_util import flatten_dict
from omegaconf import OmegaConf

from train_check import verify_checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs/coord_ring.yaml"


def load_baseline(config_path: Path, seed: int | None):
    """Read our settings and import the unmodified upstream trainer."""
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if seed is not None:
        config["SEED"] = seed
    # Keep each seed's checkpoint and logs in a separate run.
    if config["NUM_SEEDS"] != 1:
        raise ValueError("This launcher runs one seed at a time.")
    if not 0 <= config["SEED"] < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer.")

    # Import the trainer directly so the pinned upstream source stays unchanged.
    source = SCRIPT_DIR / "JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v2.py"
    spec = importlib.util.spec_from_file_location("upstream_ppo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, config


def execute_training(module, config):
    """Compile and run upstream training; no learning logic lives in this wrapper."""
    started = time.perf_counter()
    # Match upstream's single-seed handling; make_train owns the learning logic.
    key = jax.random.split(jax.random.PRNGKey(config["SEED"]), 1)[0]
    train = jax.jit(module.make_train(config))
    print("Compiling training; the first update may take a while.", flush=True)
    compiled = train.lower(key).compile()
    print("Compilation complete; starting training.", flush=True)
    result = compiled(key)

    # Results contain many arrays. Wait for each before saving; a combined
    # wait stalled on our WSL/GPU setup.
    for leaf in jax.tree.leaves(result):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    # Finish asynchronous logging callbacks before closing the progress file.
    jax.effects_barrier()
    return result, time.perf_counter() - started


def save_results(result, output: Path, elapsed: float) -> None:
    """Save weights and numeric metrics, then verify the weights reload."""
    state = result["runner_state"][0]
    params = jax.device_get(state.params)
    metrics = flatten_dict(jax.device_get(result["metrics"]), sep="/")

    # Catch invalid numbers before saving unusable weights or misleading curves.
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Training produced non-finite model weights.")
    if not all(np.isfinite(np.asarray(x)).all() for x in metrics.values()):
        raise ValueError("Training produced non-finite metrics.")

    checkpoint = output / "params.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(params))
    # Reload and run the model once to check the saved policy is usable.
    verify_checkpoint(result["runner_state"], checkpoint)

    np.savez_compressed(output / "metrics.npz", **metrics)
    summary = {
        "status": "complete",
        "completed_environment_steps": int(metrics["env_step"][-1]),
        "seconds_including_compilation": elapsed,
        "checkpoint_reload_verified": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Inspect without training.")
    args = parser.parse_args()
    module, config = load_baseline(args.config, args.seed)

    # Constructing the environment checks settings without playing any games.
    env = module.jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    updates = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
    # Fail early if settings would produce no updates or an invalid batch reshape.
    if updates < 1:
        raise ValueError("Budget must include at least one full PPO update.")
    if (env.num_agents * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"]:
        raise ValueError("Actor count must be divisible by NUM_MINIBATCHES.")
    print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"Observation shape: {env.observation_space().shape}; PPO updates: {updates}")
    if args.dry_run:
        print("Configuration checked. No simulation or training started.", flush=True)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{config['ENV_KWARGS']['layout']}-seed{config['SEED']}-{timestamp}"
    output = SCRIPT_DIR / "outputs/train" / run_name
    output.mkdir(parents=True, exist_ok=False)
    # Save chosen settings before upstream adds its own derived fields.
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Source/version/device details help explain differences between machines.
    metadata = {
        "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
        "backend": jax.default_backend(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "versions": {
            name: version(name) for name in ("jax", "jaxlib", "flax", "optax")
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {output}. Checkpoint saves on completion.", flush=True)

    # Upstream reports through W&B; redirect that hook to our local progress log.
    with (output / "progress.jsonl").open("x") as progress:
        def log_progress(metrics):
            row = {
                name: float(np.asarray(value))
                for name, value in flatten_dict(metrics, sep="/").items()
            }
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            print(
                f"Update {int(row['update_step'])}/{updates}: "
                f"{int(row['env_step']):,} steps; "
                f"reported training return {row['returned_episode_returns']:.1f}",
                flush=True,
            )

        wandb.init(mode="disabled", project=config["PROJECT"], config=config)
        # Restore the logging hook in finally, even if training fails.
        original_log = module.wandb.log
        module.wandb.log = log_progress
        try:
            result, elapsed = execute_training(module, config)
            save_results(result, output, elapsed)
            print(f"Complete. Saved and verified checkpoint: {output / 'params.msgpack'}")
        finally:
            module.wandb.log = original_log
            wandb.finish()


if __name__ == "__main__":
    main()
