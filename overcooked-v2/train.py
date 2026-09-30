"""Run one saved recurrent-IPPO self-play baseline using the pinned upstream code."""

import argparse
import hashlib
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
    """Load our explicit configuration and the unmodified upstream algorithm."""
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if seed is not None:
        config["SEED"] = seed
    if config["NUM_SEEDS"] != 1:
        raise ValueError("This launcher runs one seed at a time.")
    if not 0 <= config["SEED"] < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer.")

    source = SCRIPT_DIR / "JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v2.py"
    spec = importlib.util.spec_from_file_location("upstream_ppo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Upstream rounds the budget down to a whole number of PPO updates.
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    updates = config["TOTAL_TIMESTEPS"] // steps_per_update
    if updates < 1:
        raise ValueError("Budget must include at least one full PPO update.")
    if (env.num_agents * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"]:
        raise ValueError("Actor count must be divisible by NUM_MINIBATCHES.")
    return module, config, env, int(updates)


def print_plan(config, env, updates: int) -> None:
    print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)
    print(f"Each chef's observation shape: {env.observation_space().shape}")
    actual_steps = updates * config["NUM_ENVS"] * config["NUM_STEPS"]
    print(f"Plan: {updates} PPO updates; {actual_steps:,} environment steps.")
    print("An environment step counts one kitchen step with both chefs acting.", flush=True)


def execute_training(module, config):
    """Compile and execute the original training function, then wait for logging."""
    started = time.perf_counter()
    # Use the upstream entry point's single-seed key convention.
    key = jax.random.split(jax.random.PRNGKey(config["SEED"]), 1)[0]
    train = jax.jit(module.make_train(config))
    print(
        "Tracing and compiling training; the first progress update may take a while.",
        flush=True,
    )
    compiled = train.lower(key).compile()
    print(f"Compilation finished after {time.perf_counter() - started:.1f}s.", flush=True)
    result = compiled(key)

    # Individual waits avoid the combined readiness stall seen on this WSL setup.
    for leaf in jax.tree.leaves(result):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    jax.effects_barrier()
    return result, time.perf_counter() - started


def save_results(result, output: Path, config, elapsed: float) -> dict:
    """Save weights, check that they reload, and preserve numeric training metrics."""
    train_state = result["runner_state"][0]
    params = jax.device_get(train_state.params)
    metrics = flatten_dict(jax.device_get(result["metrics"]), sep="/")
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Training produced non-finite model weights.")
    if not all(np.isfinite(np.asarray(x)).all() for x in metrics.values()):
        raise ValueError("Training produced non-finite metrics.")

    updates = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
    expected_optimizer_steps = (
        updates * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    )
    if int(train_state.step) != expected_optimizer_steps:
        raise ValueError("Unexpected number of optimizer steps.")

    checkpoint = output / "params.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(params))
    verify_checkpoint(result["runner_state"], checkpoint)
    np.savez_compressed(output / "metrics.npz", **metrics)
    return {
        "status": "complete",
        "requested_environment_steps": config["TOTAL_TIMESTEPS"],
        "completed_environment_steps": int(updates * config["NUM_ENVS"] * config["NUM_STEPS"]),
        "ppo_updates": int(updates),
        "optimizer_steps": int(train_state.step),
        "seconds_including_compilation": elapsed,
        "checkpoint_reload_verified": True,
        "last_training_metrics": {
            name: float(np.asarray(values).reshape(-1)[-1])
            for name, values in metrics.items()
        },
        "evaluation_note": "Training statistics only; frozen-policy evaluation is still needed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect settings without playing or training.",
    )
    args = parser.parse_args()
    module, config, env, updates = load_baseline(args.config, args.seed)
    print_plan(config, env, updates)
    if args.dry_run:
        print("Configuration checked. No simulation or training was started.", flush=True)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"{config['ENV_KWARGS']['layout']}-seed{config['SEED']}-{timestamp}"
    output = SCRIPT_DIR / "outputs/train" / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = {
        "started_utc": timestamp,
        "paper": "https://arxiv.org/abs/2503.17821",
        "paper_sections": ["C.1", "C.2", "D.2.1 Table 4"],
        "scope": "Single-seed local baseline; not the paper's ten-seed reproduction.",
        "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "versions": {
            name: version(name) for name in ("jax", "jaxlib", "flax", "optax")
        },
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results directory: {output}", flush=True)
    print(
        "The policy checkpoint is saved on successful completion; "
        "interrupted runs cannot resume.",
        flush=True,
    )

    # Keep progress local. Upstream calls this hook after each PPO update.
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
        original_log = module.wandb.log
        module.wandb.log = log_progress
        try:
            result, elapsed = execute_training(module, config)
            summary = save_results(result, output, config, elapsed)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"Training complete. Saved and verified checkpoint: "
                f"{output / 'params.msgpack'}",
                flush=True,
            )
        finally:
            module.wandb.log = original_log
            wandb.finish()


if __name__ == "__main__":
    main()
