"""Check upstream recurrent PPO with a tiny budget, then save/reload its weights."""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import serialization
from flax.traverse_util import flatten_dict
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent


def load_baseline(layout: str):
    """Load the pinned upstream algorithm and configure a tiny local check."""
    source = SCRIPT_DIR / "JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v2.py"
    spec = importlib.util.spec_from_file_location("upstream_ppo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config_path = source.parent / "config/ippo_rnn_overcooked_v2.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path))
    # Eight kitchens x 32 steps x two PPO updates = 512 environment steps.
    config.update(
        NUM_ENVS=8,
        NUM_STEPS=32,
        TOTAL_TIMESTEPS=512,
        NUM_MINIBATCHES=4,
        UPDATE_EPOCHS=2,
        NUM_SEEDS=1,
        ANNEAL_LR=False,
        WANDB_MODE="disabled",
    )
    config["ENV_KWARGS"]["layout"] = layout
    return module, config


def log_progress(metrics) -> None:
    print(
        f"PPO update {int(metrics['update_step'])} complete; "
        f"{int(metrics['env_step'])} environment steps.",
        flush=True,
    )


def run_training(module, config):
    """Compile and run the two-update check, waiting for GPU work and callbacks."""
    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)
    print("Preparing 2 PPO updates (512 environment steps).", flush=True)
    started = time.perf_counter()

    train = jax.jit(module.make_train(config))
    key = jax.random.PRNGKey(config["SEED"])
    print("Tracing the training function...", flush=True)
    lowered = train.lower(key)
    print(
        f"Trace complete after {time.perf_counter() - started:.1f}s; compiling...",
        flush=True,
    )

    compiled = lowered.compile()
    print(
        f"Compile complete after {time.perf_counter() - started:.1f}s; executing...",
        flush=True,
    )
    result = compiled(key)
    print("Dispatched; synchronizing result arrays...", flush=True)

    # JAX's combined readiness wait stalled on this CUDA/WSL stack.
    for leaf in jax.tree.leaves(result):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()

    print("GPU computation complete; waiting for logging callbacks...", flush=True)
    jax.effects_barrier()
    return result, time.perf_counter() - started


def verify_checkpoint(runner_state, checkpoint: Path) -> None:
    """Reload saved weights and check one forward pass of the recurrent model."""
    train_state, _env_state, observations, dones, _update, hidden_state, _rng = (
        runner_state
    )
    restored = serialization.from_bytes(train_state.params, checkpoint.read_bytes())

    # Pack both agents' observations into one batch, with a single time step.
    obs_batch = jnp.concatenate(
        [observations[agent] for agent in sorted(observations)], axis=0
    )
    inputs = (obs_batch[jnp.newaxis], dones[jnp.newaxis])

    # Upstream TrainState.params already contains Flax's variables dictionary.
    _, policy, values = jax.jit(train_state.apply_fn)(restored, hidden_state, inputs)
    probabilities = np.asarray(policy.probs)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(np.asarray(values)).all()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    print("Checkpoint reloaded; policy inference verified.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="grounded_coord_simple")
    args = parser.parse_args()
    module, config = load_baseline(args.layout)

    output = SCRIPT_DIR / "outputs/train-check"
    output.mkdir(parents=True, exist_ok=True)
    # make_train adds derived fields; record the input configuration first.
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Use local progress messages in place of external logging for this check.
    wandb.init(mode="disabled", project="se390-overcooked-v2", config=config)
    original_log = module.wandb.log
    module.wandb.log = log_progress

    try:
        result, elapsed = run_training(module, config)
        train_state = result["runner_state"][0]

        # Flatten nested recipe metrics so the NPZ contains only numeric arrays.
        metrics = flatten_dict(jax.device_get(result["metrics"]), sep="/")
        assert all(
            np.isfinite(np.asarray(leaf)).all()
            for leaf in jax.tree.leaves(train_state.params)
        )
        assert all(
            np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(metrics)
        )
        # Two PPO updates x two optimization passes x four minibatches.
        assert int(train_state.step) == 16, (
            f"Unexpected optimizer step: {train_state.step}"
        )

        checkpoint = output / "params.msgpack"
        checkpoint.write_bytes(serialization.to_bytes(train_state.params))
        np.savez_compressed(output / "metrics.npz", **metrics)
        verify_checkpoint(result["runner_state"], checkpoint)

        summary = {
            "purpose": (
                "Training plumbing check only; insufficient budget for a useful policy."
            ),
            "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
            "jax_version": jax.__version__,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "backend": jax.default_backend(),
            "environment_steps": 512,
            "ppo_updates": 2,
            "optimizer_steps": int(train_state.step),
            "checkpoint_reload_verified": True,
            "seconds_including_compilation": elapsed,
            "final_metrics": {
                name: float(np.asarray(values).reshape(-1)[-1])
                for name, values in metrics.items()
            },
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))

    finally:
        module.wandb.log = original_log
        wandb.finish()


if __name__ == "__main__":
    main()
