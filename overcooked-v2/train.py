# NOTE: make sure to set export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
# to ensure consistency between training runs
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from omegaconf import OmegaConf

import checkpoint as ckpt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs/coord_ring.yaml"

ALGORITHMS = {
    "ippo_rnn": SCRIPT_DIR / "algorithms/ippo_rnn.py",
}

def load_config(config_path: Path, seed: int | None = None) -> dict:
    """Read and validate a config (YAML, or a saved config.json)."""
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if seed is not None:
        config["SEED"] = seed
    # Keep each seed's checkpoint and logs in a separate run.
    if config["NUM_SEEDS"] != 1:
        raise ValueError("This launcher runs one seed at a time.")
    if not 0 <= config["SEED"] < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer.")
    name = config.get("ALGORITHM", "ippo_rnn")
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown ALGORITHM {name!r}; choose from {sorted(ALGORITHMS)}")
    config["ALGORITHM"] = name
    return config


def load_algorithm(name: str):
    """Import the training module registered under `name`."""
    spec = importlib.util.spec_from_file_location(name, ALGORITHMS[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_baseline(config_path: Path, seed: int | None):
    """Config plus algorithm module. evaluate.py imports this."""
    config = load_config(config_path, seed)
    return load_algorithm(config["ALGORITHM"]), config


def apply_runtime_flags(config: dict) -> None:
    """Merge RUNTIME.XLA_FLAGS from the config into XLA_FLAGS, before jax starts.

    Flags already in the environment (such as this machine's WSL workaround
    in run.sh) are kept, unless the config sets the same flag.
    """
    flags = list(config.get("RUNTIME", {}).get("XLA_FLAGS", []))
    if not flags:
        return
    if "jax" in sys.modules:
        raise RuntimeError(
            "jax was imported before the config's XLA_FLAGS could be applied; "
            "they would be silently ignored."
        )
    names = {flag.split("=", 1)[0] for flag in flags}
    kept = [f for f in os.environ.get("XLA_FLAGS", "").split()
            if f.split("=", 1)[0] not in names]
    os.environ["XLA_FLAGS"] = " ".join(kept + flags)


def resolve_resume(path: Path):
    """Return (run folder, checkpoint folder) for a run folder or checkpoint folder."""
    path = path.resolve()
    if (path / ckpt.META_FILE).is_file():
        return path.parent.parent, path
    latest = ckpt.latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"No complete checkpoint found under {path}")
    return path, latest


def git_state() -> dict:
    """Commit and dirty flag of this repo, so a run can be tied to its code."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=SCRIPT_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def execute_training(module, config, key, output, resume_path=None):
    import jax
    import numpy as np

    started = time.perf_counter()
    init, run_updates = module.make_trainer(config)
    total = config["NUM_UPDATES"]
    interval = config["CHECKPOINT_INTERVAL_UPDATES"]
    keep_last = config["CHECKPOINT_KEEP_LAST"]
    source = ALGORITHMS[config["ALGORITHM"]]

    state = jax.jit(init)(key)  # a fresh state is also the template for restoring
    metrics, done = {}, 0
    if resume_path is not None:
        state, metrics, done = ckpt.load_checkpoint(resume_path, state, config, source)
        print(f"Restored checkpoint at update {done}/{total}.", flush=True)

    chunk_fns = {}  # one compiled function per distinct chunk length
    while done < total:
        n = min(interval, total - done)
        if n not in chunk_fns:
            print(f"Compiling a {n}-update chunk; the first chunk may take a while.",
                  flush=True)
            chunk_fns[n] = jax.jit(lambda s, n=n: run_updates(s, n))
        state, chunk_metrics = chunk_fns[n](state)

        # Wait for each array before saving; a combined wait stalled on our WSL/GPU setup.
        for leaf in jax.tree.leaves((state, chunk_metrics)):
            if isinstance(leaf, jax.Array):
                leaf.block_until_ready()
        # Finish asynchronous logging callbacks so progress.jsonl is complete.
        jax.effects_barrier()

        if not all(np.isfinite(np.asarray(x)).all()
                   for x in jax.tree.leaves(jax.device_get(state[0].params))):
            raise ValueError(f"Model weights became non-finite by update {done + n}.")
        done += n
        metrics = ckpt.concat_metrics([metrics, jax.device_get(chunk_metrics)])
        saved = ckpt.save_checkpoint(output, state, [metrics], done, config,
                                     source, keep_last)
        print(f"Saved checkpoint {saved.name}", flush=True)
        # EXTENSION POINT: run the periodic evaluation here when
        # done % config["EVAL_INTERVAL_UPDATES"] == 0.

    return state, metrics, time.perf_counter() - started


def save_results(state, metrics, output: Path, elapsed: float, resumed_from) -> None:
    """Save weights and numeric metrics, then verify the weights reload."""
    import jax
    import numpy as np
    from flax import serialization

    from train_check import verify_checkpoint

    params = jax.device_get(state[0].params)

    # Catch invalid numbers before saving unusable weights or misleading curves.
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
        raise ValueError("Training produced non-finite model weights.")
    if not all(np.isfinite(np.asarray(x)).all() for x in metrics.values()):
        raise ValueError("Training produced non-finite metrics.")

    weights = output / "params.msgpack"
    weights.write_bytes(serialization.to_bytes(params))
    # Reload and run the model once to check the saved policy is usable.
    verify_checkpoint(state, weights)

    np.savez_compressed(output / "metrics.npz", **metrics)
    summary = {
        "status": "complete",
        "completed_environment_steps": int(metrics["env_step"][-1]),
        # Covers this session only; a resumed run reports the part it ran.
        "seconds_including_compilation": elapsed,
        "resumed_from_update": resumed_from,
        "checkpoint_reload_verified": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def truncate_progress(path: Path, keep_rows: int) -> None:
    """Drop log rows from the abandoned part of the run (after the checkpoint)."""
    lines = path.read_text().splitlines() if path.exists() else []
    if len(lines) < keep_rows:
        print(f"WARNING: {path.name} has {len(lines)} rows but the checkpoint is at "
              f"update {keep_rows}; the log has a gap.", flush=True)
    kept = lines[:keep_rows]
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(line + "\n" for line in kept))
    os.replace(temp, path)


def runtime_info() -> dict:
    import jax

    return {
        "backend": jax.default_backend(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "versions": {name: version(name) for name in ("jax", "jaxlib", "flax", "optax")},
    }


def run(args, config, run_dir, resume_path) -> None:
    import jax
    import numpy as np
    import wandb
    from flax.traverse_util import flatten_dict

    module = load_algorithm(config["ALGORITHM"])

    # Constructing the environment checks settings without playing any games.
    env = module.jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    updates = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
    # Fail early if settings would produce no updates or an invalid batch reshape.
    if updates < 1:
        raise ValueError("Budget must include at least one full PPO update.")
    if (env.num_agents * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"]:
        raise ValueError("Actor count must be divisible by NUM_MINIBATCHES.")

    # Fill in checkpoint settings so the saved config records what was used.
    config.setdefault("CHECKPOINT_INTERVAL_UPDATES", updates)
    config.setdefault("CHECKPOINT_KEEP_LAST", 3)
    interval, keep_last = config["CHECKPOINT_INTERVAL_UPDATES"], config["CHECKPOINT_KEEP_LAST"]
    if not (isinstance(interval, int) and interval >= 1):
        raise ValueError("CHECKPOINT_INTERVAL_UPDATES must be a positive integer.")
    if not (isinstance(keep_last, int) and keep_last >= 0):
        raise ValueError("CHECKPOINT_KEEP_LAST must be 0 (keep all) or a positive integer.")
    if updates % interval:
        print(f"NOTE: {updates} updates do not divide evenly by the checkpoint interval "
              f"{interval}; the last chunk is shorter and compiles separately.", flush=True)

    print(OmegaConf.to_yaml(OmegaConf.create(config)), flush=True)
    print(f"Backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '')!r}")
    print(f"Observation shape: {env.observation_space().shape}; PPO updates: {updates}")
    if args.dry_run:
        print("Configuration checked. No simulation or training started.", flush=True)
        return

    algorithm_hash = ckpt.file_hash(ALGORITHMS[config["ALGORITHM"]])
    resumed_from = None
    if resume_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_name = f"{config['ENV_KWARGS']['layout']}-seed{config['SEED']}-{timestamp}"
        output = SCRIPT_DIR / "outputs/train" / run_name
        output.mkdir(parents=True, exist_ok=False)
        # Save chosen settings before the algorithm adds its own derived fields.
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # Source/version/device details help explain differences between machines.
        metadata = {
            "upstream_revision": (SCRIPT_DIR / "UPSTREAM_REVISION").read_text().strip(),
            **runtime_info(),
            **git_state(),
            "algorithm_source_hash": algorithm_hash,
            "resumes": [],
        }
        progress_mode = "x"
    else:
        output = run_dir
        metadata = json.loads((output / "metadata.json").read_text())
        resumed_from = int(json.loads((resume_path / ckpt.META_FILE).read_text())["update"])
        now = runtime_info()
        for name in ("backend", "versions"):
            if now[name] != metadata.get(name):
                print(f"WARNING: {name} differs from the original run: "
                      f"{metadata.get(name)} vs {now[name]}", flush=True)
        # Newer checkpoints belong to the timeline being abandoned; leaving them would
        # make "latest" misleading and let retention delete the fresh ones.
        for stale in ckpt.list_checkpoints(output):
            if int(stale.name.split("-")[1]) > resumed_from:
                print(f"Removing checkpoint {stale.name} (after the resume point).",
                      flush=True)
                shutil.rmtree(stale)
        truncate_progress(output / "progress.jsonl", resumed_from)
        metadata.setdefault("resumes", []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "from_update": resumed_from,
            **now,
            **git_state(),
            "algorithm_source_hash": algorithm_hash,
        })
        progress_mode = "a"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {output}. Checkpoint every {interval} update(s).", flush=True)

    # Seed handling matches upstream's single-seed behavior; on resume the
    # saved RNG state in the checkpoint takes over, so the seed is only the start.
    key = jax.random.split(jax.random.PRNGKey(config["SEED"]), 1)[0]

    # Upstream reports through W&B; redirect that hook to our local progress log.
    with (output / "progress.jsonl").open(progress_mode) as progress:
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
            state, metrics, elapsed = execute_training(
                module, config, key, output, resume_path)
            if config["NUM_UPDATES"] != updates:
                raise RuntimeError("Update count disagrees with the algorithm's own.")
            save_results(state, metrics, output, elapsed, resumed_from)
            print(f"Complete. Saved and verified weights: {output / 'params.msgpack'}")
        finally:
            module.wandb.log = original_log
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        help="YAML or saved config.json. Default: the standard config, "
                             "or the run's own config.json when resuming.")
    parser.add_argument("--seed", type=int, help="Override SEED (not allowed with --resume).")
    parser.add_argument("--resume", type=Path, metavar="PATH",
                        help="A run folder (uses its latest checkpoint) or one "
                             "checkpoints/update-NNNNNN folder.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without training.")
    args = parser.parse_args()

    run_dir = resume_path = None
    if args.resume is not None:
        if args.seed is not None:
            parser.error("--seed cannot be used with --resume; the checkpoint holds the RNG state.")
        run_dir, resume_path = resolve_resume(args.resume)
    config_path = args.config or (run_dir / "config.json" if run_dir else DEFAULT_CONFIG)
    config = load_config(config_path, args.seed)
    apply_runtime_flags(config)  # must happen before jax is imported
    run(args, config, run_dir, resume_path)


if __name__ == "__main__":
    main()
