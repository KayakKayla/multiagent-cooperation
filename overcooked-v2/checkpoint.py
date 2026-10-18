"""Full-state checkpoints for chunked, resumable training.

A checkpoint is a directory under <run>/checkpoints/ named update-NNNNNN:

    state.npz       the whole runner state (params, optimizer, env, RNG, step)
    metrics.npz     every training metric collected so far, one row per update
    meta.json       update count plus hashes used to refuse a mismatched resume

Writes go to a temporary directory that is renamed into place, so an
interrupted save never leaves a half-written checkpoint behind.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

# jax is imported inside the functions that need it, never at module level:
# train.py must set XLA_FLAGS before jax is first imported, and it also needs
# list_checkpoints/latest_checkpoint before that point.

CHECKPOINT_DIRNAME = "checkpoints"
STATE_FILE = "state.npz"
METRICS_FILE = "metrics.npz"
META_FILE = "meta.json"
FORMAT_VERSION = 1
_NAME = re.compile(r"^update-(\d{6})$")

# Settings that do not change the training trajectory, so they may differ
# between the original run and a resume (for example, a different checkpoint
# interval after a crash). Everything else must match exactly.
NON_TRAINING_KEYS = frozenset({
    "CHECKPOINT_INTERVAL_UPDATES",
    "CHECKPOINT_KEEP_LAST",
    "EVAL_INTERVAL_UPDATES",
    "EVAL_EPISODES",
    "EVAL_SEED_START",
    "RESUME_FROM",
    "ENTITY",
    "PROJECT",
    "WANDB_MODE",
})
# make_trainer writes these into the config dict while it builds, so a hash
# taken before and after building would otherwise differ.
DERIVED_KEYS = frozenset({"NUM_ACTORS", "NUM_UPDATES", "MINIBATCH_SIZE"})


def config_hash(config: dict) -> str:
    """Stable hash of the settings that determine the training trajectory."""
    relevant = {
        k: v for k, v in config.items()
        if k not in NON_TRAINING_KEYS and k not in DERIVED_KEYS
    }
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def file_hash(path) -> str:
    """Hash a source file, ignoring CRLF/LF so Windows and Linux checkouts match."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _flatten(tree, prefix="") -> dict:
    """Flatten nested dicts to {"a/b": value}; already-flat dicts pass through."""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "/"))
        else:
            flat[name] = value
    return flat


def concat_metrics(parts) -> dict:
    """Join per-chunk metric trees into one flat dict, one row per update."""
    flat_parts = []
    for part in parts:
        # A fresh run starts from {}; taking key names from an empty part
        # would silently drop every metric.
        if not part:
            continue
        flat_parts.append({
            name: np.asarray(value) for name, value in _flatten(part).items()
        })
    if not flat_parts:
        return {}
    return {
        name: np.concatenate([p[name] for p in flat_parts], axis=0)
        for name in flat_parts[0]
    }


def check_metric_rows(metrics: dict, update: int) -> None:
    """Every metric must have one row per completed update.

    Catches truncated or empty metrics at save time, and checkpoints written
    by an older version that dropped them, before they cause a confusing
    failure at the end of a long run.
    """
    if update == 0:
        return
    bad = [name for name, values in metrics.items() if len(values) != update]
    if not metrics or bad:
        detail = "it has no metrics" if not metrics else f"mismatched: {', '.join(sorted(bad)[:3])}"
        raise ValueError(
            f"Metrics do not cover all {update} updates ({detail}). A checkpoint "
            "like this was probably written by an older checkpoint.py that "
            "dropped the metrics; it cannot be resumed with a complete history. "
            "Update checkpoint.py and start a new run."
        )


def checkpoint_name(update: int) -> str:
    return f"update-{update:06d}"


def list_checkpoints(output) -> list[Path]:
    """Complete checkpoints, oldest first. Temporary directories are ignored."""
    root = Path(output) / CHECKPOINT_DIRNAME
    if not root.is_dir():
        return []
    found = [
        p for p in root.iterdir()
        if p.is_dir() and _NAME.match(p.name) and (p / META_FILE).is_file()
    ]
    return sorted(found, key=lambda p: int(_NAME.match(p.name).group(1)))


def latest_checkpoint(output) -> Path | None:
    found = list_checkpoints(output)
    return found[-1] if found else None


def _write_synced(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_state(path: Path, state) -> None:
    """Store every array in the state, in tree-leaf order.

    Flax's msgpack serializer cannot handle the simulator's environment state
    (it contains plain Agent objects), so the leaves are saved directly and
    poured back into a template's structure on load.
    """
    import jax

    leaves = jax.tree.leaves(jax.device_get(state))
    arrays = {f"leaf_{i:05d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _read_state(path: Path, template):
    import jax
    import jax.numpy as jnp

    ref_leaves, treedef = jax.tree.flatten(template)
    with np.load(path) as saved:
        if len(saved.files) != len(ref_leaves):
            raise ValueError(
                f"Checkpoint holds {len(saved.files)} arrays but the current "
                f"model/environment expects {len(ref_leaves)}."
            )
        leaves = []
        for i, ref in enumerate(ref_leaves):
            array = saved[f"leaf_{i:05d}"]
            if array.shape != jnp.shape(ref) or array.dtype != jnp.result_type(ref):
                raise ValueError(
                    f"Checkpoint array {i} is {array.dtype}{array.shape}, expected "
                    f"{jnp.result_type(ref)}{jnp.shape(ref)}."
                )
            leaves.append(jnp.asarray(array))
    return jax.tree.unflatten(treedef, leaves)


def save_checkpoint(output, state, metrics_parts, update, config,
                    algorithm_source, keep_last=0) -> Path:
    """Save the full runner state after `update` updates; prune old ones.

    `keep_last` of 0 keeps every checkpoint. The most recent `keep_last` are
    kept otherwise.
    """
    metrics = concat_metrics(metrics_parts)
    check_metric_rows(metrics, update)  # before anything is written
    root = Path(output) / CHECKPOINT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    final = root / checkpoint_name(update)
    temp = root / f"{checkpoint_name(update)}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()

    _write_state(temp / STATE_FILE, state)
    np.savez_compressed(temp / METRICS_FILE, **metrics)
    meta = {
        "format_version": FORMAT_VERSION,
        "update": int(update),
        "config_hash": config_hash(config),
        "algorithm": config.get("ALGORITHM"),
        "algorithm_source_hash": file_hash(algorithm_source),
    }
    # meta.json is written last: its presence marks the checkpoint as complete.
    _write_synced(temp / META_FILE, (json.dumps(meta, indent=2) + "\n").encode())

    if final.exists():
        shutil.rmtree(final)
    os.replace(temp, final)

    if keep_last > 0:
        for old in list_checkpoints(output)[:-keep_last]:
            shutil.rmtree(old)
    return final


def load_checkpoint(path, template, config, algorithm_source):
    """Restore (state, metrics, update) from a checkpoint directory.

    `template` is a freshly initialised runner state; it supplies the
    structure, shapes and dtypes the saved arrays are poured into. Raises ValueError
    if the checkpoint was made with different training settings or a
    different algorithm file.
    """
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {meta.get('format_version')}")
    if meta["config_hash"] != config_hash(config):
        raise ValueError(
            "Config does not match the checkpoint's. Resume needs identical "
            "training settings; only the interval/eval/logging keys may change."
        )
    if meta["algorithm_source_hash"] != file_hash(algorithm_source):
        raise ValueError(
            "The algorithm source file changed since this checkpoint was saved."
        )
    state = _read_state(path / STATE_FILE, template)
    with np.load(path / METRICS_FILE) as saved:
        metrics = {name: saved[name] for name in saved.files}
    check_metric_rows(metrics, int(meta["update"]))
    return state, metrics, int(meta["update"])
