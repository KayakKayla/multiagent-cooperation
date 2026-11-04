"""Run overcooked-v2 training and evaluation on Modal.

Place this file at the repository root (next to the `overcooked-v2/` folder).
Before the first run, make sure the JaxMARL submodule is checked out locally:

    git submodule update --init

Usage (from the repository root):

    # Train; --detach keeps the run alive if you close your terminal
    modal run --detach modal_app.py::train_main --config overcooked-v2/configs/coord_ring.yaml --seed 42

    # Check settings only
    modal run modal_app.py::train_main --dry-run

    # Resume an interrupted run (folder name as listed in the volume)
    modal run --detach modal_app.py::train_main --resume coord_ring-seed42-<timestamp>

    # Evaluate a finished (or checkpointed) run; writes replay.html and rollout.gif
    modal run modal_app.py::eval_main --run coord_ring-seed42-<timestamp>

    # Browse / download results
    modal volume ls overcooked-outputs train/
    modal volume get overcooked-outputs train/<run-folder> ./local-results/
    modal volume get overcooked-outputs evaluate/<eval-folder> ./local-results/
"""
import subprocess
import threading
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent
REMOTE_PROJECT = "/app/overcooked-v2"
REMOTE_OUTPUTS = f"{REMOTE_PROJECT}/outputs"

#GPU = "L4"  # any of "T4", "L4", "A10G", "A100" works; 8 GB (RTX 3070) was enough locally
GPU = "T4"
COMMIT_EVERY_SECONDS = 60

app = modal.App("overcooked-v2")
vol = modal.Volume.from_name("overcooked-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Strip the editable "-e ./JaxMARL[algs]" line (and comments) at build time.
    .add_local_file(str(PROJECT / "requirements-gpu.lock.txt"), "/tmp/lock.txt", copy=True)
    .run_commands(
        "grep -vE '^(-|#)' /tmp/lock.txt > /tmp/pins.txt",
        "pip install -r /tmp/pins.txt",
    )
    .env({
        "PYTHONPATH": REMOTE_PROJECT,
        "JAX_PLATFORMS": "cuda,cpu",
    })
    .add_local_dir(
        str(PROJECT),
        REMOTE_PROJECT,
        copy=True,
        ignore=["outputs/**", "**/__pycache__/**", "**/.git", "**/.venv/**"],
    )
    .run_commands(f"pip install --no-deps -e {REMOTE_PROJECT}/JaxMARL")
)


def _run_with_commits(cmd: list[str]) -> None:
    """Run a command, streaming output, and commit the volume periodically.

    Periodic commits mean checkpoints survive a crash or timeout, so the run can
    be restarted with --resume.
    """
    stop = threading.Event()

    def committer():
        while not stop.wait(COMMIT_EVERY_SECONDS):
            try:
                vol.commit()
            except Exception as exc:  # keep training even if a commit fails
                print(f"[modal] volume commit failed: {exc}", flush=True)

    thread = threading.Thread(target=committer, daemon=True)
    thread.start()
    try:
        subprocess.run(cmd, cwd=REMOTE_PROJECT, check=True)
    finally:
        stop.set()
        thread.join()
        vol.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={REMOTE_OUTPUTS: vol},
    timeout= 2 * 60 * 60,  # raise if the full 10M-step run needs longer (24h max)
)
def train(config_yaml: str | None = None, seed: int | None = None,
          resume: str | None = None, dry_run: bool = False) -> None:
    cmd = ["python", "train.py"]
    if resume:
        # The run's own config.json and RNG state are used; --seed is not allowed.
        cmd += ["--resume", f"{REMOTE_OUTPUTS}/train/{resume}"]
    else:
        if config_yaml is not None:
            Path("/tmp/config.yaml").write_text(config_yaml)
            cmd += ["--config", "/tmp/config.yaml"]
        if seed is not None:
            cmd += ["--seed", str(seed)]
    if dry_run:
        cmd.append("--dry-run")
    print("Running:", " ".join(cmd), flush=True)
    _run_with_commits(cmd)


@app.function(
    image=image,
    gpu=GPU,
    volumes={REMOTE_OUTPUTS: vol},
    timeout=60 * 60,
)
def evaluate(run: str | None = None, untrained: bool = False,
             episodes: int | None = None, eval_seed: int | None = None) -> None:
    # Pick up checkpoints/results committed by the training container.
    vol.reload()
    cmd = ["python", "evaluate.py"]
    if untrained:
        cmd.append("--untrained")
    else:
        if not run:
            raise ValueError("Pass --run <run-folder> or --untrained.")
        cmd += ["--run", f"outputs/train/{run}"]  # relative to overcooked-v2, like run.sh
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    if eval_seed is not None:
        cmd += ["--eval-seed", str(eval_seed)]
    print("Running:", " ".join(cmd), flush=True)
    _run_with_commits(cmd)


@app.local_entrypoint()
def train_main(config: str = "", seed: int = -1, resume: str = "", dry_run: bool = False):
    config_yaml = Path(config).read_text() if config else None
    train.remote(
        config_yaml=config_yaml,
        seed=None if seed < 0 else seed,
        resume=resume or None,
        dry_run=dry_run,
    )


@app.local_entrypoint()
def eval_main(run: str = "", untrained: bool = False, episodes: int = -1, eval_seed: int = -1):
    evaluate.remote(
        run=run or None,
        untrained=untrained,
        episodes=None if episodes < 0 else episodes,
        eval_seed=None if eval_seed < 0 else eval_seed,
    )
