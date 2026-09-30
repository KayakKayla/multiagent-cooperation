# OvercookedV2 local setup

Source: [JaxMARL](https://github.com/bold-lab-ai/JaxMARL), tracked as a pinned Git submodule. Its commit is also recorded in `UPSTREAM_REVISION`.
The upstream simulator and PPO algorithm are unchanged. Our scripts provide repeatable local checks.

## Current status (2026-09-30)

- Installation completed; `pip check` passed. GPU simulator check completed 400 steps and produced `outputs/smoke/random-rollout.gif`.
- The upstream recurrent PPO setup check passed on the GPU: 2 updates, 512 environment steps, and 16 optimizer steps. Saved weights were reloaded and verified with model inference; parameters, predictions, and metrics were finite.
- The saved checkpoint is a setup artifact, too briefly trained to be a useful baseline. Research intervention and evaluation design remain undecided. Further experiments are user-led.

## Run from PowerShell

From the `multiagent-cooperation` repository root:

```powershell
.\overcooked-v2\run.ps1 smoke
.\overcooked-v2\run.ps1 train-check
```

- `smoke`: runs one 400-step episode with random actions, checks observations, and saves a GIF, initial image, and JSON summary under `outputs/smoke/`. The image shows the full kitchen for the observer; each agent receives its own restricted numerical observation.
- `train-check`: runs 2 updates of the upstream recurrent PPO baseline (512 environment steps across 8 parallel environments). Saves weights (`params.msgpack`), configuration, numeric metrics, and a summary under `outputs/train-check/`. It reloads the weights and checks model inference. This budget does not even complete a training episode; it only checks that the training machinery works.
- Both commands use seed 0 and `grounded_coord_simple`. Use `-Layout <name>` to choose another built-in layout. Re-running a command overwrites its check outputs.

Each new process compiles JAX operations before running. The successful training check took about 26 seconds for compilation and training on this device, plus startup and saving. W&B external logging is disabled; no account is required. No process continues training after either command exits.

## Runtime

- Windows 10 hosts Ubuntu-24.04 in WSL2.
- GPU: NVIDIA RTX 3070, 8 GB; Windows driver 560.94.
- Isolated Linux Python 3.12 environment: `/home/ro/.venvs/se390-overcooked-v2`.
- JAX uses CUDA 12 packages inside that environment. The existing Windows driver and Anaconda environments are unchanged.
- `run.sh` defaults to `JAX_PLATFORMS=cuda,cpu`: GPU computation with CPU availability for upstream logging callbacks. GPU memory preallocation is disabled.

### Local compatibility fixes

- CUDA command buffers are disabled with `--xla_gpu_enable_command_buffer=`. Repeated batched environment steps stalled with the default setting on this WSL/GPU stack; the same probe passed with this flag. Existing unrelated XLA flags are preserved.
- The training check waits for result arrays individually because JAX's combined readiness wait stalled here. The upstream algorithm and simulator remain unchanged.
- Nested per-recipe metrics are flattened to numeric NPZ entries such as `returned_episode_recipe_returns/0_0_0`, avoiding pickled dictionaries.

For CPU-only debugging in Ubuntu: `JAX_PLATFORMS=cpu bash run.sh smoke`.

## Recreate

In Ubuntu, from this directory, run `bash setup.sh`. It initializes the JaxMARL submodule at the repository's pinned commit, verifies `UPSTREAM_REVISION`, and installs `requirements-gpu.lock.txt` when present. Python 3.12, system `pip`, and Git must be available. The virtual environment lives in Linux storage for performance; code and outputs remain in this Windows project folder. CUDA dependencies require several GB of disk space.

`requirements.txt` records the selected direct dependencies; the lock file records all resolved packages. The installed environment and generated outputs are local and ignored by Git. JaxMARL's source is kept in its own repository; our repository tracks its URL and exact commit through `.gitmodules` and the submodule entry. The optional Robotarium submodule inside JaxMARL is not initialized.

When pulling a change that updates the dependency, run `git submodule update --init` from the repository root. Do not update JaxMARL to its latest branch accidentally; changing the pinned revision is a separate project decision.

## Starting scenario

`grounded_coord_simple` uses two agents, a view radius of 2, random starting positions, penalties for incorrect deliveries, and a newly sampled recipe after each correct delivery. These settings match the upstream baseline's environment configuration. The training check reduces its compute budget; it is not a paper reproduction.

The next user-led step is to inspect this scenario. A longer baseline run and evaluation plan can follow once the user is ready.
