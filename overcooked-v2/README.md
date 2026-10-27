# OvercookedV2 training and setup

## Local Overcooked! 2 scenario prototype

`bash overcooked-v2/run.sh scenario --layout oc2_sushi_1_3` (from the repository
root) runs a JAX-native grid prototype with chopping, movable pots, sushi assembly,
and a finite dirty/clean plate cycle. A two-chef scripted check serves two dishes
and reuses a washed plate, saving an offline replay. No training is launched.
The commercial-game map and several timings remain provisional, and the PPO
launcher is not yet connected to this scenario. Existing maps/JaxMARL stay unchanged.
See [the scenario guide](scenarios/oc2_sushi_1_3/README.md) for CPU setup, actions,
observations, assumptions, and checks.

Source: [JaxMARL](https://github.com/bold-lab-ai/JaxMARL), tracked as a pinned Git submodule. Its commit is also recorded in `UPSTREAM_REVISION`.
The upstream simulator and PPO algorithm are unchanged. Our scripts provide repeatable local checks and a saved baseline training launcher.

## Training options

- **Local NVIDIA GPU:** users with a compatible NVIDIA GPU can train themselves on Linux or Windows through WSL2 using `setup.sh` and the existing launchers. The checks below were verified on an RTX 3070 in WSL2; full-budget training runtime and memory use remain unverified.
- **Modal cloud GPU:** users can choose Modal instead of a local GPU. Modal integration and setup instructions are pending; the commands below currently cover local training only.

The intended Modal path should reuse the local path's pinned dependencies, baseline settings, and evaluation protocol, and persist saved weights and results.

## Current status (2026-09-30)

- Installation completed; `pip check` passed. GPU simulator check completed 400 steps and produced `outputs/smoke/random-rollout.gif`.
- The upstream recurrent PPO setup check passed on the GPU: 2 updates, 512 environment steps, and 16 optimizer steps. Saved weights were reloaded and verified with model inference; parameters, predictions, and metrics were finite.
- The saved checkpoint is a setup artifact, too briefly trained to be a useful baseline. Research intervention and evaluation design remain undecided. Further experiments are user-led.
- Frozen-policy playback is verified on the GPU. The first seed-42 untrained reference completed five 400-step episodes (test seeds 1000-1004): zero deliveries and zero raw reward. Scores, initial weights, and an annotated GIF are in `outputs/evaluate/coord_ring-untrained-20261001T004631182874Z/`. No research training has been launched.
- Coordination Ring paper-aligned single-seed training is prepared in `train.py` and `configs/coord_ring.yaml`; its dry-run and synthetic model initialization/save/reload checks have passed on the GPU. No baseline training has been launched.

## Coordination Ring baseline

`configs/coord_ring.yaml` records the paper's partially observable original-layout
experiment: Appendix C.1/C.2 and Appendix D.2.1, Table 4 of
[the OvercookedV2 paper](https://arxiv.org/abs/2503.17821).
This is **Coordination Ring (`coord_ring`)**, distinct from Grounded Coordination Ring
(`grounded_coord_ring`). The algorithm and simulator remain pinned, unmodified upstream code.

Environment:

- Two chefs, local view radius 2 (5 x 5 cells; 30 numerical channels).
- Random starting positions and directions; standard empty inventories/pots.
- One possible recipe: three onions. No recipe indicator or recipe-reader role.
- Negative delivery rewards enabled, as described in C.1. This single-recipe layout has no alternative wrong recipe.
- 400-step episodes, automatic cooking once three ingredients are added, and the default numerical grid observation. These are pinned simulator defaults, explicitly recorded in our configuration.
- No full starting-state augmentation, ingredient permutations, or added communication channel.

Training:

- Parameter-sharing recurrent IPPO in self-play; CNN + 128-unit GRU, matching C.1.1.
- 10,000,000 requested environment steps; the upstream implementation completes 152 full updates / 9,961,472 steps.
- 256 parallel kitchens, 256 steps per rollout, 64 minibatches, 4 optimization epochs.
- Learning rate 0.0005, 5% warm-up followed by cosine decay; shaping bonuses fade to zero by 5,000,000 steps.
- Remaining hyperparameters are explicit in the YAML. The per-minibatch size is 2,048 agent transitions, derived from two chefs per kitchen.
- This launcher runs **one seed** (default 42); the paper's Figure 14 aggregates ten seeds. Our first run is not a ten-seed reproduction.

From the repository root in PowerShell, first inspect the settings without playing or training:

```powershell
.\overcooked-v2\run.ps1 train -DryRun
```

To start the baseline yourself:

```powershell
.\overcooked-v2\run.ps1 train
```

Keep the terminal open. Compilation happens before update messages appear. The command
prints its results directory and saves configuration, version/source metadata, and
per-update progress under a new `outputs/train/coord_ring-seed42-<timestamp>/` directory.
On successful completion it saves `params.msgpack`, numeric `metrics.npz`, and
`summary.json`, and verifies that the checkpoint reloads. Existing results are not overwritten.

The checkpoint is saved **only on completion**. Ctrl+C stops the run; partial progress
logs remain, but interrupted runs cannot resume with this launcher. Runtime and full-budget
GPU memory requirements have not yet been measured. Training statistics describe games played while the weights were changing. Use the
separate frozen-policy evaluation below to measure the finished policy.

To run another independent seed later, use `train -Seed 43`. Seeds are local run choices;
the paper's exact seed sequence has not been reproduced. External logging is disabled.
Linux equivalent: `bash overcooked-v2/run.sh train --seed 42`.

## Before/after policy evaluation and replays

Training and evaluation are separate commands. Evaluation freezes the policy's weights:
it plays games and records results, without updating anything. Training saves only the
final weights, not an optimizer/resume checkpoint, and does not automatically create a GIF.

Before training, record the freshly initialized **neural policy** (not the random-action
setup check):

```powershell
.\overcooked-v2\run.ps1 evaluate -Untrained
```

This plays five 400-step Coordination Ring episodes. Initialization seed 42 matches the
baseline trainer's starting weights. Test seeds 1000-1004 are separate from the training
seed. Both chefs share the policy but have separate recurrent memory, reset each episode.
Actions are sampled from the policy's probabilities, as during training.

After a training run finishes, evaluate its saved weights:

```powershell
.\overcooked-v2\run.ps1 evaluate -Run ".\overcooked-v2\outputs\train\<run-folder>"
```

Replace `<run-folder>` with the directory printed by training. This reads that run's
`config.json` and `params.msgpack`, checks the pinned source revision, and uses the same
five test seeds and action-sampling rule. Compare runs with matching environment settings.
The same seeds align starting kitchens and random-number streams; the chefs' decisions
and resulting trajectories will differ after learning.

Each evaluation creates a new folder under `outputs/evaluate/` containing:

- `episodes.csv`: raw team reward, correct/wrong deliveries, steps, and seed per episode.
- `summary.json`: averages and total deliveries across the episodes.
- `config.json` and `protocol.json`: environment/model settings, test seeds, policy hash,
  source revision, and the recorded episode selection. An untrained evaluation also
  saves its actual initial weights in `params.msgpack`.
- `replay.html`: a self-contained illustrated browser replay with every simulator
  step, chef positions/facing/inventories/actions, pot status, delivery jumps, and
  raw team score. **Illustrated / Original** compares the same recorded state.
- `rollout.gif`, `initial.png`, and `final.png`: the existing annotated kitchen previews.

Open `replay.html` directly in a browser on Windows, Linux, or macOS. It starts paused
at step 0 and works offline; viewers need no simulator environment or GPU. Drag the
timeline backward or forward, use the step buttons or enter a step number, and select
0.25x, 0.5x, 1x, or 2x speed. Pot overlays show the remaining cooking steps, or
Ready/Waiting. A grid toggle shows exact cell coordinates. Space plays/pauses; left/right arrows move one step when
you are not editing a control. Seeking or stepping pauses playback; it stops at step
the final recorded step, and Replay starts it again. At 1x each upstream step lasts
100 ms, a display speed rather than measured simulator runtime. The sushi scenario
uses its configured step duration. Layout dimensions, recipes, steps, pots, and
delivery totals come from the episode rather than the 18-soup reference.

There are 401 states: the initial kitchen plus the result of all 400 joint actions.
The displayed action is the action that produced that state, so step 0 has no previous
action. Chef 0 is red (`agent_0`), and chef 1 is blue (`agent_1`); positions are
zero-based `(column, row)`. The HTML embeds a versioned JSON payload with metadata,
state/action/event data, and PNG frames, separate from its playback UI. Future panels
can consume additional recorded information. The current policy does not produce
explicit task plans.

The shared viewer supports upstream layouts (including recipe indicators/switches
and numbered ingredient IDs) and the sushi prototype (partial assembly, movable
rice pots, chopping/washing progress, and dirty/clean plates). Other ingredient IDs
use labelled pouches rather than inventing food names. Unsupported artwork falls
back to original recorded frames with an explanation. GIF/PNG previews retain their
original rendering; the new illustration is in the HTML spectator view.

To restyle an existing saved HTML replay without simulation, evaluation, or training,
run from the repository root:

```bash
python3 overcooked-v2/replay.py path/to/replay.html
# Or upgrade all replay.html files under a directory:
python3 overcooked-v2/replay.py overcooked-v2/outputs/evaluate
```

This replaces the HTML UI in place while preserving recorded metadata, states,
actions, totals, and original PNG frames. The restyling command uses standard-library
Python only. [Artwork and renderer notes](assets/replay/README.md).

The HTML replay and GIF always show the **first** test episode. The HTML retains every
step; the GIF has one frame per two steps. Their score
is for that episode, not the five-episode average. The dashboard is for spectators and
adds nothing to the policy's observations. Correct and wrong dishes are counted as
separate delivery events, including simultaneous deliveries. The shared team reward is
counted once and excludes training's shaping bonuses. On this layout a correct dish
is worth 20 points; five episodes are a small before/after check, not a paper-scale benchmark.

Optional controls: `evaluate -Untrained -Seed 43` selects another initialization;
`-Episodes 10 -EvalSeed 1000` changes test count/seeds for either policy. Apply identical
evaluation choices to both sides of a comparison. Keep the first five-episode results as
the initial reference rather than choosing a particularly good or bad GIF later.
Linux equivalents: `bash overcooked-v2/run.sh evaluate --untrained`, then
`bash overcooked-v2/run.sh evaluate --run outputs/train/<run-folder>` (the latter path is
relative to `overcooked-v2`, since `run.sh` sets that working directory).

Replay data checks, using the installed simulator Python environment from this directory:
`python -m unittest discover -s tests -v`. These check state/action alignment, delivery
totals, pot status, the self-contained HTML payload, and the saved scripted replay
against its full audit when that replay is available. No training is performed.

## Scripted 18-soup reference

The saved fixed-start Coordination Ring schedule delivers 18 soups (360 raw team
reward), with no chef overlaps. Verify it and generate its HTML replay and annotated GIF from the repository root:

```powershell
.\overcooked-v2\run.ps1 scripted
```

Linux equivalent: `bash overcooked-v2/run.sh scripted`. Files are saved under
`docs/theoretical-maximum/media/18-soups/`. Open `replay.html` there to pause or inspect
any step. This runs saved joint actions without training. See
[the maximum-score analysis](../docs/theoretical-maximum/theoretical-maximum.md) for the replay, full schedule,
and 19-soup search results. The exact optimum remains 18 or 19; 18 is verified achievable.

## Run setup checks from PowerShell

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

## Setup-check scenario

`grounded_coord_simple` uses two agents, a view radius of 2, random starting positions, penalties for incorrect deliveries, and a newly sampled recipe after each correct delivery. These settings match the upstream baseline's environment configuration. The training check reduces its compute budget; it is not a paper reproduction.

This remains the default scenario for setup checks. The user-led research baseline launch above uses Coordination Ring instead.
