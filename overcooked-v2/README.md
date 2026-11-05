# OvercookedV2 training and setup

## Setup

Use Python 3.12 and Git on Linux or macOS. The macOS setup uses CPU; the pinned CUDA
environment is for Linux machines with a compatible NVIDIA GPU.
From the repository root:

```bash
bash overcooked-v2/setup.sh
```

A fresh checkout defaults to a CPU environment in `overcooked-v2/.venv/`. To explicitly install the
pinned Linux/NVIDIA CUDA 12 environment, use `bash overcooked-v2/setup.sh --gpu`.
`--cpu` explicitly selects the CPU dependencies, including when local settings select GPU.
Setup initializes only the pinned JaxMARL submodule, installs dependencies, and checks
package compatibility. It does not play games or train a policy.

Run commands below through `bash overcooked-v2/run.sh <mode> ...` from the repository root.

Machine settings are optional and ignored by Git:

- `.local/overcooked.env`: Bash settings loaded by `setup.sh` and `run.sh`; use `OVC_VENV`
  for an existing virtual environment and `OVC_REQUIREMENTS` for a chosen requirements file.
  Runtime variables such as `JAX_PLATFORMS` and `XLA_FLAGS` can also be set here.
- `.local/AGENTS.md` and other notes: personal context, device details, troubleshooting,
  and references to locally generated results.

Do not put machine paths, driver details, or individual run histories in shared files.
Experiment settings belong in `configs/`; outputs contain each run's settings and metadata.

## Local Overcooked! 2 scenario prototype

`bash overcooked-v2/run.sh scenario --layout oc2_sushi_1_3` (from the repository
root) runs a JAX-native grid prototype with chopping, movable pots, sushi assembly,
and a finite dirty/clean plate cycle. A two-chef scripted check serves two dishes
and reuses a washed plate, saving an offline replay. No training is launched.
The commercial-game map and several timings remain provisional. The PPO launcher
also accepts this scenario; the small check below verifies the hookup. Existing
maps/JaxMARL stay unchanged.
See [the scenario guide](scenarios/oc2_sushi_1_3/README.md) for CPU setup, actions,
observations, assumptions, and checks.

### Learned sushi policy

A separate behavior-cloning workflow now learns from audited demonstrations.
The latest frozen v8 policy served seven dishes in all 20 standard-start,
600-step episodes; mean dishes were 6.40 with 5% action delays and 6.30 with
changed initial poses. This is fixed-layout simulator performance. The complete
frozen v8 model is included under `model/oc2_sushi_1_3/policy-spawn-v8/` at the
repository root; raw training runs and datasets remain local ignored artifacts.
See the [model guide](../model/README.md) for CPU inference and file integrity,
or the [sushi learning guide](../docs/sushi-training.md) for the method, results
and limitations.
The tiny PPO check below remains a setup check, distinct from these results.

### Tiny sushi PPO check

With the full PPO dependencies installed, run from `overcooked-v2/`:

```bash
JAX_PLATFORMS=cpu .venv/bin/python train.py --config configs/oc2_sushi_check.yaml
JAX_PLATFORMS=cpu .venv/bin/python evaluate.py --run outputs/train/<printed-run-folder> --episodes 1
```

The default setup installs the full CPU PPO dependencies. The standalone
`requirements-scenario-cpu.txt` environment alone is insufficient for PPO.
Use `setup.sh --gpu` for the pinned Linux/NVIDIA dependencies.

The check runs two kitchens for two updates: **32 environment steps**, with
eight-step episodes to exercise resets. It uses a smaller recurrent/dense size,
the existing CNN/PPO code, and zero added reward shaping. Checkpoints, reload
verification, and evaluation outputs use the standard paths. This budget cannot
produce a useful policy; the check config is not a research baseline. The first
CPU run took about 20 seconds including compilation, saved/reloaded its weights,
and produced an eight-step evaluation replay with zero deliveries.
Full-state checkpoint resumes still require identical algorithm source; older
checkpoints need their original code revision to resume.

### Experimental Mac GPU check

The M5 Pro check uses **jax-metal 0.1.1 with JAX/jaxlib 0.4.38** in a separate
environment. Apple's plugin is [experimental](https://developer.apple.com/metal/jax/).
It loses boolean outputs stored by compiled loops on this tested setup, lacks QR
initialization, and cannot run our logging callback. `--experimental-metal` opts
in to workarounds: store trajectory flags as integers inside the loop, use a
numeric recurrent reset mask so its saved gradients avoid boolean loop buffers,
initialize on CPU before transferring to GPU, and log returned metrics from
Python. PPO loss/update math and the default CPU/CUDA path stay unchanged.

From `overcooked-v2/`, using Python 3.12:

```bash
python3.12 -m venv .venv-metal
.venv-metal/bin/python -m pip install -r requirements-metal.txt
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-metal/bin/python train.py --config configs/oc2_sushi_check.yaml --experimental-metal
```

Keep CPU available for initialization. The opt-in is saved in `config.json` and
reused by `--resume`. Use the existing CPU environment for `evaluate.py`; GPU
evaluation has not been adapted. Run the focused GPU correctness checks with:

```bash
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
  .venv-metal/bin/python -m unittest tests.test_metal_compatibility -v
```

These compare two GPU updates against CPU, including episode boundaries inside
rollouts, and check exact checkpoint/resume recovery. Short M5 Pro benchmarks
also passed CPU agreement with these warmed update timings:

| Kitchens × rollout steps | CPU/update | Metal/update | GPU speedup |
| --- | ---: | ---: | ---: |
| 16 × 64 | 0.832 s | 0.471 s | 1.77× |
| 64 × 64 | 6.425 s | 1.239 s | 5.19× |

Each result is the median of five interleaved, synchronized samples after three
untimed calls, repeating a fixed initial state. Both use the same CNN with
32-wide recurrent/dense layers, one PPO epoch, two minibatches, and 32-step
episodes. Compilation and checkpoint/file logging are excluded. The original
two-kitchen, eight-step check favored CPU (7.6 ms versus 43.3 ms), so batch size
matters. These checks establish neither long-run stability nor full research
performance; other layouts and GPU evaluation remain unverified.

Source: [JaxMARL](https://github.com/bold-lab-ai/JaxMARL), tracked as a pinned Git submodule. Its commit is also recorded in `UPSTREAM_REVISION`.
The pinned upstream simulator is unchanged. Our trainer preserves the PPO update
math and adds local setup, checkpointing, and optional Metal compatibility.

## Training options

- **Local NVIDIA GPU:** users with a compatible NVIDIA GPU can train themselves on Linux or Windows through WSL2 using `setup.sh` and the existing launchers. The checks below were verified on an RTX 3070 in WSL2; full-budget training runtime and memory use remain unverified.
- **Modal cloud GPU:** users can choose Modal instead of a local GPU. Modal integration and setup instructions are pending; the commands below currently cover local training only.

The intended Modal path should reuse the local path's pinned dependencies, baseline settings, and evaluation protocol, and persist saved weights and results.

## Coordination Ring baseline

`configs/coord_ring.yaml` records the paper's partially observable original-layout
experiment: Appendix C.1/C.2 and Appendix D.2.1, Table 4 of
[the OvercookedV2 paper](https://arxiv.org/abs/2503.17821).
This is **Coordination Ring (`coord_ring`)**, distinct from Grounded Coordination Ring
(`grounded_coord_ring`). The simulator remains pinned upstream code. The local
PPO wrapper adds chunked checkpoints and optional GRU unrolling to the upstream algorithm.

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

An opt-in CUDA graph configuration preserves the baseline environment, model,
PPO settings, and checkpoint interval while enabling GPU command buffers:

```powershell
.\overcooked-v2\run.ps1 train -CudaGraphs -DryRun
.\overcooked-v2\run.ps1 train -CudaGraphs
```

Linux equivalent: `bash overcooked-v2/run.sh train --config configs/coord_ring_cuda_graphs.yaml`.
The ordinary training command keeps its existing runtime settings. See the
[performance experiment guide](benchmarks/README.md) for measurements and
correctness/learning comparisons. A fusion-only configuration is also available
at `configs/coord_ring_fusion.yaml`.

GRU scan unrolling is also opt-in. `-GruUnroll` uses a factor of 16 and can be
combined with CUDA graphs:

```powershell
.\overcooked-v2\run.ps1 train -GruUnroll -DryRun
.\overcooked-v2\run.ps1 train -CudaGraphs -GruUnroll
```

Linux equivalent: `bash overcooked-v2/run.sh train --config configs/coord_ring_cuda_graphs.yaml --gru-unroll`.
Omit the graph configuration to use unrolling alone. The default unroll factor
remains 1. The selected factor is saved as `GRU_UNROLL` in `config.json` and
automatically reused by `train.py --resume <run-folder>`. Resume checks reject a
different factor. It preserves the network and PPO equations, but can change
floating-point results and learned trajectories; the short learning comparison
does not establish equal final effectiveness. Existing full-state checkpoints
retain their algorithm-source check and need their original code revision to
resume after algorithm edits; their policy weights remain usable for evaluation.

Keep the terminal open. Compilation happens before update messages appear. The command
prints its results directory and saves configuration, version/source metadata, and
per-update progress under a new `outputs/train/coord_ring-seed42-<timestamp>/` directory.
On successful completion it saves `params.msgpack`, numeric `metrics.npz`, and
`summary.json`, and verifies that the checkpoint reloads. Existing results are not overwritten.

Full-state checkpoints are saved every ten updates and at completion. Ctrl+C stops
the run; resume from the latest complete checkpoint with
`bash overcooked-v2/run.sh train --resume outputs/train/<run-folder>`.
Saved settings, including GRU unrolling, are reused automatically. Training
statistics describe games played while the weights were changing. Use the
separate frozen-policy evaluation below to measure the finished policy.

To run another independent seed later, use `train -Seed 43`. Seeds are local run choices;
the paper's exact seed sequence has not been reproduced. External logging is disabled.
Linux equivalent: `bash overcooked-v2/run.sh train --seed 42`.

## Before/after policy evaluation and replays

Training and evaluation are separate commands. Evaluation freezes the policy's weights:
it plays games and records results, without updating anything. Training saves
final policy weights and periodic full-state checkpoints, and does not automatically create a GIF.

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
- `evaluation.json`: portable, uploadable evidence with every episode's recorded
  states, exact checkpoint assignments and provenance; no weights or PNG frames.
  Export size is printed; the sharing website accepts at most 20 MiB. See the
  [format/export guide](../evals-sharing-website/docs/format-and-export.md), including
  conversion of existing outputs without rerunning evaluation. Local previews
  continue to show the first episode.
- `summary.json`: averages and total deliveries across the episodes.
- `config.json` and `protocol.json`: environment/model settings, test seeds, policy hash,
  source revision, and the recorded episode selection. An untrained evaluation also
  saves its actual initial weights in `params.msgpack`.
- `replay.html`: a self-contained illustrated browser replay with every simulator
  step, chef positions/facing/inventories/actions, pot status, delivery jumps, and
  raw team score. **Illustrated / Original** compares the same recorded state.
- `rollout.gif`, `initial.png`, and `final.png`: the existing annotated kitchen previews.
- `decisions-000.npz`, etc.: the pre-action probabilities, executed actions, agent
  IDs, and evaluation seed for each episode. Upstream-layout GIFs include action
  probabilities and milestone trajectories.

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
step; upstream-layout GIFs also retain every step, while sushi GIFs use one frame
per two steps. Their score
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

## Cross-policy evaluation

Use `--partner-run` to assign a second saved policy to chef 2 (blue). `--run` controls
chef 1 (red). Omitting `--partner-run` keeps both chefs on the `--run` policy.
Each chef has separate recurrent memory, reset at the start of every episode; no
weights are updated. Both checkpoints must use the pinned source revision, matching
environment settings, and matching network architecture. Training seeds and training
hyperparameters may differ.

From the repository root, set paths relative to `overcooked-v2/`:

```bash
run_a="outputs/train/<first-run-folder>"
run_b="outputs/train/<second-run-folder>"
bash overcooked-v2/run.sh evaluate --run "$run_a" --episodes 50 --eval-seed 2000
bash overcooked-v2/run.sh evaluate --run "$run_b" --episodes 50 --eval-seed 2000
bash overcooked-v2/run.sh evaluate --run "$run_a" --partner-run "$run_b" --episodes 50 --eval-seed 2000
bash overcooked-v2/run.sh evaluate --run "$run_b" --partner-run "$run_a" --episodes 50 --eval-seed 2000
```

These are A+A, B+B, A+B, and B+A. The same test seeds align initial kitchens and action
random-number streams across conditions; different policies produce different decisions.
Test both mixed orders because assignment to chefs can matter.

Cross-policy results use a new `cross-policy` output folder. `protocol.json` identifies
both checkpoints by source path, training seed, hash, and chef/color assignment;
`partner-config.json` preserves the second policy's settings. The Coordination Ring GIF identifies each
chef's policy seed and still shows only the first episode. Scores remain shared team
scores, recorded once. If using the local PowerShell bridge, its equivalent option is
`-PartnerRun <second-training-directory>`.

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

## Dependencies and reproducibility

`requirements.txt` pins the shared CPU dependencies. `requirements-gpu.lock.txt` pins
the resolved Linux CUDA 12 environment and is selected explicitly with `setup.sh --gpu`.
The simulator version is pinned separately through the Git submodule and `UPSTREAM_REVISION`.
The launchers use an existing environment at `OVC_VENV` when supplied; otherwise they use
`overcooked-v2/.venv/`. Existing virtual environments and generated outputs are ignored.

Runtime workarounds for an individual device belong in `.local/overcooked.env`.
Training records device/backend details in each output folder so environment differences
can be investigated without adding a particular machine's configuration to the repository.

When pulling a change that updates the dependency, run `git submodule update --init` from the repository root. Do not update JaxMARL to its latest branch accidentally; changing the pinned revision is a separate project decision.

## Setup-check scenario

`grounded_coord_simple` uses two agents, a view radius of 2, random starting positions, penalties for incorrect deliveries, and a newly sampled recipe after each correct delivery. These settings match the upstream baseline's environment configuration. The training check reduces its compute budget; it is not a paper reproduction.

This remains the default scenario for setup checks. The user-led research baseline launch above uses Coordination Ring instead.
