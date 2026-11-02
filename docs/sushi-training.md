# Learning to serve sushi on the Story 1-3 simulator

Ongoing findings, training settings and cooperation evidence: [training journal](sushi-training-journal.md). Previous stopping point and restart commands: [current handoff](sushi-training-handoff.md).

This work trains a shared neural policy to serve completed dishes. Comparisons
keep the same empty kitchen, recipe, physical rules and **600-step episode**
(150 simulated seconds). Raw reward is20 per actual delivery; it is not the
commercial game's score. This is the local simulator, not a transferred Steam
policy. Fixed-layout/start performance does not establish unseen-layout skill.

## Latest selected model: v8

The October 28 stopping point is frozen `policy-spawn-v8`: a shared CNN+GRU
with 190,976 parameters, full 10×14×48 observations, and seven primitive actions.
It learned through behavior cloning, not PPO. During evaluation, the neural
policy chooses its own actions without teacher actions or task queues.

V8 was initialized from v7 and trained for 1,500 updates on 57 audited
demonstrations, using fresh Adam with learning rate 0.0001 and gradient clip 1.
The paired final evaluation used seeds 6000–6019: 20 episodes per condition,
600 steps per episode, with matching initial poses and imposed-delay masks.

| Condition | V7 mean dishes | V8 mean dishes |
| --- | ---: | ---: |
| Standard start | 7.00 | 7.00 |
| Independent 5% action delay | 5.75 | 6.40 |
| Radius-one starting positions and random facing | 4.20 | 6.30 |

V8 served seven in every standard episode. One delayed episode and two
changed-start episodes served zero. Two changed-start joint poses matched
training starts; on the other 18, v8 averaged 6.22 dishes versus v7's 3.89.
These are fixed-layout robustness results. Additional training, the changed
data mixture, and warm-start ancestry are confounds; this does not isolate the
effect of pose coverage. Seven is achievable, not a proven maximum.

## Evaluate the frozen policy

Use the full CPU learning dependencies described in the
[setup guide](../overcooked-v2/README.md#tiny-sushi-ppo-check); standalone scenario
dependencies are insufficient. Run the commands below from `overcooked-v2/`
with the submodule initialized and that Python environment active.

The complete frozen model is included in Git at
[`model/oc2_sushi_1_3/policy-spawn-v8/`](../model/oc2_sushi_1_3/policy-spawn-v8/).
No separate model download is needed. It includes weights, config, original
metadata, source snapshots and manifests; the [model guide](../model/README.md)
explains the files and integrity check. Its `params.msgpack` SHA-256 must be:

```text
8d086d8bd3f18e751e08974145dae825f0e18ed4b1edaed93515c7b4e474c46a
```

For one sampled standard-start episode, without learning or weight updates:

```bash
JAX_PLATFORMS=cpu python evaluate_sushi_policy.py \
  --run ../model/oc2_sushi_1_3/policy-spawn-v8 \
  --episodes 1 --eval-seed 6000
```

The printed result directory contains summaries, the evaluation protocol, and
`replay.html`. Outputs use a new timestamped directory. For the recorded 20-seed
standard comparison, use `--episodes 20 --eval-seed 6000`. Add
`--action-dropout 0.05` for delayed actions, or `--spawn-jitter 1 --random-facing`
for the changed-start condition. These seeds reproduce previously used tests;
they are not fresh final-test data for selecting a future model.

## Implementation and verification

- `sushi_imitation.py` learns recurrent action predictions from whole causal
  trajectories, with masked injected waits/padding, action-frequency weighting,
  source/data hashes, and resumable optimizer/RNG state.
- The scenario's demonstration helpers generate real simulator transitions.
  Throughput schedules overlap independent work and prefetch rice; robustness
  helpers vary starts and timing without changing food, resources or timers.
- `evaluate_sushi_policy.py` runs closed-loop CPU inference and checks physical
  deliveries, shared reward, finite predictions, and plate/pot conservation.
- `train.py --init-weights` and `sushi_value_warmup.py` support separate PPO
  trials. Those trials did not improve the selected imitation policy.
- The simulator uses numeric internal event/work buffers to avoid lost boolean
  scatter updates on Metal; its public event fields remain boolean.

Packaging verification on November 1, 2026: **84 CPU tests passed; 11 optional
Metal tests skipped** (95 discovered). The CPU run covers demonstrations,
recurrent loss/masks, exact checkpoint/resume, warm starts, evaluation/replays,
scenario mechanics, and agreement with upstream PPO. Earlier real Metal
action/observation and imitation-numerics checks are recorded in the journal;
they were not rerun during packaging. No research training was launched.

From `overcooked-v2/`, in the CPU learning environment:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

The historical progression follows. V5–v8 and the full experiment chronology
are recorded in the [journal](sushi-training-journal.md) and
[current handoff](sushi-training-handoff.md).

## Previous settings and first useful result

The original plumbing check used a5×5×48 local view, the existing six-layer CNN,
a32-wide GRU and32-wide policy/value heads: **73,216 scalar parameters**. It ran
2 kitchens ×8 steps ×2 PPO updates =32 environment steps, one PPO epoch and two
minibatches, learning rate0.0005, seed0. Episodes ended after8 steps; cooking
alone needs48 steps. It was a setup check and served zero dishes.

The older paper-aligned `coord_ring.yaml` uses128-wide recurrent/dense layers,
256 kitchens ×256 rollout steps, four PPO epochs,64 minibatches, and a10-million
step budget. That research run was never launched and is a different map.

The first competence model keeps the CNN and32-wide GRU/heads but sees the
10×14×48 kitchen, increasing the parameter count to **190,976**. Its actor learns
by behavior cloning from audited simulator-action demonstrations; it is not a
PPO-from-scratch result. The teacher's task queues, action labels and future
states are never inputs during neural evaluation.

## Verified progression

The following sampled numbers are validation results on seeds1000–1009 unless
specified otherwise. Every evaluation is closed-loop with600 actual transitions.
Different seeds vary action sampling, not the fixed reset layout/positions.

| Candidate | Training stage | Greedy dishes | Sampled mean dishes |
| --- | --- | ---: | ---: |
| Untrained full-map reference | seed42 initialization | — |0 across20 initial reference episodes |
| v1 |1000 imitation updates,4 delayed two-dish demonstrations |3 |3.0 validation;2.9 on20 unused seeds2000–2019 |
| v2 |500 additional updates, position-augmented demonstrations | — |3.0 standard validation |
| v3 |1000 updates initialized fromv1, five-dish serial workflow |5 |4.0 |
| v4 |1500 updates initialized fromv3, six-dish concurrent workflow |6 |6.0 |

Each stage starts a fresh optimizer when initialized from a previous model.
Warm-start ancestry is recorded by policy SHA-256. v1 used Adam learning rate
0.001; later imitation stages use0.0003, gradient norm clipping at1.0 and
inverse-square-root class-frequency weighting. Full trajectories preserve each
chef's causal GRU history. Injected arbitrary delays and trailing padding have
zero supervised loss. Current training preserves float32 observations.

The v1 frozen20-seed test served58 dishes: all20 episodes served at least one,
and19 served at least two. Its positive control is a real three-dish rollout;
the untrained reference served none. Its first full-map success was learned
from386-step audited demonstrations, not manually installed action sequences.

## Why the throughput improved

The first teacher washed/reused one plate before beginning the next dish. The
v1 replay contains143 jointly idle ticks, including three40-step return waits,
while three other clean plates remain unused. Later teachers use all four clean
plates before washing and leave rice pots on their hobs. The short serial route
serves at83,175,274,378,497. A training-only queue compactor overlaps independent
real actions and serves at **78,158,250,346,458,570**.

The compactor uses a longer reference only to prepare task queues. The recorded
trajectory still begins with the standard empty kitchen and runs exactly600
steps under original rules/resources. Independent replay checked every state,
pre-action observation, food ledger and actual plate-to-dirty delivery. The
neural evaluator never imports or runs the teacher.

## Checks and unsuccessful trials

- CPU/GPU loss, every gradient, and one Adam update agree on a real512-step
  imitation sequence (maximum absolute difference2.79e-6).
- New positive-action Metal tests exposed lost boolean scatter updates: food
  changed physically while reward/delivery counters were wrong. Internal event
  and worked-item buffers now use integers, preserving boolean public info and
  the one-work-tick-per-item bound. Eight real-GPU action/observation gates pass.
- The naive CPU PPO warm start lost the working behavior:0 dishes on10 validation
  episodes after40 updates/20,480 environment steps. It is not a selected policy.
- A larger Metal PPO ten-update chunk exceeded the diagnostic wall-time bound
  without saving a checkpoint and was deliberately terminated. This is not a
  successful full-GPU PPO training result.
- A separate critic-only preparation experiment fits value heads while proving
  all actor/backbone parameters and checked action probabilities unchanged. Any
  subsequent PPO result must pass the same independent dish-count evaluation.

## Scope and artifacts

The selected v8 inference model is tracked under
`model/oc2_sushi_1_3/policy-spawn-v8/`. Original raw runs, training datasets,
optimizer checkpoints, v5/v7 references and generated replays remain local
ignored artifacts under `overcooked-v2/outputs/dish-learning/`. Preserve those
separately. Frozen policies contain weights, config, metadata, weight hashes
and source manifests; the tracked v8 folder is not an exact training-resume
checkpoint or a complete reproduction dataset.

The simulator retains provisional geometry, simplified scoring, no burning,
and one cucumber-rice-nori recipe. Full-map observations are an explicit first
competence choice. Partial-view cooperation, unseen layouts and real-game policy
transfer remain untested. These results are practical lower bounds on learned
throughput, not a proof of the maximum possible number of dishes.
