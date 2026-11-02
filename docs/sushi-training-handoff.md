# Sushi training handoff — 2026-10-28

Paused at the user's request. No training/evaluation/audit job remains running. The goal is not marked complete: seven is not a proven maximum, and some perturbed starts still fail.

The full chronological findings, training settings, data and reasoning are in [the training journal](sushi-training-journal.md). The older stopping point is archived in [the October27 handoff](sushi-training-handoff-2026-10-27.md).

## Current verified models

Workspace: `/Users/zuriz/.codex/worktrees/mac-training-support/multiagent-cooperation`.

All artifact paths below are relative to `overcooked-v2/outputs/dish-learning/`.

- **Latest broader-start model:** `policy-spawn-v8/`, weights SHA256 `8d086d8bd3f18e751e08974145dae825f0e18ed4b1edaed93515c7b4e474c46a`.
- Prior robustness reference: `policy-cooperation-v7/`, SHA256 `9705019dbedeaa9d10283e35b1718c5b622461fd4771c3ca7d1438d21ddd40b8`.
- Clean-start reference: `policy-throughput-v5/`, SHA256 `6f9045849d7194d69395fe1eaf8f8075e7623c92afa704d34f1bd982ac22e1d3`.

Each frozen folder contains weights/config/metadata/summary plus saved inference-source snapshots and manifests. Parent weights/manifests were rechecked at wrap-up.

Fresh paired final seeds6000–6019, each condition20episodes of600steps:

| Condition | V7 mean dishes | V8 mean dishes |
| --- | ---: | ---: |
| Standard start | 7.00 | **7.00** |
| Independent5% action delay | 5.75 | **6.40** |
| Radius1starting pose + random facing | 4.20 | **6.30** |

V8 served seven in every standard episode. Under pose changes it succeeded in18/20 versus v7's12/20. Two final joint poses matched training starts; on the18 unmatched starts, v8 averaged6.22 versus v7's3.89. This is fixed-layout start/timing robustness, not unseen-layout/partner performance or Steam-game transfer. One delayed episode and two pose episodes still served zero.

Evidence: `v8-final-comparison.json`, `eval-v8-final-{standard,dropout05,jitter1}/`, `eval-v7-paired6000-{standard,dropout05,jitter1}/`, log `v8-final-paired6000.log`. Actual neural replays: `eval-v8-final-standard/replay.html` and `eval-v8-final-jitter1/replay.html`.

## What was trained

Method: behavior cloning, not PPO. Shared CNN+GRU,190,976parameters, GRU/head width32, full10×14×48 observations, seven actions. No teacher/task queues at evaluation.

V8 starts from frozen v7 with fresh Adam: LR0.0001, beta defaults0.9/0.999, epsilon1e-8, global clip1, seed42, one complete two-chef trajectory/update, padded length640,1,500updates (declared maximum5,000), save every500. Completed in279.1seconds on the Apple GPU. Actual success was measured separately on CPU.

Training run/resume state: `bc-spawn-v8/`, `checkpoints/update-001500`, log `bc-spawn-v8.log`.

Data: `demonstrations-v8/`,57trajectories: previous17 anchors copied byte-identically plus40 explicitly declared spawn variations. The new examples cover all20 individual nearby position/facing choices per chef twice,40/400joint tuples. Food/resources/timers remain pristine; only initial poses change, with600actual transitions. Each has independent5% chef delays, with own and unobservable partner-dependent labels masked. All40teachers served seven. All transitions/observations independently replayed;64,973supervised decisions,64,761distinct input histories,zero conflicting histories. Audit/source snapshots: `demonstrations-v8-{audit,label-audit}.json`, `demonstrations-v8/sources/`; reproduction script/log `generate_sushi_v8.py`, `spawn-demonstrations-v8.log`.

The changed mixture, extra training and warm start are confounds; do not claim a pure isolated pose-coverage effect. V8 validation:7.0normal,6.5delayed,6.3pose on seeds1000–1009, before final selection.

## Cooperation findings

- Actual neural role division/handoffs are recorded: one chef chops/serves; the other assembles/washes, exchanging food/plates via counters.
- Using available plates, overlapping tasks and preparing next rice during assembly improved demonstrated throughput to seven.
- Varied timing and starts improved robustness; latest broader examples improved changed-start completion while preserving standard output on the fresh paired test.
- Three bounded teacher variants aimed at eight still served seven. One increased simultaneous activity294→310without changing delivery times. More concurrency alone did not improve output. Source/report/log: `eight-teacher-search/`. No optimum claim.

## Code and checks

This continuation only extends the local training-data helper `scenarios/oc2_sushi_1_3/robust_demonstrations.py` with explicitly labeled spawn variation and adds two focused tests in `tests/test_sushi_robust_demonstrations.py`. Current six focused tests pass, including invalid input/reset preservation and old default equivalence. Earlier whole-suite checkpoint:82CPU passes,11optional Metal skips. Model/learner/evaluator/physics and pinned JaxMARL were unchanged in this continuation.

The generic replay-analysis skill's extractor does not support this sushi replay schema. It rejected a failure replay at `eval-v7-jitter-failure1001/`; no exact-step diagnosis was claimed through that helper. Use a sushi-compatible validated state audit before interpreting residual failures.

## Resume later

Do not launch anything until the user resumes. Select the next intervention first; protect the frozen models. Likely next: inspect remaining starting-pose/recovery failures or a more informed critical-path search for eight dishes. Broader joint coverage is a data/training direction, not an improvement already demonstrated.

Final ranges4000–4019,5000–5019,6000–6019 have now been used. Choose a new predeclared final range, such as7000–7019, after future validation/selection. Fresh seeds alone do not guarantee novel joint poses; retain overlap reporting.

Existing isolated runtimes:

- CPU: `/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/.venv/bin/python`.
- Metal: `/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/outputs/jax-metal-probe/venv/bin/python`, with `ENABLE_PJRT_COMPATIBILITY=1`, `JAX_PLATFORMS=METAL,cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and the base venv site-packages plus this worktree's JaxMARL on PYTHONPATH (exact older launch example in the archived handoff).

For an exact resume of v8, keep source/data/config unchanged, use the same arguments below, omit `--init-weights`, add `--resume` and choose a larger `--stop-after`. Do not bypass hash guards. For new data/settings, use a new run directory and warm-start frozen weights.

```sh
sushi_imitation.py --dataset outputs/dish-learning/demonstrations-v8 \
  --output outputs/dish-learning/bc-spawn-v8 \
  --updates 5000 --stop-after 3000 --save-every 500 \
  --padded-steps 640 --lr .0001 --max-seconds 900 --resume
```

This is a future argument record, not a command already run. No extra training was started during wrap-up.

## Git and preservation

Local packaging on November 1, 2026 commits the learning sources, tests,
configs and documentation on `sushi-dish-training`. The final CPU check passed
84 tests with 11 optional Metal skips. No additional research training ran.
The user authorized publication: branch `sushi-dish-training` is pushed and
[PR #44](https://github.com/KayakKayla/multiagent-cooperation/pull/44) is open
against `main`, including the frozen v8 model. PR #43's earlier Mac/PPO
integration is merged. The historical journal and October 27 handoff retain their
original stopping-point descriptions.

The complete frozen v8 inference model is now tracked under
`model/oc2_sushi_1_3/policy-spawn-v8/`, with original provenance, source snapshots
and integrity hashes. Teammates can use the model and learning guides to run it
without a separate download. Ignored outputs still contain the training data,
optimizer checkpoints, v5/v7 references, raw runs and replay evidence; preserve
those separately. The original checkout remains untouched.
