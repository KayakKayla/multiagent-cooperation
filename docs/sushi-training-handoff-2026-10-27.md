# Sushi training handoff — 2026-10-27

Historical stopping point. Work resumed on 2026-10-28; see the [ongoing journal](sushi-training-journal.md) for current findings and model selection.

Work is paused at the user's request. No training or evaluation job remains running.

## Best verified result

The frozen **policy-throughput-v4** neural policy served **six dishes in every one of 20 held-out, 600-step episodes**: 120 dishes total, mean raw team reward 120. All episodes washed and reused plates. The untrained model with the same architecture and evaluation seeds served zero dishes. One greedy rollout and ten sampled validation episodes also served six dishes each.

This is behavior cloning (learning from demonstrations), not a PPO-from-scratch success. Evaluation runs the neural policy closed-loop against the actual simulator, without teacher actions or task queues. The policy is a shared CNN + GRU with 190,976 parameters, width 32 and full 10×14×48 kitchen observations. Episode length stays 600 steps (150 simulated seconds); kitchen rules and resources were not relaxed.

Seeds 3000–3019 vary action sampling, not the fixed layout or starting positions. These results establish performance in this simulator configuration, not unseen-layout generalization or transfer to Steam Overcooked! 2. Geometry remains partly provisional, scoring simplified, and burning absent.

All paths below are relative to `overcooked-v2/` unless specified otherwise.

- Best model: `outputs/dish-learning/policy-throughput-v4/`.
- Weights SHA-256: `e8fa92a222494039e47a463e5ce147ecdb428bf06aa960a5e421a63126ce471c`.
- Final evaluation: `outputs/dish-learning/eval-v4-heldout20/summary.json`.
- Best-model replay: `outputs/dish-learning/eval-v4-heldout20/replay.html`.
- Paired untrained result: `outputs/dish-learning/eval-untrained-paired3000/summary.json`.
- Frozen directory includes weights, config, metadata, source snapshots and manifests. Preserve the whole directory.

The replay previously open in the browser, `eval-v1-heldout20/replay.html`, shows an older model. Use the v4 replay above for the latest selected result.

## What changed

- `sushi_imitation.py`: recurrent behavior-cloning learner with causal trajectories, loss masks, class weighting, atomic saves, full-state resume and source/data provenance checks.
- `scenarios/oc2_sushi_1_3/demonstrations.py`: action-derived demonstrations and optional legal movement-prefix augmentation.
- `throughput_expert.py` and `throughput_schedule.py` in that scenario: better demonstrations using all four plates, overlapping independent chef work and preparing the next rice early. These are training tools, not evaluation controllers.
- `evaluate_sushi_policy.py`: independent CPU closed-loop evaluation, physical delivery/resource checks, replay export and optional perturbations.
- `sushi_value_warmup.py`: critic-only fitting, checking that actor/backbone parameters and action probabilities remain unchanged.
- `train.py`: optional `--init-weights` with schema/shape/dtype/finite checks; fresh optimizer and state. It is distinct from `--resume`.
- `environment.py`: fixed Metal losing boolean scatter updates for delivery/work events. Internal buffers use integers; public info remains boolean and CPU semantics remain unchanged. This positive-delivery bug was not covered by the earlier tiny setup run.
- Three PPO fine-tuning configs and focused tests were added. Pinned JaxMARL and PPO update mathematics remain unchanged.

## Model progression and unsuccessful trials

| Candidate | Training | Result |
| --- | --- | --- |
| v1 | 1,000 imitation updates from four delayed two-dish demos | 3 greedy; 2.9 mean over 20 held-out episodes |
| v2 | 500 updates from v1 with position augmentation | Improved position-jitter performance, worse action-noise performance; not selected as best overall |
| v3 | 1,000 updates from v1 using a five-dish serial teacher | 5 greedy; 4.0 sampled validation mean |
| **v4** | **1,500 updates from v3 using six-dish concurrent teacher** | **6 greedy; 6.0 over 10 validation and 20 held-out episodes** |

v1 used Adam learning rate 0.001; later imitation stages used 0.0003 and gradient clipping at 1. Each warm start began a fresh optimizer. v4 used padded length 640 and one episode per update. Its dataset is `outputs/dish-learning/demonstrations-throughput/compacted-short.npz`, SHA-256 `a4d55a410b99723ae7da1e289dfb17374ada7069ab32e7989b76984496e67710`.

PPO trials did not beat the selected imitation model:

- Large Metal PPO: a ten-update chunk exceeded the diagnostic time bound without saving a checkpoint; it was terminated. This does not establish a deadlock, but larger PPO stability remains unverified. Diagnostic: `outputs/dish-learning/ppo-finetune-v1-diagnostic.json`.
- Naive CPU PPO from v1: 40 updates / 20,480 environment steps completed, but validation dropped to zero dishes. Run: `outputs/train/oc2_sushi_1_3-seed42-20261027T162448125350Z`; evaluation: `outputs/dish-learning/eval-ppo-cpu-validation/`.
- Critic preparation from v3 preserved actor outputs and lowered value MSE from 205.11 to 38.32. Subsequent conservative CPU PPO completed but averaged four dishes, no improvement over v3. Run: `outputs/train/oc2_sushi_1_3-seed42-20261027T163302583078Z`; evaluation: `outputs/dish-learning/eval-ppo-prepared-validation/`.

Do not select a policy using training loss alone. Preserve v4 until a new frozen model passes actual dish-count evaluation.

## Seven-dish target is ready — no neural v5 trained yet

A new independently audited teacher serves **seven dishes** at steps **78, 140, 211, 285, 375, 465, 555**. It overlaps work and buffers the next raw rice while the current sushi is completed. Rules, resources and the actual episode remain unchanged. A longer reference is used only to plan task queues.

- Dataset: `outputs/dish-learning/demonstrations-throughput/compacted-short-prefetch.npz`.
- SHA-256: `c43c80b0223257be89d7123782d58b7bdc0e0ec1cfc1c634f70717b6a0a0aa0d`.
- Audit: `outputs/dish-learning/compacted-short-prefetch-verification.json`.
- All 600 transitions, all 601 states and all pre-action observations matched independent CPU replay. There were no conflicting labels among 692 unique observations.

**This is a seven-dish teacher, not a seven-dish trained neural policy, and not an optimality proof.**

## Exact next step when resuming

Train a new candidate from frozen v4 on the single seven-dish dataset. Do not pass the whole demonstrations-throughput directory: it contains different strategies that can supply conflicting labels. The commands below were prepared for the next session and have not been run.

From the repository's `overcooked-v2/` directory, using the existing isolated Metal runtime:

```sh
PYTHONPATH='/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/.venv/lib/python3.12/site-packages:/Users/zuriz/.codex/worktrees/mac-training-support/multiagent-cooperation/overcooked-v2/JaxMARL' \
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MPLCONFIGDIR=outputs/dish-learning/matplotlib \
'/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/outputs/jax-metal-probe/venv/bin/python' \
sushi_imitation.py \
  --dataset outputs/dish-learning/demonstrations-throughput/compacted-short-prefetch.npz \
  --output outputs/dish-learning/bc-throughput-v5 \
  --init-weights outputs/dish-learning/policy-throughput-v4 \
  --updates 5000 --stop-after 1500 --save-every 500 \
  --padded-steps 640 --lr .0003 --max-seconds 900
```

Evaluate on CPU: first greedy and sampled validation seeds 1000–1009. Select and freeze weights before a final test on unused seeds 4000–4019. Example validation command:

```sh
JAX_PLATFORMS=cpu \
'/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/.venv/bin/python' \
evaluate_sushi_policy.py \
  --run outputs/dish-learning/bc-throughput-v5 \
  --output outputs/dish-learning/eval-v5-validation \
  --episodes 10 --eval-seed 1000
```

For greedy evaluation add `--action-selection greedy --episodes 1` and choose another output directory. For the final frozen test use `--episodes 20 --eval-seed 4000` with the frozen model directory, plus a paired untrained comparison. Keep the horizon at 600.

If resuming an interrupted learner, source/data/config hash guards must match. Do not bypass them. If source changes, create a fresh output run and warm start weights instead.

## Verification completed

- Full CPU suite: 88 tests discovered, 77 passed and 11 optional Metal checks skipped (`outputs/dish-learning/all-cpu-tests.log`). This run preceded the final optional prefetch teacher changes.
- After those teacher changes: eight focused throughput tests passed and the seven-dish dataset passed complete independent replay. The whole suite was not rerun afterward.
- Eight actual Metal action/observation tests passed, including positive deliveries, chopping/washing bounds and full CPU/GPU observation agreement (`metal-observations-fixed.log` under the same output root).
- Real 512-step imitation loss, gradients and one Adam update agreed between CPU and GPU, maximum absolute difference about 2.79e-6 (`imitation-numerics.json`).
- Every reported policy evaluation checked finite predictions, plate/pot conservation and physical delivery transitions against reward/counters.

## Git and preservation

Workspace: `/Users/zuriz/.codex/worktrees/mac-training-support/multiagent-cooperation`.

Current branch: **sushi-dish-training**. Learning changes and this handoff are local, uncommitted and unpushed. The original checkout was left untouched. Do not clean untracked files: they include the new learner, evaluator, teachers and tests. Ignored outputs contain the only local model/data/evidence artifacts and are not backed up by Git commits.

PR #43, **Include Mac training support and sushi PPO integration**, contains the earlier setup work: https://github.com/KayakKayla/multiagent-cooperation/pull/43 . It does not contain this subsequent dish-learning work or the newly found positive-delivery Metal fix. Obtain the user's confirmation before pushing a new branch/PR. No new commit or push was made during wrap-up.

More background: `docs/sushi-training.md` at the repository root. This handoff is the current stopping-point record.
