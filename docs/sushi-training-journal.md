# Sushi training journal

Updated: 2026-10-28 (America/Toronto). **Latest: frozen v5 serves seven dishes in all 20 final episodes; the first robustness pilot was stopped to fix a supervision inconsistency; PAUSED at user request. V8 final results:7.00 clean /6.40 delayed /6.30 changed-start mean; all files/models saved.** This is the ongoing record; [the previous handoff](sushi-training-handoff.md) preserves the earlier stopping point. Run outputs are local ignored artifacts, not stored in Git. No new work has been pushed.

Workspace: `/Users/zuriz/.codex/worktrees/mac-training-support/multiagent-cooperation`. Paths below are repository-relative unless stated otherwise.

## Latest result at a glance

- **Latest broader-start model:** frozen `policy-spawn-v8`, seven dishes in all20 fresh standard episodes; paired means7.00 clean /6.40 delayed /6.30 changed-start. Frozen v5/v7 references remain preserved.
- **Matched latest comparison:** on fresh seeds6000–6019, v7 means7.00 /5.75 /4.20; v8 means7.00 /6.40 /6.30. Prior comparisons below use their own stated seed ranges.
- **Training:** recurrent behavior cloning. V5: one canonical concurrent trace; v7:17 audited timing/prefix traces; v8:57traces (17anchors+40explicit spawn variants),1500updates fromv7 atLR0.0001. No PPO improvement claimed.
- **Verification:** final physical/resource audits passed. Current six focused spawn tests passed; previous full-suite checkpoint:82CPU passes/11optional Metal skips.
- **Status:** paused at user request; no jobs running; local uncommitted/unpushed branch `sushi-dish-training`. Both selected models, data, logs, checkpoints and source snapshots are saved in `overcooked-v2/outputs/dish-learning/` (ignored by Git).

## Goal and measurement

Train two chefs to finish as many cucumber–rice–nori sushi dishes as possible in the local Story 1-3 simulator, ideally by coordinating their work. Keep standard resources, fixed layout/starts, and a 600-step episode (150 simulated seconds). A physical delivery earns 20 raw shared reward. This is not the commercial game's score or a policy deployed into Steam.

We select models by separate CPU closed-loop dish-count evaluations, not training loss. The evaluator runs neural actions against actual subsequent observations and checks physical delivery transitions, rewards, finite predictions and plate/pot conservation. Seeds currently vary policy sampling, not unseen kitchens. No optimality or broad generalization claim is established.

## Current best at the start of this session

**Frozen v4: six dishes in every one of 20 held-out sampled episodes (seeds 3000–3019), 120 dishes total.** A greedy episode and ten validation episodes also served six. All 20 held-out episodes washed and reused plates. The paired untrained model served zero.

- Model: `overcooked-v2/outputs/dish-learning/policy-throughput-v4/`.
- Replay: `overcooked-v2/outputs/dish-learning/eval-v4-heldout20/replay.html`.
- Results and evaluation protocol: `summary.json`, `episodes.json`, `protocol.json` in that evaluation directory.
- Model weights SHA-256: `e8fa92a222494039e47a463e5ce147ecdb428bf06aa960a5e421a63126ce471c`.

## What type of training is being used?

The successful models use **behavior cloning**: learn the two chefs' recorded actions from audited simulator demonstrations. Each chef shares the same CNN + GRU policy weights and receives a full-map observation from its own perspective. The model has 190,976 scalar parameters, a 32-wide GRU, 32-wide actor/value heads, seven discrete actions and 10×14×48 observations. Recurrent histories are kept in their original sequence; there is no teacher/task queue at evaluation.

This differs from **PPO reinforcement learning**, which updates from rewards collected by the policy. Earlier CPU PPO fine-tuning attempts did not improve the selected policy; naive fine-tuning lost the working behavior. A larger Metal PPO chunk exceeded its diagnostic time bound without saving weights. See the previous handoff for run locations. We are using imitation now because it already learns real deliveries and lets us test better coordinated examples with a clear target.

## Previous training and results

| Model | Training data and updates | Result |
| --- | --- | --- |
| v1 | Four delayed two-dish demonstrations; 1,000 imitation updates, Adam LR 0.001 | 3 greedy dishes; 2.9 mean over 20 held-out episodes |
| v2 | Legal movement-prefix pose augmentation; 500 updates from v1, LR 0.0003 | Better initial-pose robustness; worse action-noise results; not best overall |
| v3 | Five-dish serial workflow using available plates; 1,000 updates from v1, LR 0.0003 | 5 greedy; 4.0 mean on ten validation seeds |
| v4 | Six-dish concurrent workflow; 1,500 updates from v3, LR 0.0003 | 6 greedy and 6.0 mean on ten validation and twenty held-out seeds |

Each warm start creates a fresh optimizer. Declared BC `--updates` is a maximum budget; `--stop-after` sets the actual staged stopping update. PPO fields retained in the shared config (such as NUM_ENVS and PPO epochs) do **not** describe these imitation runs. The active imitation parameters are under `IMITATION` and the actual update count is in `summary.json`.

## Cooperation: evidence so far and limits

- **Complementary work and handoffs:** demonstrations assign ingredient preparation and final assembly/serving to different chefs, with food crossing a shared counter. The neural rollout must reproduce actual interactions to serve dishes.
- **Overlapping work:** the six-dish teacher overlaps independent real actions; v4 then learned six-dish performance. This supports learning a coordinated workflow, but several teacher changes were made together, so their individual causal effects are not yet isolated.
- **Resource planning:** later teachers use all four clean plates before washing and reuse returned plates. Every v4 held-out episode demonstrably washed and reused plates.
- **Preparing ahead:** the next teacher buffers raw rice for the next dish while the partner completes the current one. It serves seven dishes with the same rules and resources. Whether the neural policy learns it remains to be measured.
- **Not yet established:** flexible role switching, recovery from teammate mistakes, unseen teammates/layouts, explicit communication, or superiority to an optimized one-chef baseline. The current fixed-start shared policy may learn stable roles from position/history.

A separate read-only audit is examining exact handoff and simultaneous-action evidence. Counts of non-STAY actions alone are activity measures, not proof of productive cooperation.

## Experiment 2026-10-28: v5, learn the seven-dish workflow

**Status: frozen v5 passed the fresh final test; robustness data extension is next.** Both parent weights and dataset hashes were rechecked before launch. No previous training process was running.

### Why this experiment

Keep the architecture, horizon and environment fixed and change the demonstrations to a better overlapping workflow. This tests whether learned throughput can rise from six to seven without relaxing the task. Warm-start v4 to retain the already working kitchen interactions.

### Training data

One complete audited 600-transition episode; both chefs supply labels (1,200 decisions). File: `overcooked-v2/outputs/dish-learning/demonstrations-throughput/compacted-short-prefetch.npz`.

SHA-256: `c43c80b0223257be89d7123782d58b7bdc0e0ec1cfc1c634f70717b6a0a0aa0d`.

Teacher deliveries: **78, 140, 211, 285, 375, 465, 555**. All 600 transitions, 601 states and pre-action observations matched independent CPU replay. No conflicting labels among 692 unique full-map observations. Audit: `overcooked-v2/outputs/dish-learning/compacted-short-prefetch-verification.json`. A longer reference prepares task queues only; the recorded episode itself is the unchanged 600-step task.

Use this exact file, not the containing directory, which includes other strategies. Padding receives zero loss. Full-map observations encode the current kitchen; future states, teacher phases and task queues are not inputs.

### Parameters and locations

- Method: recurrent behavior cloning, cross entropy with inverse-square-root action-frequency weighting.
- Parent: frozen `policy-throughput-v4`; fresh Adam optimizer.
- Seed: 42; one trajectory per update, both chefs; padded sequence length 640.
- LR: 0.0003; global gradient norm clip: 1.0.
- Declared maximum: 5,000 updates; first stage stops at 1,500, saves every 500; wall-time bound 900 seconds.
- Training device: Apple GPU using the isolated JAX Metal runtime; CPU for evaluation.
- Run: `overcooked-v2/outputs/dish-learning/bc-throughput-v5/`.
- Log: `overcooked-v2/outputs/dish-learning/bc-throughput-v5.log`.
- Inspect `config.json`, `metadata.json`, `summary.json` and checkpoint subdirectories for exact provenance, device, update count, loss and weights.
- Training source: `overcooked-v2/sushi_imitation.py`; demonstrations/planner: `overcooked-v2/scenarios/oc2_sushi_1_3/{throughput_expert,throughput_schedule}.py`; evaluator: `overcooked-v2/evaluate_sushi_policy.py`.

### Evaluation plan, set before seeing results

1. CPU greedy episode and ten sampled validation episodes, seeds 1000–1009; actual dishes and delivery times.
2. If the candidate improves, freeze weights/config/source manifest before a final sampled test on seeds 4000–4019, with paired untrained reference. These are sampling seeds, not novel layouts.
3. Inspect cooperative activity/handoffs and clearly report what is observed versus not causally proven. Keep v4 if the candidate does not improve actual deliveries.
4. Record findings and any changed training decision below as they occur.

## Session results and decisions

### Verified cooperation evidence from the existing v4 replay

Independent state/action audit: `overcooked-v2/outputs/dish-learning/sushi-cooperation-audit.json`; reproduction script: `sushi_cooperation_audit.py` in the same directory. The report hashes its input artifacts and defines its metrics.

In the first v4 held-out episode (seed 3000), **29 item transfers crossed between chefs via counters**. Chef 0 finished all six chops and served all six dishes; chef 1 completed all six full-plate assemblies and both washes. The four initial plates were used, then two were washed and reused (delivery slots 3, 4, 5, 6, 3, 4). This is concrete complementary-role behavior, not just two agents moving.

First dish timeline (simulator step):

| Step | Observed contribution |
| ---: | --- |
| 9 | Rice passed from chef 0 to chef 1 |
| 28 | Cucumber passed from chef 1 to chef 0 |
| 38 | Chef 0 finished chopping |
| 57 | Partial plate passed from chef 0 to chef 1 |
| 63 | Chef 1 completed the plate |
| 70 | Completed plate passed back to chef 0 |
| 78 | Chef 0 served the dish |

On the same validation seeds, v3 averaged 4 dishes, 0.8 simultaneous non-STAY ticks and 66.7 invalid interactions; v4 averaged 6 dishes, 46 simultaneous non-STAY ticks and zero invalid interactions. This supports better execution of an overlapping workflow, but simultaneous actions can include turns/blocked actions. It does not isolate the effect of each training change.

The older v1 example spent 143 ticks with both chefs choosing STAY; 117 of those had a plate queued for return while three other clean plates remained unused. Using available plates and preparing while timers advance addressed visible wasted capacity. These are observations from the recorded examples, not a randomized ablation.

### New finding: v4 is sensitive to delays

A separate ten-episode CPU check independently replaced each chef action with STAY with probability 0.05 (same validation seeds 1000–1009). v4 fell from 6.0 to **2.4 mean dishes**; only 6/10 episodes served any dish. There were 627 selected action overrides, 324 of which actually changed a proposed action (some proposals already were STAY).

Files: `overcooked-v2/outputs/dish-learning/eval-v4-dropout05/{summary,episodes,protocol}.json`; log: `eval-v4-dropout05.log`. This is an externally imposed delay test, not a new trained teammate. It shows that the learned workflow is effective under normal execution but recovery from disruption remains weak. Evaluate v5 under the same intervention before claiming more reliable cooperation.

### v5 training in progress

The parent and seven-dish dataset hashes matched the handoff. The training process reports Metal backend and 190,976 parameters. It reached 100% teacher-action accuracy early, but this alone is not learning success: free-running evaluation is still required. No seven-dish neural result is claimed yet.

### Controlled teacher comparison: rice prefetch enables the seventh dish

Regenerated the no-prefetch teacher with the current source. Its observations, actions and every state array exactly match the previous six-dish control. Both current-source variants have identical rules, initial state, short routing, nori-first ordering and 1,200-step planning reference; only `prefetch_rice` differs. Actual recorded episodes remain 600 steps.

| Scripted teacher | Dishes | Simultaneous non-STAY ticks | Joint-STAY ticks |
| --- | ---: | ---: | ---: |
| Prefetch disabled | 6 | 89 | 18 |
| Prefetch enabled | 7 | 294 | 3 |

This isolates a useful cooperation change **in the teacher**: one chef supplies the next rice through a second counter while the partner finishes the current plate. It does not yet prove the neural learner's gain or adaptive teammate modeling. Exact hashes/settings/state comparisons: `overcooked-v2/outputs/dish-learning/prefetch-ablation/comparison.json`; regenerated control: `prefetch-ablation/no-prefetch.npz`.

### v5 stage completed

The first stage completed **1,500 updates in 269.1 seconds** on Metal. Final weighted cross entropy was **0.00007182**, unweighted NLL **0.00007251**, and teacher-action accuracy **100%**. The saved weights and full optimizer/RNG checkpoint are in `bc-throughput-v5/`; checkpoint `checkpoints/update-001500` supports exact resume under unchanged source/data/config. These are training metrics, not a dish-count result. CPU greedy, sampled validation and 5% action-delay evaluation are now running; log `overcooked-v2/outputs/dish-learning/v5-validation.log`.

### v5 at 1,500 updates: seven learned, sampling fragility remains

CPU greedy evaluation served **7 dishes**. Sampled validation served seven on 9/10 seeds and three on seed 1003: **6.6 mean**, 66 total. That failing episode delayed its third delivery to step 226, then accumulated invalid interactions. Training accuracy therefore did not guarantee every closed-loop sampled trajectory succeeds. Files: `eval-v5-greedy/`, `eval-v5-validation/`; log `v5-validation.log`, all under `overcooked-v2/outputs/dish-learning/`.

At 5% action dropout, the same candidate averaged **2.1 dishes** (8/10 served any), versus v4's 2.4 (6/10 served any). These small, mixed results do not establish improved robustness. Delay selections were identical (627), but actual changed actions differed because the policies choose different actions.

Preserved the 1,500-update weights/config/source snapshots separately in `policy-throughput-v5-1500/` before resuming the training run. Next stage: **exact resume from update 1,500 to 3,000**, unchanged dataset, optimizer, seed/config and source; same LR 0.0003 and padded length 640, max 900 seconds for the new stage. Reason: normal-execution validation showed a rare sampled departure from an otherwise successful seven-dish sequence. More fitting may reduce that sampling error; it is not expected to solve teammate-delay recovery by itself. No final held-out test has been used to make this decision.

### Reading the training data and metrics

The v5 NPZ contains `observations[600,2,10,14,48]`, `actions[600,2]`, `supervision_mask[600,2]`, `metadata_json` and `state_*` arrays for 601 states. NumPy loads it with `allow_pickle=False`. Model inputs use the observations and GRU history; `state_*` arrays support auditing and are not extra policy inputs. Action counts in this trajectory are right 214, down 108, left 205, up 99, STAY 309, pickup/place 175 and work 90.

The learner pads to 640 time steps with zero supervision weight on padding. Optimizer: Optax Adam defaults (beta1 0.9, beta2 0.999, epsilon 1e-8), learning rate 0.0003, preceded by global gradient clipping at 1. One update uses the full two-chef trajectory; repeated updates do not create new independent demonstrations. No reward objective, entropy bonus, value loss or PPO minibatches are used in behavior cloning. The successful model's value head has not been trained as a reliable critic by this objective.

Why imitation first: the demonstration supplies feedback at every supervised action, whereas reward is only earned when a complete dish is delivered. This is a practical rationale supported by the working imitation results and failed small PPO trials; it is not a controlled claim that PPO could never work.

### Additional fixed-layout robustness check, declared before results

Compare v4 and the resumed v5 on ten validation seeds (1000–1009), with each initial chef position jittered within one Manhattan tile to a valid distinct floor pair and facing randomized. Keep all resources empty/standard. This tests sensitivity to starting poses and fixed role routines; it is not an unseen-layout or unseen-partner test. Existing evaluator options: `--spawn-jitter 1 --random-facing --episodes 10 --eval-seed 1000 --no-replay`. v4 output: `eval-v4-jitter1/`; v5 output will be named with its actual update count. No policy updates occur during this check.

### v5 at 3,000 updates: validation improved; model frozen

Exact resume completed another 1,500 updates in **264.9 seconds**, for **3,000 total** and about **534 seconds of training across both stages**. The resume stage retained optimizer and RNG state. Final weighted CE: **0.00001812**; training accuracy: **100%**.

| CPU validation condition | v4 mean dishes | v5 at 1,500 | v5 at 3,000 |
| --- | ---: | ---: | ---: |
| Standard starts, sampled seeds 1000–1009 | 6.0 | 6.6 | **7.0** |
| Independent 5% per-chef action delay | 2.4 | 2.1 | 2.1 |
| Start jitter radius 1 plus random facing | 0.0 | not measured | 0.7 |

The resumed v5 served seven in **all ten normal validation episodes** and in its greedy episode. It washed and reused plates in every normal episode. This supports the continuation decision for normal throughput. **More fitting did not solve robustness**: only one of ten jittered episodes served anything, and delay performance remained low. Pose changes and action delays are separate interventions; we did not isolate position from facing effects.

Selected and froze this candidate **before** final seeds 4000–4019. Model/config/source snapshots: `overcooked-v2/outputs/dish-learning/policy-throughput-v5/`; weights SHA-256: `6f9045849d7194d69395fe1eaf8f8075e7623c92afa704d34f1bd982ac22e1d3`. Full training resume state remains in `bc-throughput-v5/checkpoints/update-003000`. Validation directories are `eval-v5-3000-{greedy,validation,dropout05,jitter1}/`; log `v5-3000-validation.log`.

Next predeclared final comparison: frozen v5, frozen v4 and untrained full-map reference on the same unused action-sampling seeds 4000–4019. Preserve the selected weights regardless of final results; do not tune against these seeds.

### Final frozen v5 comparison: seven dishes in every episode

The predeclared final test completed on action-sampling seeds **4000–4019**:

| Frozen policy | Dishes per episode | Total dishes | Washed-plate reuse |
| --- | ---: | ---: | ---: |
| Untrained same-architecture reference | 0 in all 20 | 0 | 0/20 |
| v4 | 6 in all 20 | 120 | 20/20 |
| **v5, 3,000 updates** | **7 in all 20** | **140** | **20/20** |

This is **one additional dish per episode, about 16.7% more than v4**, under unchanged rules/resources/horizon. All prediction/resource/physical-delivery audits passed. The first two delivery times are 78 and 140 steps (v4: 78 and 158). Final results: `eval-v5-heldout20/`, `eval-v4-paired4000/`, `eval-untrained-paired4000/`; combined log: `v5-heldout-and-controls4000.log`, all below `overcooked-v2/outputs/dish-learning/`.

Latest actual neural replay: **`overcooked-v2/outputs/dish-learning/eval-v5-heldout20/replay.html`**, also `rollout.gif`. The frozen v5 remains the best verified normal-start model. Seven is a demonstrated achievable count, not a maximum proof. Twenty sampling seeds share the same kitchen and starting poses.

### Next experiment: varied timing and action-derived starts

Reason: the seven-dish model has strong standard execution but weak delay/pose recovery. Extend only the training demonstration planner to generate legal short movement prefixes and independent per-chef action delays. Keep physics, network, learner and evaluation unchanged. Prefixes must be real movements with food/resources untouched; begin training at the reached pose with zero recurrent carry, preserving its actual elapsed time. Mask artificially delayed commands out of supervised loss, but retain the resulting observations and the partner's legitimate waits/actions. Keep all generated trajectories within the original 600-step horizon.

Before training, independently replay every generated transition/observation and inspect food/plate/pot conservation. Add focused checks for zero-delay preservation, delayed-command masks and action-derived initial states. Preserve frozen v5 as the normal-throughput baseline. Subsequent validation uses the same declared standard/dropout/jitter conditions; any newly selected model needs unused final seeds 5000–5019 rather than reusing v5's final test for selection.

### Cooperation in the actual seven-dish neural rollout

The saved first final-test rollout (seed 4000) was audited from exact states/actions: **35 cross-chef item transfers**, **294 simultaneous non-STAY ticks**, and **3 joint-STAY ticks**. These match the intended overlapping workflow rather than the old mostly serial pattern. Chef 0 completed seven chops and seven serves; chef 1 completed seven plate assemblies and four washes. Three washed plates were served again; the fourth wash at step 575 prepares the unfinished next cycle. This is one recorded neural episode, not a claim that all 20 had identical action traces. More transfers are not automatically better; the independently verified seventh dish is the performance outcome. Exact event steps/ownership: `overcooked-v2/outputs/dish-learning/v5-cooperation-audit.json`.

The new augmentation passed three focused CPU tests: unchanged no-augmentation seven-dish behavior (all state/action/observation arrays), legal-prefix state reconstruction with resource preservation, independent replay of every delayed transition/observation, and argument bounds. Log: `overcooked-v2/outputs/dish-learning/robust-demo-tests.log` (3 tests, 7.217 seconds). Physics/network/learner/evaluator sources remain unchanged; only the training planner gained optional initial-state and per-chef-delay controls, plus `robust_demonstrations.py` and focused tests.

Augmentation generation now requests all seeds 200–215 with no retries/performance filtering: eight standard-start delayed trajectories and eight trajectories after eight actual legal movement-prefix steps, all with independent 5% per-chef delays. The original canonical seven-dish episode is included as an anchor. Directory: `overcooked-v2/outputs/dish-learning/demonstrations-v6/`; log: `robust-demonstrations-v6.log`; successful final audit will be `demonstrations-v6-audit.json`. No v6 training will start unless generation and independent replay finish successfully.

### v6 training protocol and data audit

All 16 requested augmented demonstrations generated successfully with **seven dishes each**, without retries or seed filtering. Every transition and pre-action observation independently replayed exactly; food/resource checks passed. Together with the canonical anchor there are **17 trajectories** (eight delayed standard starts, eight action-derived alternative starts, one original standard trace). Prefix episodes contain 592 supervised-history steps and preserve actual times 8→600; others contain 600. Each is padded to 640 for training. Hashes and checks: `demonstrations-v6-audit.json` and `demonstrations-v6-label-audit.json` under the output root.

Observation-only label audit: **19329 supervised agent decisions**, **4066 distinct observation tensors**, **22 tensors with multiple action labels across trajectories. These conflicts can be valid alternative or history-dependent scheduling choices; the recurrent learner receives the actual histories. They are recorded as a possible source of imitation difficulty, not silently removed.

**Next training stage, declared before launch:** recurrent behavior cloning from frozen v5, fresh Adam at **LR 0.0001**, gradient clip **1.0**, seed **42**, one full trajectory per update sampled from the 17-trajectory dataset, padded length **640**, **1,500 updates** initially (declared maximum 5,000), save every 500, wall-time bound 900 seconds. Lower LR is chosen to adapt the working policy while reducing abrupt loss of the canonical behavior; whether it helps is unproven. Run: `overcooked-v2/outputs/dish-learning/bc-cooperation-v6/`; log `bc-cooperation-v6.log`. Same 190,976-parameter network, physics, scoring, episode horizon and CPU evaluator.

Validation will compare v5 and v6 on standard starts, the already declared 5% delay test and the same radius-1/facing perturbation. Select separately for normal throughput and robustness; do not silently replace the seven-dish baseline with a weaker standard-start model. Any new final test must use fresh seeds 5000–5019.

### Independent review caught an unobservable partner-delay label; v6 stopped

The transition/resource audit passed, but a more stringent **entire observation-history** check found one conflicting recurrent-history group among 19,116 distinct supervised histories. At time index 8, canonical and seed207 have array-identical chef-1 input histories. Canonical teaches pickup; seed207 masks chef 0's injected delay but incorrectly supervises chef 1 to wait because the planner knows the current hidden partner-delay coin flip. No GRU can infer that independent coin flip. Therefore not all observation-label ambiguity is history-resolvable.

Stopped only the v6 training process with SIGINT; confirmed exit code **130**. Saved checkpoint: **500 updates**; last printed update **700**. The run is not selected or reported as a successful policy. Original data, log, and original planner sources are preserved in `demonstrations-v6/`, `bc-cooperation-v6/`, `bc-cooperation-v6.log`, and `v6-original-sources/`. The explicit stopped-run record is `bc-cooperation-v6/summary.json`. Frozen v5 is intact.

Correction: compare the teacher's actions on the same state with and without that instant's imposed delays. Additionally mask available-chef labels that changed because of the hidden current partner delay. Genuine waiting caused by the already observed kitchen state stays supervised. This changes supervision only, not executed actions, physics or resources. A regression test uses the exact conflicting full-history case. Corrected data will be a **new directory/run**, never an in-place replacement of the interrupted run's inputs.

### Corrected supervision checks passed; v7 data verification running

The correction now compares restricted and unrestricted teacher choices on the **same pre-transition state and queue heads**. No simulator state changes during this comparison. Own imposed delays are always masked; partner labels changed by that instant's delay are additionally masked. Counts distinguish those two causes. A first correction attempt failed assertions (pre/post-state comparison and overly broad STAY assumption); it was corrected before generating or training on replacement data.

The corrected suite passes **4 CPU tests in 10.277 seconds**, including the exact canonical-vs-seed207 identical-history regression, complete transition replay, prefix resource preservation and zero-augmentation preservation. Log: `robust-demo-maskfix-tests-corrected.log`. Independent review found no further implementation blocker.

Corrected data is being saved separately to `overcooked-v2/outputs/dish-learning/demonstrations-v7/`. The reproduction script `outputs/dish-learning/generate_sushi_v7.py` independently replays every transition, requires observations/actions/all state arrays to equal v6, verifies masks only remove supervision, and rejects conflicting **entire recurrent histories**. Final records will be `demonstrations-v7-audit.json`, `demonstrations-v7-label-audit.json`, and `robust-demonstrations-v7.log`. No replacement-data policy has been trained yet.

### v7 data passed: zero conflicting supervised recurrent histories

All 17 corrected files independently replayed. Every observation/action/state array exactly matches the corresponding original v6 file; only supervision masks and metadata differ. The correction removed **nine additional labels** overall: 19,329→**19,320** supervised decisions. There are **19,108 distinct supervised recurrent histories and zero conflicting history groups** in this finite corpus. Twenty-one current-observation-only conflicts remain, which can be consistent when histories differ. These checks establish corpus consistency, not that the GRU will generalize.

Final records: `demonstrations-v7-audit.json`, `demonstrations-v7-label-audit.json`, `robust-demonstrations-v7-final-audit.log`; exact generator sources copied to `demonstrations-v7/sources/`. The first audit script was interrupted only to fix repeated NPZ decompression; all generated files were preserved and the efficient final audit completed from them. No duplicate generation or in-place data replacement occurred.

**v7 training, declared before launch:** fresh run from frozen v5, not the interrupted v6 model; seed42, recurrent behavior cloning on the same 17 physical trajectories with corrected masks, Adam LR **0.0001**, clip **1.0**, one trajectory/update, padded length **640**, first **1,500 updates** of declared 5,000, save every500, bound900seconds. Run: `overcooked-v2/outputs/dish-learning/bc-cooperation-v7/`; log `bc-cooperation-v7.log`. Standard/dropout/jitter validation is unchanged. Frozen v5 remains the selected seven-dish baseline until actual validation justifies a different choice.

### Regression verification after the augmentation changes

The full current CPU suite completed: **93 tests total, 82 passed, 11 optional Metal tests skipped**, in **42.174 seconds**. It covers the simulator, replay, checkpoint/resume, policy warm starts, imitation, evaluation and new augmentation tests. Log: `overcooked-v2/outputs/dish-learning/all-cpu-tests-20261028.log`. The pinned JaxMARL submodule has no local changes. Actual v5/v7 training uses Metal; no new simulator/GPU numerical implementation was introduced by the CPU-only demonstration augmentation.

The running v7 experiment remains a **candidate**, not a result. Its training accuracy is measured on the trajectory sampled at each update; it is not an aggregate free-running success rate. Keep the final frozen v5 as the verified normal-start reference until the declared CPU evaluation completes.

### v7 validation: more reliable execution of the cooperative workflow

The corrected-data run completed **1,500 updates in 267.3 seconds**. Final sampled-trajectory teacher accuracy was **99.56%**, weighted CE **0.007764**. Lower training accuracy than v5 did **not** imply worse free-running behavior: the data is more varied and recovery is evaluated directly.

| CPU validation, seeds 1000–1009 | Frozen v5 | v7 |
| --- | ---: | ---: |
| Standard starts, sampled | 7.0 mean, all ten seven | **7.0, all ten seven** |
| Independent 5% per-chef action delays | 2.1 mean, 7/10 served any | **6.5 mean, 10/10 served any** |
| Radius-1 starting-pose jitter plus random facing | 0.7 mean, 1/10 served any | **2.8 mean, 4/10 served any** |

V7 also served seven greedily. Nine of ten delayed episodes served seven; one served two. This supports a useful data intervention while retaining standard throughput, but the small validation set and combined timing/pose augmentation do not isolate which part caused each gain. Initial-pose coverage remains incomplete. Files: `eval-v7-{greedy,validation,dropout05,jitter1}/`; log `v7-validation.log`.

Selected and froze v7 **before** fresh final seeds 5000–5019, on all three declared conditions with frozen v5 paired on the same seeds. Frozen directory: `overcooked-v2/outputs/dish-learning/policy-cooperation-v7/`; weights SHA-256 **9705019dbedeaa9d10283e35b1718c5b622461fd4771c3ca7d1438d21ddd40b8**. Weights/config/metadata/summary/source manifest are preserved; full resume checkpoint remains `bc-cooperation-v7/checkpoints/update-001500`. A fresh final result is pending; no further fitting will use those final seeds.

### Final paired result: robustness improved, clean-start tradeoff retained

The frozen-model comparison finished on **20 fresh paired seeds 5000–5019 for each condition**. Both models receive identical initial poses and imposed-delay masks. Models were frozen before these seeds were used. Independent physical-delivery/resource/finite-prediction checks passed; frozen weights and saved source hashes were rechecked.

| Final condition | V5 mean / total dishes | V7 mean / total dishes | V5 / V7 episodes serving any dish |
| --- | ---: | ---: | ---: |
| Standard start | **7.00 / 140** | 6.85 / 137 | 20/20 / 20/20 |
| Independent 5% per-chef action delays | 2.30 / 46 | **6.30 / 126** | 17/20 / 20/20 |
| Radius-1 starting poses plus random facing | 0.35 / 7 | **4.55 / 91** | 1/20 / 13/20 |

V7's normal-start loss came from one four-dish episode; the other 19 served seven. Under delays, 16 episodes served seven; the remaining four counts were 2, 4, 6 and 2 (see exact per-seed rows; do not infer perfect recovery). Under pose jitter, 13 served seven and seven served none. The limited pose coverage still needs improvement.

Paired per-seed wins/ties/losses for v7 versus v5: **0/19/1** normal; **18/2/0** with action delays; **12/8/0** with changed starts. These are descriptive results on a small fixed-layout test, not statistical proof of broad cooperation. The corpus had **0 of these 20 exact joint starting-pose/facing tuples as training sequence starts**. This comparison does not establish unseen-layout or unseen-partner performance, and individual poses can occur elsewhere in training histories.

Selection: keep **v5 as the clean-start throughput reference**, and **v7 as the tested robustness candidate**. Both models remain frozen; neither result is hidden by replacing the other's label. Varying timing and initial examples, with causally valid supervision, helped the neural workflow tolerate the tested disruptions. Timing and pose augmentation were combined, so their individual contributions were not ablated. The supervision repair is logically validated, but its performance effect was not isolated against an equally trained uncorrected model.

Final evidence: `overcooked-v2/outputs/dish-learning/v7-final-comparison.json` records paired checks and input hashes. Per-condition results/protocols/episodes: `eval-v7-final-{standard,dropout05,jitter1}/` and `eval-v5-paired5000-{standard,dropout05,jitter1}/`. Log: `v7-final-paired5000.log`.

Actual neural replays: **`overcooked-v2/outputs/dish-learning/eval-v7-final-standard/replay.html`** and **`overcooked-v2/outputs/dish-learning/eval-v7-final-dropout05/replay.html`** (also GIFs). The delay replay contains actual proposed/executed actions saved in `episodes.json`; no teacher ran in evaluation.

### Current conclusions about what helped cooperation

1. **Complementary roles and physical handoffs worked:** the neural policy actually exchanges ingredients/partial plates/completed dishes; one chef chops/serves while the other assembles/washes. This is demonstrated workflow execution, not proof of adaptive task allocation.
2. **Use available plates and overlap work:** reduced visible idle waiting; v4 learned six dishes compared with earlier serial-policy results. These earlier training stages changed several things together.
3. **Prepare the next rice during current assembly:** same-code teacher ablation improves six→seven; the neural learner then reached seven in every one of 20 original final tests.
4. **Teach varied timing and reachable starts:** the new frozen policy improves held-out delayed execution 2.3→6.3 mean and changed starts 0.35→4.55, with a small standard-start loss 7→6.85.
5. **Keep labels causally learnable:** do not ask a chef to predict a hidden current partner-delay coin flip. This was checked with exact complete input histories and corrected before the selected robustness run.

Unproven: exact maximum dish count, all-start reliability, role switching, unseen partners/layouts, explicit communication, PPO-from-scratch success and commercial-game transfer. Next useful work is broader legal-start coverage plus inspecting the remaining failed coordination cases, while protecting the seven-dish clean reference. Searching for an eight-dish teacher is a separate throughput direction.

### Reproduction and next-session handoff

Use the isolated Metal runtime/environment described in the previous handoff for behavior-cloning training. V7 source is `overcooked-v2/sushi_imitation.py`; successful original command arguments:

```sh
sushi_imitation.py --dataset outputs/dish-learning/demonstrations-v7 \
  --output outputs/dish-learning/bc-cooperation-v7 \
  --init-weights outputs/dish-learning/policy-throughput-v5 \
  --updates 5000 --stop-after 1500 --save-every 500 \
  --padded-steps 640 --lr .0001 --max-seconds 900
```

This is an argument record, not a command to rerun over an existing directory. For exact continuation, keep the same source/data/config, use `--resume` and omit `--init-weights`; choose a larger `--stop-after`. Do not bypass checkpoint hash guards. For changed settings/data, create a fresh run and warm start frozen weights. The canonical v5 and robustness v7 are preserved independently; the interrupted v6 is not a selected model.

CPU evaluation example, from `overcooked-v2/`:

```sh
JAX_PLATFORMS=cpu \
'/Users/zuriz/School/se390/multiagent-cooperation/overcooked-v2/.venv/bin/python' \
evaluate_sushi_policy.py --run outputs/dish-learning/policy-cooperation-v7 \
  --output outputs/dish-learning/CHOOSE-NEW-EVALUATION-DIRECTORY \
  --episodes 20 --eval-seed 6000 --action-dropout .05 --no-replay
```

Seeds 4000–4019 and 5000–5019 have now been used for final tests; do not tune against them or call them unused again. Future selected candidates should use a new predeclared range, such as 6000–6019. No additional training/evaluation process remains running after this session. Goal work remains open: seven is not a proven maximum, and starting-pose failures remain. Nothing was committed or pushed; PR #43 is still the earlier Mac/PPO setup, not these experiments. Preserve untracked sources and ignored artifacts before any Git cleanup.

## Continuation: broader initial-pose coverage

The previous goal turn made progress: frozen seven-dish throughput and stronger delayed execution were tested, saved and documented. Rechecked current files/manifests; no previous job is live. The next intervention addresses the remaining changed-start failures rather than assuming more fitting on the same 17 traces will solve them.

A failed validation episode (seed1001, radius1/facing jitter) was recorded at `overcooked-v2/outputs/dish-learning/eval-v7-jitter-failure1001/replay.html`. The repository replay-analysis helper rejected its custom-simulator schema (`metadata.steps` differs from the original producer), so exact-step analysis using that skill was stopped. No causal episode narrative is claimed from that failed extraction. Aggregate evaluation protocols and independently audited simulator states remain valid evidence.

**Reason for new data:** v7 had only eight alternative action-derived initial poses, while the declared radius1+facing test spans five nearby floor positions × four facings per chef (up to400 joint combinations). Train broader factored coverage, without selecting poses from final seeds. Preserve the original standard-start, action-derived and delayed traces as anchors.

**Planned data:** explicitly declared empty-kitchen spawn variations covering all20 individual position/facing combinations for each chef, using two deterministic pairings (40 joint poses). Keep all food/plates/pots, time, scoring and rules at the pristine reset values; only initial positions and directions vary, exactly as in the existing pose evaluator. These are spawn-augmentation demonstrations, not claimed action-derived prefixes. They complement the existing real-prefix data. Each recorded episode still contains600 actual transitions under unchanged physics. Independent5% per-chef delays use the corrected causal supervision masks.

**Before training:** validate valid distinct floor poses and allowed directions; ensure all non-pose reset fields stay identical; independently replay every transition/observation; audit entire-history labels. Use a new versioned dataset/run and keep all old frozen models untouched. Validate on seeds1000–1009; any newly selected candidate needs fresh final seeds6000–6019. No new results are claimed yet.

### V8 dataset ready; training protocol

Six focused CPU tests passed (12.045seconds), including new spawn validation/reset preservation plus all previous no-augmentation, delay-mask and independent-replay checks. Log: `spawn-augmentation-tests.log`.

The new dataset contains **57 trajectories**: the prior17 anchors copied byte-identically, plus40 declared spawn variants. The deterministic pairings `(7i+3)%20` and `(11i+1)%20` cover every one of20 individual position/facing choices per chef twice, with40 distinct joint tuples out of400 possible combinations. No final-test poses or seeds selected these combinations. All40 teacher episodes served **seven dishes**, with every transition and pre-action observation independently replayed; all non-pose reset fields were identical to pristine reset.

Corpus audit: **64,973 supervised decisions**, **64,761 distinct recurrent input histories**, **zero conflicting supervised histories**. Directory: `overcooked-v2/outputs/dish-learning/demonstrations-v8/`; exact sources in its `sources/`; generation script/log `generate_sushi_v8.py`, `spawn-demonstrations-v8.log`; hashes/results `demonstrations-v8-audit.json`, `demonstrations-v8-label-audit.json`.

**Training declared before launch:** warm start frozen `policy-cooperation-v7`, fresh Adam, LR **0.0001**, global clip **1**, seed **42**, one full two-chef trajectory per update, padding **640**, first **1,500 updates** of declared5,000, save every500, max900seconds. Model/physics/learner unchanged (190,976parameters). Run: `overcooked-v2/outputs/dish-learning/bc-spawn-v8/`; log `bc-spawn-v8.log`.

Interpretation controls: uniform trajectory sampling now assigns17/57≈29.8% probability to the old corpus; expected exposure per file at1500updates is26.3, versus88.2 with17files. Corpus action-frequency weights also change; actual counts are saved in run metadata. More exposure to new poses, the data mixture, warm-start ancestry and training budget are not independent causal factors in this comparison. Any final jitter test may overlap some exact training joint starts; report overlap and separate covered/uncovered results rather than claiming all starts are unseen.

Validation remains standard/delay5%/radius1+facing on seeds1000–1009. Evaluate before increasing training budget; freeze any selected model before fresh6000–6019 final tests. Best v5/v7 models remain untouched.

A separate bounded CPU teacher-only check is exploring an eight-dish schedule by couriering/chopping cucumber before returned-plate washing, so the two work tasks can overlap. It must preserve all resources, rules and600steps and pass independent replay. No eight-dish neural or teacher result is claimed yet.

### Bounded eight-dish teacher search: no improved target found

Three teacher-only CPU variants were tested under the original rules/resources/geometry/600-step horizon. None produced eight; no new training target was selected. These runs used the existing per-state food/resource audits and had no deadlocks. Since no target improved, no dataset was exported or independently replayed; this is a limited negative search, not a proof that eight is impossible.

| Scripted variant | Dishes / delivery steps | Simultaneous non-STAY ticks |
| --- | --- | ---: |
| Existing baseline | 7:78,140,211,285,375,465,555 | 294 |
| Chop cucumber early only when washing needed | 7:same delivery steps | 310 |
| Chop early every cycle | 7:81,157,230,305,395,485,575 | 309 |
| Early chop when washing plus nearer native stations | 7:78,145,217,293,387,481,575 | 303 |

**Cooperation finding:** extra simultaneous activity alone did not improve output. The first variant increased concurrency294→310 yet did not change any delivery time. This suggests other dependencies on the delivery path dominate that change; it does not identify a proven global bottleneck. Preserved source/report/log: `overcooked-v2/outputs/dish-learning/eight-teacher-search/{search.py,report.json,search.log}`.

### V8 actual corpus action counts

Saved metadata confirms Metal backend and190,976parameters. Supervised counts in action-code order right/down/left/up/STAY/pickup-place/work: **[11805, 5975, 11036, 5417, 16144, 9554, 5042]**. Weights are inverse-square-root frequency normalized to mean1; these corpus counts, the changed data mixture and warm-start ancestry are recorded rather than treating the intervention as a pure isolated pose effect. Run metadata: `bc-spawn-v8/metadata.json`.

### V8 completed and selected by validation

Training completed **1,500updates in279.1seconds**, last sampled-trajectory teacher accuracy99.47%, weighted CE0.008491. These training metrics are not free-running success measures.

| CPU validation, seeds1000–1009 | Frozen v7 | V8 |
| --- | ---: | ---: |
| Standard starts, sampled | 7.0 (10/10 served7) | **7.0 (10/10 served7)** |
| Independent5% action delays | 6.5 | **6.5** |
| Radius1+random facing | 2.8 (4/10 served any) | **6.3 (9/10 served7)** |

Greedy V8 served7. The remaining zero-dish pose validation case was seed1001; no claim of all-start reliability. Broader initial examples helped this validation set while retaining its standard/delay means, but extra fitting/data mixture and warm-start ancestry are confounded. Files: `eval-v8-{greedy,validation,dropout05,jitter1}/`; log `v8-validation.log`, all under `overcooked-v2/outputs/dish-learning/`.

Selected and froze V8 before fresh6000–6019 tests. Frozen model/config/source snapshots: `overcooked-v2/outputs/dish-learning/policy-spawn-v8/`; weights SHA-256 **8d086d8bd3f18e751e08974145dae825f0e18ed4b1edaed93515c7b4e474c46a**. Parent: frozen v7; original v5/v7 remain intact. Full-state resume checkpoint: `bc-spawn-v8/checkpoints/update-001500`. Final evaluation compares frozen v8/v7 under all three declared conditions, with exact pose overlap stratification for the changed-start condition. Results pending.

### V8 fresh final results and stopping point

Completed the preregistered paired test on seeds**6000–6019**, with frozen v8/v7, identical start poses and imposed-delay masks. All physical delivery, plate/pot conservation and finite-prediction checks passed. Frozen v5/v7/v8 weights and saved inference-source manifests were reverified.

| Final condition | Frozen v7 mean / total | Frozen v8 mean / total | Episodes serving any: v7 / v8 |
| --- | ---: | ---: | ---: |
| Standard starts | 7.00 /140 | **7.00 /140** | 20/20 /20/20 |
| Independent5% action delays | 5.75 /115 | **6.40 /128** | 19/20 /19/20 |
| Radius1+random facing | 4.20 /84 | **6.30 /126** | 12/20 /18/20 |

The latest training mixture improved mean dishes in the two robustness conditions while retaining seven in every standard episode of this paired test. It did not solve every disruption: one delay episode served zero, and two changed-start episodes served zero. Seven remains the demonstrated count, not a proven maximum.

Exact input hashes, paired wins/ties/losses and training-start overlap groups: `overcooked-v2/outputs/dish-learning/v8-final-comparison.json`. Per-condition protocols/rows: `eval-v8-final-{standard,dropout05,jitter1}/` and `eval-v7-paired6000-{standard,dropout05,jitter1}/`; log `v8-final-paired6000.log`. Actual v8 replays: **`eval-v8-final-standard/replay.html`** and **`eval-v8-final-jitter1/replay.html`**, beneath the same output root.

- Changed-start subgroup **covered** (exact joint position/facing tuple as a training-sequence start): **2 episodes**; v8 mean **7.0**, v7 mean **7.0**.
- Changed-start subgroup **uncovered** (exact joint position/facing tuple as a training-sequence start): **18 episodes**; v8 mean **6.22**, v7 mean **3.89**.

These small subgroups are descriptive only; individual poses can occur in other training histories. No unseen-layout or unseen-partner claim is established. Data-mixture, extra updates and warm-start ancestry remain confounded, so do not attribute the whole gain solely to coverage.

**User requested wrap-up:** no more experiments launched. Preserve frozen v8 as the latest broader-start candidate and v5/v7 as references. Six current focused spawn/augmentation tests passed; the previous whole-suite checkpoint had82CPU passes/11optional Metal skips. No further whole-suite rerun was needed after the narrowly scoped spawn mode passed its default-preservation and new-branch checks.

Next session: review remaining failure mechanisms with sushi-compatible validated state tooling, consider more joint-start coverage or recovery demonstrations, and keep throughput search separate. The bounded three-variant eight-dish search did not improve output; it was not exhaustive. Seeds6000–6019 are now used final-test data too. Use a new predeclared final range, such as7000–7019, for future selected candidates. Goal is paused, not declared complete. All changes and artifacts remain local/uncommitted/unpushed.
