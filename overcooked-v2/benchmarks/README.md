# PPO performance experiments

These opt-in scripts benchmark the pinned Coordination Ring recurrent PPO on a
single CUDA GPU. They do not change the ordinary trainer, baseline configuration,
or JaxMARL submodule. Each candidate is generated from the local trainer with
asserted source anchors, and the generated source and helper snapshots are saved
with its results. Source changes are experimental until correctness and learning
results support adoption.

Use the project's Linux GPU Python from `overcooked-v2/`:

```bash
python benchmarks/ppo.py --output outputs/performance/my-baseline --samples 5
python benchmarks/sweep.py --output outputs/performance --group systems
python benchmarks/sweep.py --output outputs/performance --group refine
python benchmarks/sweep.py --output outputs/performance --group tuning
python benchmarks/sweep.py --output outputs/performance --group extras
python benchmarks/summarize.py outputs/performance
```

Do not run GPU cases concurrently. The sweep enforces per-case subprocess
timeouts and records GPU telemetry. Existing result directories are preserved;
use a fresh output root for a new experiment. Failed cases are retained for
inspection and require a new case name to retry.

The benchmark uses the full 256-kitchen, 256-step rollout and original 152-update
learning-rate/reward-shaping schedules. Compilation and initialization are timed
separately. Two real updates warm the executable, then three synchronized updates
are measured by default; use more samples for confirmation. Early exploratory
cases may explicitly use five samples. Single-update benchmarks omit production
checkpoint overhead; the extras group measures checkpoint costs separately.
All candidate initial weights, optimizer state, and random streams come from the
original trainer. A two-update comparison crosses a real 400-step episode reset.

For closer timing comparisons under variable desktop load, alternate baseline
and candidate executions in one process:

```bash
python benchmarks/paired.py --output outputs/performance/paired-unroll \
  --options '{"host_logging":true,"gru_unroll":4}' --samples 6
```

Run a bounded learning comparison without compressing the original schedules:

```bash
python benchmarks/ppo.py --output outputs/performance/learning-baseline \
  --stop-after 48 --eval-every 16 --episodes 32 --seed 42
python benchmarks/ppo.py --output outputs/performance/learning-candidate \
  --options '{"host_logging":true,"gru_unroll":4}' \
  --stop-after 48 --eval-every 16 --episodes 32 --seed 42
```

Use 152 updates for the complete original environment-step budget. Checkpoints
contain optimizer/environment/RNG state and can be resumed using `--resume`
with the same settings and a new output directory. Frozen evaluations use the
original policy network and physical delivery counter, sampled actions, shared
raw reward counted once, and fixed independent evaluation seeds. They omit
rendering and replay export. Per-episode results and parameter hashes are saved.
Learning comparisons include the same initialization and test seeds on both
sides. One seed cannot establish equal final learning performance.

Interpret three distinct claims separately:

- Bitwise agreement: every saved runner-state and metric array is identical.
- Mathematical agreement: equivalent equations agree within numerical tolerance;
  compiler arithmetic differences can still change sampled actions and learning.
- Learning effectiveness: independently evaluated reward/deliveries over training
  steps and measured training-execution time. Short-run curves are not evidence
  of equal final performance.

`phases.py` times collection and learning as separate executables. Those results
localize costs but can differ from a jointly optimized executable. `kernels.py`
contains the experimental batched reset and fused GRU; `variants.py` also supports
compact integer observations, recurrent scan unrolling, and reduced-precision
CNNs. Compact observations are restricted to this known integer schema.

Run correctness gates before using a candidate:

```bash
JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  python -m unittest tests.test_ppo_performance -v
```

`configs/coord_ring_cuda_graphs.yaml` explicitly enables the pinned XLA's default
CUDA command types; `coord_ring_fusion.yaml` enables only fusion capture. Both
preserve every baseline training/environment setting, enforced by a test. Use
an opt-in configuration for a new production run:

```bash
bash overcooked-v2/run.sh train --config configs/coord_ring_cuda_graphs.yaml
```

Measure both full production runs and independently score retained policies:

```bash
python benchmarks/quality.py --output outputs/performance --stage production
python benchmarks/quality.py --output outputs/performance --stage partial
python benchmarks/results.py outputs/performance
```

The production stage uses the unchanged launcher, ordinary checkpoint interval,
and original full training budget. It compares policy hashes at every retained
save and final runner/metric arrays. It evaluates training checkpoints on 32
fixed seeds and final policies on 100 separate seeds. The partial stage compares
three paired training seeds up to 48 updates, with optional deadline-based
skipping; it preserves the original schedules rather than compressing them to
48 updates. The final checkpoints of partial runs can be resumed exactly.
Use `--updates 32` on the partial stage for a shorter paired study; the original
learning-rate and reward-shaping schedules are still preserved.
Use `--seeds 42 --conditions reference unroll16 mb32` to restrict the study;
the unrolling condition is selected from existing timing results. Optional
`production_run.py --stop-after-checkpoint 10` retains the original schedule
while bounding the run at a saved checkpoint. `--reference-policies` stops at
the first saved weight mismatch rather than spending a complete training budget
on a failed bitwise-equivalence hypothesis. Process termination occurs only
after the checkpoint weights have been copied to the measurement directory.

Passing a two-update exactness check does not establish full-run equivalence.
Runtime-only compiler settings can change later learned weights even when the
PPO equations and all hyperparameters are unchanged. Inspect production hashes
and independent policy evaluations before treating such a setting as equivalent.

The extras group also compares native CPU training, two independent GPU seeds
vectorized together, and checkpoint/resume recovery. Two-seed batching measures
aggregate throughput; it does not promise a shorter wait for one model.

The fused-GRU equation/gradient check explicitly requests full float32 matmul
precision to distinguish implementation mistakes from Tensor Core rounding.
Default-precision and whole-training differences are measured by the actual
experiment probes. Machine details and generated measurements belong in ignored
local outputs rather than shared setup documentation.
