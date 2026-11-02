# Saved models

## Story 1-3 sushi: policy-spawn-v8

`oc2_sushi_1_3/policy-spawn-v8/` contains the complete frozen v8 policy for
the local Story 1-3 simulator. Its ten original files total 846,902 bytes
(about 0.8 MiB), so they are tracked directly in Git.

- `params.msgpack`: shared CNN+GRU weights for both chefs, 190,976 parameters.
- `config.json`: architecture, observations, kitchen rules and training settings.
- `metadata.json`, `summary.json`, `FROZEN.json`: original training provenance,
  training metrics, and the model-selection record.
- `sources/` and `source_manifest.json`: exact saved learner, algorithm,
  environment and evaluator sources with SHA-256 hashes.
- `SHA256SUMS`: hashes of all ten original files.

Original provenance is preserved byte-for-byte. In particular, the original
Git commit and dirty flag describe the training run before its sources were
committed. The frozen folder contains inference weights, not optimizer/RNG
resume state or the demonstration dataset. Its training-loss summary is not
an evaluation result.

V8 learned through behavior cloning from 57 audited demonstrations. Final
paired tests on seeds 6000–6019 used 20 episodes per condition, 600 steps each:

| Condition | Mean dishes |
| --- | ---: |
| Standard start | 7.00 |
| Independent 5% action delay | 6.40 |
| Nearby starting positions and random facing | 6.30 |

Packaging check: all ten original files passed SHA-256 verification, and one
600-step CPU evaluation from this checked-in model path served seven dishes,
including washing and reusing plates. No training was run during packaging.

All standard episodes served seven. Some perturbed starts/timings still fail.
This is fixed-layout simulator performance; optimality, unfamiliar partners
or layouts, and commercial-game transfer have not been established. See the
[learning guide](../docs/sushi-training.md) for comparisons and limitations.

## Run a saved policy

Initialize the pinned submodule and use the full CPU learning dependencies
from the [setup guide](../overcooked-v2/README.md#tiny-sushi-ppo-check).
From `overcooked-v2/`, with that Python environment active:

```bash
JAX_PLATFORMS=cpu python evaluate_sushi_policy.py \
  --run ../model/oc2_sushi_1_3/policy-spawn-v8 \
  --episodes 1 --eval-seed 6000
```

This performs CPU inference without learning or modifying the saved model.
The printed output directory contains the summary, protocol and `replay.html`.
Use `--episodes 20` to repeat the recorded standard test; add
`--action-dropout 0.05` or `--spawn-jitter 1 --random-facing` for the respective
robustness condition. These are previously used test seeds, not fresh tests
for selecting future models.

To check file integrity, from `model/oc2_sushi_1_3/policy-spawn-v8/`:

```bash
shasum -a 256 -c SHA256SUMS
```

Keep frozen versions unchanged; give new models new version folders. Large
training datasets, optimizer checkpoints, raw logs and generated replays stay
under ignored `outputs/` directories.
