# Evaluation JSON version 1

`shared/evals_shared` uses only Python's standard library. It imports neither
JAX nor the web application. The simulator bootstraps that package from the
website folder; a sparse website install includes it through `pyproject.toml`.

The [synthetic example](../tests/fixtures/evaluation-v1.json) has three players,
one checkpoint assigned to all three, and two fully recorded episodes. It is
format test data, not a research result.

An upload is ordinary UTF-8 JSON with `format: "se390-evaluation"`, `version: 1`,
`source_evaluation_id`, timezone-qualified `created_at`, optional `title`/`notes`,
and these objects/arrays:

| Field | Meaning |
| --- | --- |
| `training_runs` | Source-run ID, name/notes, method, seed, progress unit, original training config, provenance and training device |
| `checkpoints` | Source checkpoint ID, source-run ID, exact serialized-file SHA-256 and checkpoint-specific progress (or null) |
| `participants` | Agent ID, display order, exact source checkpoint ID, effective observation/action settings |
| `environment` | Name/layout, effective environment config, action meanings, renderer description |
| `protocol` | Actual episode seeds and their meaning, reward definition, selection/reset/intervention settings, recording coverage |
| `summary` | Evaluation-wide statistics, separate from selected-episode values |
| `episodes` | Zero-based index, seed, actual action count, termination information, metrics and recording/null |
| `provenance`, `device` | Evaluation source/dependency and execution-device information |

`recording` retains the existing version-1 replay payload, without `frames` or
derived `presentation`. `steps[0]` is the initial state with no action;
`steps[t]` is the state after the attached action. Thus 400 actions have 401
states. Metadata identifies the correct episode and seed. Sushi recordings
preserve proposed/executed action codes and separate dropout/random override
masks; blocked environment movements remain distinct diagnostics.

`metrics.team_reward` is shared raw environment reward counted once. Existing
scenario metrics can be added to the metrics object. Additive fields and an
`extensions` object are permitted; a nonempty `required_features` list or a
different format/version is rejected until its semantics are implemented.
Progress units are named strings (current producers use `environment_steps`
and `optimizer_updates`); missing historical units/progress remain unknown.

`parse_document(bytes)` enforces the exact 20,971,520-byte upload limit and
rejects duplicate keys, non-finite values, malformed structure, invalid
references, inconsistent seeds/actions/totals, and bounded resource violations.
`validate_document(dict)` applies the same semantic validation without a byte
limit. All malformed types raise `ValidationError`. `canonical_json` gives
deterministic UTF-8 bytes; `content_digest` hashes every field, including source
annotations, for exact duplicate/conflict detection.

Bounds are 32 JSON levels, four million nodes, 10,000 episodes, 64 participants,
100,000 actions per episode, 256×256 grids and 65,536-character text values.
Training-run algorithm values must be null or bounded text, and seeds must be
null or unsigned 32-bit integers; malformed/object values and oversized integers
are rejected before a database preview/import.
These are format sanity limits, not evidence that the simulator supports every
combination. Executable/media/model fields, embedded media URIs and machine-local
paths are excluded. Producers scrub operational paths and environment flags,
while preserving scientific settings. No model is loaded by validation/import.

## Producing and converting

Both existing evaluator commands additionally write `evaluation.json`. They use
one common policy execution/iteration/export implementation. Defaults, seed
splits, separate recurrent memories, CPU-only specialized sushi evaluation,
sampled/greedy modes and perturbations remain unchanged. `--no-replay` still
disables the specialized local preview; all-episode JSON is still produced.
Local GIF/PNG/HTML tools continue to render the first episode.

Episode JSON is spooled as each native trajectory is released. Only the first
native trajectory is retained for the established local previews. The complete
JSON is fsynced then atomically renamed; an interrupted/erroring export cannot
appear as complete. Export size is printed. Exceeding 20 MiB never silently drops
episodes or states; the website rejects oversized files.

New training runs record UUID source identities independently of RNGs. Existing
checkpoint saves now include inference `params.msgpack` and `checkpoint.json`
with the exact file hash, actual progress and unit. Evaluators can select a
run root or one `checkpoints/update-NNNNNN` directory; intermediate progress is
read only from that checkpoint. This changes metadata/artifacts, not training
math or random streams. An initialization-only run has progress zero.

Historical runs and initialization-only runs receive independent random UUIDs
persisted in local `source-identity*.json` sidecars. Weights hashes identify exact
checkpoint bytes within a run; they never establish run sameness. Shared-player
assignments reuse the recorded identity, while independent initializations with
identical weights remain distinct. Copy the sidecars with the evidence to keep
identities portable; they are ignored by Git. Explicit historical source IDs are
persisted for repeated conversion. A supplied ID conflicting with an already
recorded trainer ID is rejected; website review can associate compatible aliases.
Historical source folders that are unavailable receive separate persisted IDs for
distinct recorded source references rather than merging runs by seed or weights.
Legacy evaluations also receive their own persisted evaluation UUID.

Run provenance captures immutable original training evidence. Resume history is
stored with the selected checkpoint and in evaluation
`provenance.training_checkpoint_provenance`; appending a resume does not change
the run evidence used for association. An explicit checkpoint-sidecar hash
mismatch leaves progress unknown and never falls back to the root's final
summary. A historical root with no sidecar can still use its recorded final
summary; intermediate directories never borrow that summary.

From the repository root on Linux/macOS:

```bash
PYTHONPATH=evals-sharing-website/shared python -m evals_shared.legacy \
  overcooked-v2/outputs/evaluate/RESULT --source-run-id 'run:your-stable-id'
```

On PowerShell set `$env:PYTHONPATH = "$PWD/evals-sharing-website/shared"` first.
Optional `--partner-source-run-id`, `--progress`, `--partner-progress` (only
independently known), and `--output` support historical cases. `--progress`
applies only to the primary checkpoint/shared-primary slots; it never supplies
progress for a distinct partner checkpoint, even when both checkpoints belong
to the same training run. `--partner-progress` requires a distinct checkpoint;
passing it for shared A+A slots is rejected instead of silently ignored.
Conversion reads local config/protocol,
statistics and only the inert replay JSON script; it executes no HTML and runs
no simulator. Original PNGs and derived scenes are discarded. Every statistics
row is retained; trajectories absent from the original output remain null and
the viewer labels them unavailable. Historical training device stays unknown
when missing; the conversion machine is never substituted. If the original
source folder was moved away, saved `partner-config.json` and each policy's
recorded training seed remain authoritative for the partner. A specialized
saved config containing effective evaluation overrides is retained as evaluation
evidence; it is not relabelled as an unavailable original training config.
Evaluation creation
time uses the original timestamp encoded in standard output folder names, with
an explicitly labelled file-mtime fallback for nonstandard historical folders.

## Measured size and verification

The checked-in 18-soup, 400-action reference has 401 recorded states: 372,918
canonical JSON bytes (9,465 gzip bytes) without frames/presentation. Repeating
similar recordings 50 times would use about 18.6 MB before outer metadata;
layout/state complexity and diagnostics can increase this. The three-player,
two-episode synthetic example is 4,910 canonical bytes. These are measurements
of particular recordings, not a universal capacity promise.

Tests under `tests/test_format.py` and `tests/test_export_legacy.py` cover bounded
parsing, adversarial types, semantic references, action alignment, totals,
variable players, atomic failure, privacy and honest partial legacy conversion.
Simulator regression tests cover policy assignment/memory resets and specialized
sampled/greedy/perturbation behavior without research experiments or training.

The initial refactor was additionally checked against the original evaluator
source using `overcooked-v2/tests/check_evaluation_compatibility.py`. From the
`overcooked-v2` directory with its installed simulator environment:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python tests/check_evaluation_compatibility.py
python -m unittest tests.test_evaluate tests.test_sushi_policy_evaluation \
  tests.test_evaluation_export tests.test_replay -v
```

The standalone check pins the original Git revision in its source (that object
must be available locally). It compares every returned tree leaf exactly across
12 original/refactored rollouts: two sushi sampled seeds, sampled/greedy with no
noise, dropout, random actions, and pose/facing jitter, and upstream shared and
cross-policy execution. Horizons are three/four steps with newly initialized
weights. It also validates a real two-episode upstream JSON export. The original
memory-policy regression tests independently check per-chef assignment and
memory reset. Initial results were 12/12 exact comparisons, 16 evaluator tests,
five replay tests, two checkpoint-artifact tests, and eight format/legacy tests
passing. No training or research experiments were run.

The metadata review regressions also cover moved A+B source folders (distinct
saved partner configs and seeds), independently supplied partner progress,
persisted/copyable initialization UUIDs, concurrent identity creation, and a root
sidecar/hash mismatch. `tests/test_export_import.py` exercises the actual shared
source-record builder through upload/review/import before and after a resume:
one run, two checkpoints, two evaluations, immutable original run evidence and
both checkpoint-specific resume histories retained.
