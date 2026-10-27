# Overcooked! 2 Story 1-3 scenario

Status: runnable JAX-native grid prototype, verified on CPU on 2026-10-25.
It is a local scenario extension; the pinned JaxMARL implementation is unchanged.

## Run the first check

From the repository root on this Mac (local CPU environment already installed):

```bash
bash overcooked-v2/run.sh scenario --layout oc2_sushi_1_3
```

For a fresh Python 3.11+ CPU environment, from `overcooked-v2/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-scenario-cpu.txt
bash run.sh scenario
.venv/bin/python -m unittest discover -s tests -p test_sushi_scenario.py -v
```

The existing Linux/WSL environment also supports `bash run.sh scenario` with
`OVC_VENV` set to its path. The Windows launcher accepts
`.\overcooked-v2\run.ps1 scenario -Layout oc2_sushi_1_3` and runs through WSL.
That PowerShell dispatch has not been executed on this Mac.

The script needs no Steam game or model. Chef 0 supplies rice, chops, assembles,
and serves; Chef 1 cooks, passes ingredients/pots, and washes plates. They use a
shared counter to hand off items. The same plate is washed and reused for dish 2.
The calibrated script completed **386 steps and two deliveries**, preserving all three pots
and four plates and checking an independent ingredient ledger after every action.
The fixed prototype reward was 40, not the commercial game's score.

Outputs under `outputs/scenarios/` include `replay.html`, `initial.png`,
`assembled.png`, `final.png`, `trace.json` (every action, event, and state), and `summary.json`
(rules, assumptions, resource counts, and milestones). The replay opens offline
and uses the shared illustrated viewer, with pause, seeking, exact stepping, item
inspection, and an **Illustrated / Original** comparison toggle. Carried pots,
partial plates, chopping/washing progress, and dirty/clean plate states are shown
from recorded data. GIF/PNG output keeps the original rendering. Use
`--output PATH` to choose a directory; that directory's generated files are replaced.

The user selected a new scenario in this repository, based on the commercial
game's Story 1-3 kitchen. It should be selectable alongside existing maps, with
grid movement and an optional set of rules for chopping, movable pots, sushi
assembly, and washing. A layout definition alone cannot add these mechanics.

## First useful version

- Two chefs using discrete grid movement and a fixed kitchen layout based on
  measured stations; walkable geometry still needs a manual fidelity check.
- Ingredient dispensers, counters, chopping boards, hobs, serving window, plate
  return, sink, and drying area based on the observed station locations.
- One recipe: chopped cucumber, cooked rice, and seaweed on a plate.
- Track raw/prepared ingredients, work progress, container contents, pot location,
  cooking progress, and clean/dirty plates explicitly.
- Support preparing and serving the dish, waiting for a plate to return, and
  washing it for reuse. Plate and food quantities must be conserved through the
  scripted cycle, except at ingredient dispensers and delivery.
- Keep pickup/place and work (chop/wash) distinguishable in the action interface;
  their exact encoding remains an implementation choice.

No throwing, dashing, continuous physics, dynamic kitchens, extra recipes, or
trained-policy transfer is required for this first version. Orders and rewards
must be labeled prototype rules until commercial-game telemetry is available.
One fixed requested recipe and a simple successful-delivery reward are sufficient
for an initial scripted check; they are not a reproduction of commercial scoring.

## Measured rules and open parameters

Evidence comes from bridge v0.2.0 snapshots and a v0.3.0 continuous recording
during manual play. The normalized
observations and source assembly hash are recorded in
[`reference_workflow.json`](reference_workflow.json) and
[`calibration_reference.json`](calibration_reference.json). Raw game captures stay local;
the simulator does not require the Steam game or bridge implementation to run.

| Rule | Current evidence | Needed before a fidelity claim |
| --- | --- | --- |
| Kitchen layout | Native/world station mapping verified: 1.2 world units per cell, origin (6, 7.2) in X/Z. Collider centers identify solid obstacles at (3,7) and (9,8), now blocked. | Full collision edges, boundary clearance, station reach and spawn cells remain approximate. |
| Chopping | One uninterrupted cucumber changed from `Cucumber` to `ChoppedCucumber` on the same board. Sampled active-work bounds were about 1.30–1.50 seconds. | Repeat trials before claiming a precise duration; 1.4 seconds is a representative estimate, rounded to six simulator steps. |
| Cooking | Required progress is 12 units; observed progress advanced at about one unit per native layer second until `Cooked`. | A 12-second rule is supported for this recipe. Interruptions, heat changes and burning need further checks. |
| Pot transport | The same cooked pot retained contents and cooking state after moving from a hob to a counter. | Check placement/transfer failures and whether cooking stops immediately off heat. |
| Assembly | The same plate held cucumber, a cooked-rice subtree, and seaweed; the emptied pot reset to zero. | Specify valid/invalid combinations and transfer rules explicitly. |
| Plate cycle | Configured wash is 3 seconds, supported by one observed wash. Dirty and clean stacks each contained one plate. Configured return delay is 10 seconds. | Pending-return telemetry failed during the recording; its backing-array fix needs a live check. Plate disappearance to dirty arrival was 9.5 seconds, which is not measured from delivery start. Stack behavior and wash interruption remain open. |
| Orders and score | No order/score telemetry in the bridge. | Add telemetry or retain explicitly simplified prototype rules. |

Unknown timing values may be configurable provisional parameters for development.
Record them as assumptions; do not silently present them as measured game rules.

## Repository integration

Keep `overcooked-v2/JaxMARL` pinned and unchanged. Define the scenario and its rule
extension locally, reusing upstream components where they fit. Existing layouts,
training configurations, and recorded baselines must retain their behavior.

The local `scenarios.make_env(layout, **kwargs)` factory selects
`oc2_sushi_1_3` and delegates other layouts to the original OvercookedV2 factory.
The `scenario` runner selects the extension; existing `smoke`, `train`, `evaluate`,
and `scripted` commands retain their original behavior. The PPO launcher and
evaluation/dashboard metrics are **not yet wired to train this new scenario**.
There is no claim of checkpoint compatibility or real-game policy transfer.

## State, actions, and observations

`SushiEnvironment` supplies functional `reset(key)`, `step_env(key, state, actions)`
(no automatic reset), and `step(...)` (automatic reset) with agent-keyed dictionaries.
It is JIT- and `vmap`-compatible. State is a fixed-size array pytree, with one
location per item slot and four finite plate resources. It is not the upstream
OvercookedV2 state schema. Food composition uses bits 1=cucumber, 2=rice, 4=nori;
pot progress distinguishes cooking rice from cooked rice, and only cooked rice
can enter a plate.

| Code | Action |
| --- | --- |
| 0–3 | Right, down, left, up; blocked movement still turns the chef. |
| 4 | Stay. |
| 5 | Pickup/place/transfer/serve, depending on hands and facing station. |
| 6 | Work on a chopping board or wash at a sink, with empty hands. |

Movement is simultaneous: same-cell collisions and swaps are blocked, and moving
into a vacated cell is allowed. Interaction priority alternates each step. Each
item can receive at most one unit of work per step even if both chefs work on it.
These are explicit prototype rules, not measurements of commercial collision or
cooperative work behavior. Invalid actions are no-ops with an `invalid_action`
event; passive cooking and return timers still advance.

Default observations are full-map `10 × 14 × 48` float arrays. Channels are
terrain one-hot (14), item kind (8), item composition (3), item progress (1),
self/other location (2), facing at self/other locations (8), and own held item
features broadcast across the grid (12). Visible carried items appear at their
chef's cell. `agent_view_size=2` produces `5 × 5 × 48` crops and does not reveal
an unseen teammate's facing or inventory. Queued returning plates are off-map.

## Explicit provisional rules

`Rules` configures durations and capacity. Defaults use 0.25 seconds per step,
12 seconds of rice cooking, an estimated 1.4 seconds of chopping, 3 seconds of washing,
10 seconds before plate return, 600 episode steps, and 32 item slots. Cooking and
washing are supported by native configuration and one observed cycle. The return
delay is configured in the game but its delivery-to-return interval was not fully
verified. Chopping is one measured estimate; the 0.25-second action interval is
still a modeling choice. Durations round up to whole steps (chopping becomes 1.5
simulator seconds).

Cooking advances only on a hob, retains progress off heat, and is capped at
completion; **burning is not modeled**. Interrupted chopping/washing retains
progress. Return, sink, and drying area each hold one item; returning plates
wait off-map if the return station is occupied. Completed washing waits for the
drying area to clear. Stack behavior is not reproduced. The bin discards food
without destroying plates or pots. The prawn dispenser is drawn but inactive.

Station coordinates are transcribed from bridge-native grid indices, with
column = X + 6 and row = 3 − Z. Outer walls, the extra bottom passage, the
serving cell, and rounded chef starts remain partly provisional. Two collider-derived
solid obstacles now replace incorrect floor cells. The lower passage at (3,8)
remains floor: real off-center movement was observed there, although the real chef
cannot necessarily stand on its nominal grid center with full clearance. This
coarse grid cannot reproduce that passage's exact shape. Script reachability and
recorded routes do not prove exhaustive tile-by-tile fidelity.

## Verification

Fifteen focused tests cover scripted reuse/conservation, invalid transfers,
simultaneous collisions and pickup, work limits, cooking on/off heat, blocked
return/drying stations, finite item capacity, episode termination/autoreset,
observation masking, and JAX JIT/batching. Five existing replay regressions and
an original `coord_ring` reset/step passed on CPU. Browser playback, seeking,
and backward/forward stepping were checked. No PPO updates were run.

The script compares normalized checkpoints with the tracked
[`reference_workflow.json`](reference_workflow.json), derived from the manual
game trace: partial/cooked rice, transported pot, assembled plate, emptied pot,
dirty return, and clean plate after washing. This checks the preparation sequence;
it does not validate commercial physics, exact durations, stack behavior, or score.
Generated summaries include source hashes and the reference comparison's scope.

## Further acceptance before training

1. Render the map and check geometry against the level snapshot and manual play.
2. Retain the verified deterministic recipe/plate-reuse script and invariant checks
   when revising mechanics or timing.
4. Compare semantic milestones with the real-game trace, normalizing object IDs
   and coordinates rather than expecting the commercial game's IDs in the simulator.
5. Integrate the new observation/action schema and metrics into a training launcher
   explicitly; freeze and document provisional mechanics before designing a comparison.

Training and research experiment design remain separate, user-directed steps.
