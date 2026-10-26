# Overcooked! 2 local integration test

This small bridge reads chef state and sends short movement/pickup commands to the
installed commercial Steam game on macOS. No model is trained by these commands.

Version 0.2.0 adds a read-only kitchen catalog to `state`. The original four
operations and their control limits remain the same.

## Verified on this Mac (2026-10-25)

The Steam/BepInEx launch, Python connection, and live chef state passed in the
single-player tutorial. Three 250-ms movements displaced the selected chef by
1.56, 1.56, and 1.44 world units, each followed by zero measured idle drift.
Three separate pickup presses acquired fresh `Lettuce` objects, confirmed visually.
Manual controls remained usable afterward. Explicit release, disconnect recovery
(observed after 0.216 seconds, before the 500-ms deadline), paused movement
rejection, and rejection of the old scene token after restarting the level passed.
Live checks used the same Python client/action helpers provided here, with the
user preparing each position. The interactive wizard itself was not run end to end.

The local [acceptance summary](outputs/live-acceptance/summary.json) and
[exchange log](outputs/live-acceptance/events.jsonl) contain exact state/command
evidence. They are generated, ignored artifacts. Current offline checks passed:
75 C# assertions and eight Python tests. This establishes local control of this game build;
trained-policy transfer, training throughput, and other platforms remain untested.

## Checkpoint 1: load the plugin

The game stays in its Steam installation. Our plugin uses BepInEx 5.4.23.6 and the
game's Unity 2017.4.8 / .NET 3.5 API. A project-local ARM64 .NET SDK compiles an
AnyCPU DLL; the Intel game and mod loader run under Rosetta on Apple Silicon.

From this directory:

```bash
python3 setup.py dependencies
python3 setup.py build
python3 setup.py install --previous-launch-options ''
```

Downloads come from the official BepInEx GitHub release and Microsoft, and are
checked against pinned SHA-256/SHA-512 digests. Dependencies and game assemblies
are not committed. Set `--game-dir` for a different Steam installation.

First confirm an ordinary Steam launch reaches the title screen, then quit the
game. In Steam → Overcooked! 2 → Properties → General → Launch Options enter the
line printed by `setup.py install`. For the current installation it is:

```text
"/Users/zuriz/Library/Application Support/Steam/steamapps/common/Overcooked! 2/run_bepinex.sh" %command%
```

Launch through Steam again, then run:

```bash
python3 setup.py check
```

Success requires fresh `SIMS_BRIDGE_STARTUP` and `SIMS_BRIDGE_READY` lines in the
game's `BepInEx/LogOutput.log`. Merely reaching the title screen does not prove the
plugin loaded. If the game closes or no log appears, preserve the log and diagnose
the loader before proceeding. After installing a new build, restart the game.

## Read state, move, and pick up

Open a local level with one player. The single-player game may show two chefs;
only the currently selected chef can receive a bridge command. Switch chefs using
the game's normal controls if necessary.

```bash
python3 client.py state
```

This lists each `chef_id`, the current `scene_token`, position, facing, selection,
readiness, and held-object identity. Chef IDs/tokens must be read again after
loading/restarting a level. The bridge also reports whether the game is paused
and in a round. `can_control` requires the game window to have focus; state reads
still work while the game is in the background.

## Read-only kitchen snapshot

```bash
python3 client.py state
python3 client.py state --json > kitchen.json
```

`state` prints station/item summaries. `--json` writes the complete response,
including game-assembly hash and plugin version, to stdout; the report location
goes to stderr so redirected output stays valid JSON. The response remains
available in the usual local report as well. Reads work with a loaded, paused
game in the background; no movement, pickup, or other interaction is sent.

The additive `state.kitchen` object has `schema_version: 1`, `available`,
`coverage: "supported_components"`, `stations`, `items`, and `warnings`:

| Data | Meaning |
| --- | --- |
| Station identity and position | GameObject instance ID, scene, world-space position, and attach point when present. Native grid coordinates and grid-manager ID are included when available. |
| Station roles | Derived from actual components: dispenser, chopping board, cooking/heated station, serving station, plate stack/return, washing station, disposal, conveyor, or attachment surface. Component names accompany them. Unclassified grid markers retain the `grid_location` role. |
| Dispenser and surface contents | The current dispenser prefab name/static ingredient definition, attached item ID, and plate-stack size when available. |
| Item identity and ownership | Physical item instance ID/name, world position, held-by-chef ID or station ID, and attachment state. This includes held and placed items as well as unattached objects. |
| Ingredients | Live container contents and assembled recipe definitions, preserving repeated ingredients. Definition names/IDs come from this game build. |
| Chopping | Native completion fraction, stage count, and next prefab for items with a workable component. A fully chopped ingredient may replace the old object with a new ID and no chopping component. |
| Cooking | Raw accumulated/required game cooking time, completion fraction clamped to 0–1, native state, required station type, and native cooked/burning flags. Cooking speed may differ from wall-clock time. |

IDs are valid only in the accompanying `scene_token`. Station `item_id` references
an entry in `items`; item `held_by_chef_id` references the existing `chefs` array.
Items are labeled `held`, `station`, `unattached`, or `attached_unknown` from
explicit game ownership/getters. Conflicting chef/station ownership is marked
`ambiguous` with a warning. No nearest-counter guess is used.

Unavailable optional fields are `null`. `contents_known: false` means contents
were unavailable, rather than an empty container. `warnings` reports failed reads
from individual components; other entries remain available. Recipe nesting has
a depth limit of eight and reports truncation. Responses are bounded at 4 MiB in
the client; incoming command limits remain 8192 bytes.

This is a catalog of supported active game objects, **not a complete geometry or
walkability map**. Decoration, walls/floor collision meshes, order queues, rewards,
and unsupported mechanisms are not reconstructed. Native grid coordinates are
not converted into OvercookedV2 cells, and partial-observation masking is not
applied. Absent progress components do not imply an item is raw or finished.

The tutorial live check found 30 station objects, all three ingredient dispensers,
stable paused reads, explicit chef/item ownership, and partial chopping progress
(lettuce 2/7 and tomato 6/7). A tracked lettuce changed from held by Chef 2 at
zero progress to attached to a chopping board at 2/7 progress with the same item ID.
Evidence is saved locally in
[`outputs/kitchen-snapshot/`](outputs/kitchen-snapshot/). Raw tutorial vegetables
expose a prefab name and workable progress but no assembled ingredient definition.
Plates inside a stack may be `attached_unknown`; the tutorial rubbish bin currently
appears as an attachment surface with its native `RubbishBin` component listed.
The tutorial timer expired before the prepared lettuce/nonempty plate check.
A subsequent Story 1-3 check verified a manually chopped cucumber on a plate:
both container contents and assembled composition report one `Cucumber` ingredient,
and the plate links to its counter. This definition has no separate chopped flag.
Story 1-3 evidence is saved locally in [`outputs/level-1-3/`](outputs/level-1-3/).
Story 1-3 also verified rice cooking from 0 to about 4/12 and then `Cooked` at
12.55/12 game-time units, retaining its pot identity when moved from hob to
counter. The completed plate reported cucumber, cooked rice, and seaweed, while
the emptied pot reset to zero progress. Prepared-item replacement IDs, burning,
and washing progress remain unverified live. After manual serving, the sushi
plate disappeared and a `DirtyPlateStack` appeared at the plate-return station;
after washing, it was replaced by a `CleanPlateStack` and a fresh empty plate at
the drying area. This verifies the observed plate cycle, not exact washing
progress, stack counts, correct-order acceptance, or delivery reward.

See [`SIMULATOR-GAPS.md`](SIMULATOR-GAPS.md) for the resulting comparison with
the pinned simulator and the remaining measurement gaps. Story 1-3 checks used
manual controls and read-only bridge requests; they do not verify automated
placing/chopping/washing or independent two-chef control.

### Manual snapshot checks

Load and pause a level, then read two snapshots to compare stable station IDs,
positions, and item ownership. Manually pick up an ingredient, place it on a
chopping board, and chop briefly, pausing between reads. Check that ownership
links and progress follow the actual item. When it is fully chopped, check the
replacement item's name/definition and ID rather than assuming the old ID persists.
On a cooking level, put ingredients into a pot/pan and compare its contents and
cooking progress across two reads. These are observational checks; perform all
game interactions yourself.

## Movement and pickup checks

For the first movement check, place the selected chef on clear stationary floor
and release your keys. Substitute the ID and token printed by `state`:

```bash
python3 client.py move --chef CHEF_ID --scene-token SCENE_TOKEN --x 1 --y 0 --duration-ms 250
```

The three-second countdown lets you focus the game window and resume the level
before input is sent. Watch for a short move, followed by a stop. The client also
checks recorded displacement and a later idle sample. Use `--delay 0` when running
from a separate terminal without changing the game's focus. `x`/`y` are the game's
raw directional input axes, with vector length at most one; they are not world
coordinates or a destination.

Next, manually place the same chef empty-handed beside an ingredient dispenser,
facing it. Then run:

```bash
python3 client.py pickup --chef CHEF_ID --scene-token SCENE_TOKEN
```

The client sends one pickup press, then checks that held-object state changes from
empty to occupied. Visually confirm that it is the expected ingredient. A failed
pickup is never automatically retried. An occupied hand is rejected so this check
does not accidentally drop or throw an existing item.

To run all acceptance checks with prompts and manual resets:

```bash
python3 client.py verify
```

The wizard first compares two state reads around a manual movement, then requests
three movement/pickup repetitions, a disconnect check, and a scene-change check.
Use clear stationary floor, keep the other chef out of the path, and release all
manual keys during automated input. A failed check stops the wizard for diagnosis.
Record a visual check that the chef holds the expected ingredient after each
pickup. After a scene change, verify normal keyboard movement still works.

Each command creates an ignored `outputs/test-.../` report with `events.jsonl` and
`summary.json`. Exchanges include UTC timestamps, before/after state, plugin
version, and the installed `Assembly-CSharp.dll` SHA-256 hash. The report retains
failed checks too. Passing an individual command does not establish that the
whole acceptance procedure passed.

## Interface and safeguards

One persistent client at a time connects to TCP `127.0.0.1:8765`. Each UTF-8 command
is one flat JSON object followed by a newline (maximum 8192 bytes). The response
is one JSON object and newline. A command is acknowledged once accepted; use a
later `state` request to observe its effects.

| Operation | Fields in addition to `op` and optional string `request_id` |
| --- | --- |
| `state` | none |
| `move` | `scene_token`, integer `chef_id`, numeric `x`/`y`, integer `duration_ms` (1–500) |
| `pickup` | `scene_token`, integer `chef_id` |
| `release` | none; releases the current automated input |

Successful responses include `ok`, `request_id`, `plugin_version`,
`assembly_sha256`, and `state`; accepted actions also include `before`.
Errors include `ok: false`, `error`, and `message`. State includes `scene`,
`scene_token`, `frame`, `monotonic_seconds`, pause/round flags, input activity,
and `chefs`. Each chef includes its ID/name, world-space position/forward vector,
selection/readiness, and `held` (`known`, `occupied`, `instance_id`, `name`). A held
object's Unity instance ID/name is its identity for this test, not a recipe label.

Movement expires using a monotonic clock; pickup is a single press within a
150-ms window. Input wrappers also stop on disconnect, pause, loss of focus,
selection change, or scene change. Original input references are restored after
each action. All Unity reads/writes happen on its main thread, and the worker
handles only socket bytes. No teleporting, item spawning, game-speed changes,
or simulator/model changes are used. The plugin enables background updates while
loaded so Python can read state; the game must still be focused and unpaused for
commands. Input Monitoring permission for Steam is not needed by this bridge.

Change `Bridge/Port` in the generated
`BepInEx/config/ca.waterloo.sims.overcooked2bridge.cfg`, restart the game, and use
`python3 client.py --port PORT state` if port 8765 is occupied. The service always
binds to loopback. It has no authentication and is intended only for this local,
user-supervised feasibility test.

## Automated checks

```bash
bash test.sh
```

The C# harness compiles the actual protocol, JSON writer, transport, and deadline
gate sources against the local SDK. It tests malformed messages, duplicate fields,
numeric bounds, framing, nested state serialization, timeouts, reconnection, and
real socket disconnects. Python checks response correlation, report metadata,
incomplete replies, and the absence of pickup retries. These run without the game;
they do not replace the live acceptance checks above.

## Remove the test

Quit the game. Restore the Steam launch option recorded in
`outputs/install.json` (it was empty for the first installation here), then run:

```bash
python3 setup.py uninstall
```

The installer records the exact files it owns and refuses to overwrite an
unrelated mod installation. Removal checks file hashes and leaves generated
BepInEx logs/configs available for diagnosis. The game application bundle is not
rewritten. Keep the install manifest until removal is finished.

## Sources

- [BepInEx 5.4.23.6](https://github.com/BepInEx/BepInEx/releases/tag/v5.4.23.6)
- [Launching through Steam](https://docs.bepinex.dev/articles/advanced/steam_interop.html)
- [Overcooked2-TAS control example](https://github.com/gua248/Overcooked2-TAS/blob/master/plugin/TASControl.cs)

Game-specific members must match this installed build. The source example is a
reference for the approach, not a plugin installed by this project.
