# Overcooked! 2 local integration test

This small bridge reads chef state and sends short movement/pickup commands to the
installed commercial Steam game on macOS. No model is trained by these commands.

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
evidence. They are generated, ignored artifacts. Offline checks passed: 58 C#
assertions and six Python tests. This establishes local control of this game build;
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
