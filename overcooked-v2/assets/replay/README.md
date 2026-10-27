# Shared replay artwork and renderer

The illustrated style approved with the 18-soup replay is the default HTML
spectator viewer. It does not change simulator transitions, observations, rewards,
checkpoints, or training. The pinned JaxMARL submodule is unchanged.

- `replay_scene.py` translates upstream and sushi recorded schemas to a derived
  `presentation` payload. Recorded `metadata`, `steps`, and original `frames` stay
  intact. Timing fractions use exported rules; older sushi recordings with no
  known cook limit show the exact remaining steps without an invented percentage.
- `replay_template.html` contains the shared responsive viewer, canvas compositing,
  playback, delivery event navigation, and exact-state inspection. Layout sizes,
  recipes, pot counts, chef identities, and episode lengths are data driven.
- `replay.py` embeds both atlases, presentation data, and original frames into a
  portable HTML file. Existing saved replays can be upgraded without dependencies
  or simulation using `python3 overcooked-v2/replay.py FILE_OR_DIRECTORY`.

## Artwork

`kitchen-atlas.png` and `sushi-atlas.png` are original transparent sprite sheets
generated with the built-in imagegen tool. They are committed assets; generation
is not required at runtime. No commercial-game artwork was extracted. Exact prompts
are in [KITCHEN-PROMPT.md](KITCHEN-PROMPT.md) and [SUSHI-PROMPT.md](SUSHI-PROMPT.md).
The second atlas used the first only as a style reference.

Each atlas has five columns and four rows. Source rectangles use the image's
actual width/5 and height/4, rather than assuming generation produced a requested
pixel size. The first contains directional red/blue chefs and soup equipment; the
second contains sushi items/stations and generic ingredient/recipe controls.

The viewer supports all mapped upstream terrain types and the local sushi
scenario. Unknown terrain, item kinds, or chef artwork fall back to Original with
an explanation. Additional ingredient identities use neutral pouches labelled by
simulator ID; they are not arbitrarily renamed tomato, mushroom, etc. Multiple
correct/wrong deliveries on one step are represented by their separate recorded
count deltas. No task intentions are inferred from actions.

Movement follows exact recorded grid steps. Walking/work animations and illustrated
GIF exports are not implemented; existing GIFs/PNGs retain the original renderer.

## Checks

12 replay/scene tests check state/action alignment, script audit equivalence,
escaping, mixed ingredient IDs, recipe switch changes, partial sushi plates,
carried pots, return-queue omission, custom/legacy timings, fallback, and idempotent
restyling. 16 sushi regressions and the 386-step two-delivery script pass.
Browser checks cover soup, sushi, a Grounded Coordination layout, short policy
episodes, simultaneous delivery events, no-delivery/initial-only episodes, and
unsupported-art fallback, with responsive layouts and original-frame comparison.
