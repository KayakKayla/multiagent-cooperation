# Replay evidence and conventions

The current producer is `overcooked-v2/preview.py::build_replay_data`; `overcooked-v2/replay.py` embeds its schema-v1 payload in `<script id="replay-data" type="application/json">`. Read those sources if the format changes. The extractor deliberately supports version 1 only.

## Fields

- `metadata`: layout, mode, episode index, seed, action count (`steps`), starts, agent view size, upstream revision, frame stride, and display timing. Scripted and randomized starts are different conditions.
- `steps`: initial state 0, followed by every post-action state. A 400-action episode has 401 states. In state `t`, the recorded action caused transition `t-1 → t`; inspect the preceding state to explain its effect. Initial actions and action codes are null.
- `agents`: ID/index/color, position, numeric direction and readable facing, numeric inventory and readable holding, numeric action code and readable action. Use stable IDs (`agent_0`, `agent_1`), optionally red/blue; legacy GIF numbering can differ.
- `pots`: stable index/location, encoded contents, ingredient count, remaining cooking timer, and readable status. Decreasing timers are progress even when chefs stay still.
- `grid`: indexed `[row][column]`, with each cell `[static_object, dynamic_contents, timer]`. Agent positions separately use zero-based `[column, row]`. Static objects include empty floor 0, wall/counter 1, delivery goal 4, pot 5, plate pile 9, and ingredient piles starting at 10. Verify uncommon objects against the pinned `common.py` before interpreting them.
- Dynamic contents encode plate as bit 0 and cooked as bit 1; each ingredient count occupies two bits starting at bit 2. For the three-onion layout, onion=4, three onions=12, cooked pot=14, plated soup=15. Do not generalize readable `onion` labels to every recipe; the producer uses generic ingredient labels for other codes.
- `recipe_code`/`recipe`: current recipe, potentially changing during an episode.
- `reward`, `correct_dishes`, `wrong_dishes`: events for the transition into that state. Simultaneous deliveries may exceed one and correct/wrong events can occur together.
- `totals`: cumulative shared raw team reward counted once, and correct/wrong deliveries. No training shaping bonuses. Use recorded values rather than assuming each delivery is worth 20 on every layout.
- `frames`: embedded PNGs, one per state, omitted from helper output. The GIF samples every second step and cannot substitute for exact action evidence.

## Interpretation limits

Movement commands can change facing without changing position. Inspect the intended destination, walls, the other chef's simultaneous action, and neighboring states before naming the cause. Unchanged inventory after `interact` is not sufficient to call the action ineffective: pot contents, counters, deliveries, or recipe state may change.

The payload is spectator state, not a log of policy observations or internal plans. `agent_view_size` describes restricted observation configuration; it does not record what the chef attended to or understood. Apparent blocking, task division, redundant work, and missed opportunities require contextual interpretation.

Evaluation HTML records episode index 0, not all evaluated episodes. One episode's score is not a run average or evidence of generalization. The saved 18-soup schedule is a scripted reference with original fixed starts; it is not a trained policy or a proven optimal benchmark.

`frame_ms` is viewer timing, not simulator runtime. Cite simulator steps; do not translate a waiting interval into real seconds using playback speed.
