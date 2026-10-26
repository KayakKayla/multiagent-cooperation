# Story 1-3 simulator comparison

This audit covers one cucumber-sushi workflow in the installed macOS game,
observed on 2026-10-25 with bridge v0.2.0. All preparation in this level was
performed by the user; the bridge only read state. It does not establish policy
transfer or identify every mechanic in Overcooked! 2.

The comparison below describes the pinned upstream simulator. The separate
[`oc2_sushi_1_3` prototype](../overcooked-v2/scenarios/oc2_sushi_1_3/README.md)
implements several observed mechanics locally; its geometry and timing still
need calibration. These gaps are not a claim that those mechanics remain absent
from that prototype.

## Confirmed differences

| Area | Evidence from the real game | Current pinned OvercookedV2 | Implication |
| --- | --- | --- | --- |
| Movement | Earlier bounded inputs produced continuous world-space displacement. Stations and chefs have world positions. | Directional actions move on a discrete grid. | A controller must translate grid/task decisions into bounded movement and verify arrival. Collision and turning calibration are still needed. |
| Preparation | Tutorial lettuce progressed from 0 to 2/7 while on a chopping board. In 1-3, the user chopped cucumber and the plate then reported a `Cucumber` ingredient. | No chopping station or chopping-progress state. | Chopping requires a preparation mechanic if it is part of the training task. The plate's ingredient label alone does not encode the full preparation history. |
| Cooking | One rice ingredient in a pot progressed from empty/0 to about 4/12, then `Cooked` at 12.55/12 game-time units. | Default cooking starts after three ingredients; a pot timer counts down 20 simulator steps. Optional explicit start-cooking interaction already exists. | Match recipe-specific cooking conditions and action timing. Game-time units are not calibrated to simulator steps. |
| Containers | The same cooked pot moved from a stove to an ordinary counter, retaining contents and cooking state. | Pots are static grid cells, not movable inventory objects. | Pot ownership and transport require additional state/transitions if the target task allows them. |
| Plate contents | The same plate progressed from cucumber alone to cucumber, a cooked rice subtree, and seaweed. Transferring the rice reset the emptied pot to zero progress. | Interaction logic places ingredients into pots and transfers cooked dishes onto empty plates; it does not implement this preparation chain or adding ingredients to a nonempty plate. | Recipe preparation/assembly needs more than renaming the simulator's ingredients. |
| Plate cycle | After the user served the sushi, its plate left the snapshot and a `DirtyPlateStack` appeared at the return station. After manual washing, that stack disappeared and a `CleanPlateStack` plus a new empty plate appeared at `DryingPart`. | Plates come from a plate pile; there is no dirty-plate return or washing station/state. | Finite plate supply, delayed returns, and washing create tasks and resource constraints absent from the simulator. Return delay and wash duration have not been calibrated. |

Simulator evidence is in
[`common.py`](../overcooked-v2/JaxMARL/jaxmarl/environments/overcooked_v2/common.py),
[`overcooked.py`](../overcooked-v2/JaxMARL/jaxmarl/environments/overcooked_v2/overcooked.py),
and [`settings.py`](../overcooked-v2/JaxMARL/jaxmarl/environments/overcooked_v2/settings.py).
The current three-onion Coordination Ring configuration is described in
[`overcooked-v2/README.md`](../overcooked-v2/README.md). The simulator already supports
multiple ingredients/recipes and restricted observations; those are not entirely
missing capabilities.

## Checks still needed

- Read order requirements, deadlines, score, and delivery events. The bridge currently exposes none of these; a missing bridge field is not evidence of a simulator mismatch.
- Read washing progress and plate-stack counts. Current evidence identifies stack names and before/after objects, not exact stack membership or intermediate wash progress.
- Observe a raw ingredient completing its chop and record the replacement ID. Existing plate contents and partial chopping checks do not capture that exact transition.
- Verify overcooking/burning behavior if it is included in the target task; only uncooked, partially cooked, and safely cooked rice have been checked.
- Calibrate a blocked move, turning, chef-to-chef obstruction, and movement-to-grid mapping.
- Decide whether the target policy chooses low-level inputs or tasks, and what each agent may observe. Full bridge snapshots do not match the current baseline's restricted observation.

## Bridge work is separate

The installed bridge still offers only `state`, bounded `move`, empty-handed
`pickup`, and `release`. Automated placing, chopping/washing, and independent
two-chef control are not implemented. Native keyboard automation proved unreliable
in the menu and was abandoned for this check. Geometry coverage is partial, and
the snapshot has no dedicated washing-progress or dirty-plate field.

The observed plate cycle establishes removal, return, and cleaning. It does not
independently establish correct-order acceptance or reward: the user performed
delivery, and no order/score telemetry was available. The two controlled chefs in
the later snapshots also made `local_single_player` false; this read-only check
does not extend the original bridge's single-player control guarantee.

Evidence remains local and ignored in [`outputs/level-1-3/`](outputs/level-1-3/)
and [`outputs/kitchen-snapshot/`](outputs/kitchen-snapshot/). This bridge extension
changes no simulator code. Use the remaining gaps to prioritize calibration and
the intended cooperation experiment; not every real-game mechanic needs to enter
the simulator.
