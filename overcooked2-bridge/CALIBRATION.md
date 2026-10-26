# Story 1-3 calibration recording

One manual cucumber-sushi and plate-washing cycle was recorded on 2026-10-26 UTC
with bridge v0.3.0. No automated gameplay or training was used. The local recording
contains 1,875 samples over about 187 seconds, at a median interval of 0.100 seconds
and p95 interval of 0.104 seconds. Relevant game-layer clocks froze while paused.
Median state-capture cost was about 12 ms; median request latency was about 15 ms.

| Measurement | Evidence | Simulator choice |
| --- | --- | --- |
| Cucumber chopping | One uninterrupted work interval, approximately 1.30–1.50 seconds, ended with an explicit `Cucumber` → `ChoppedCucumber` replacement on the same board. | Representative estimate 1.4 seconds; rounded up to six 0.25-second steps (1.5 seconds). |
| Rice cooking | Required progress 12; progress advanced at approximately one native unit per game-layer second and reached `Cooked`. | Retain 12 seconds. |
| Washing | Configured 3 seconds per plate; one wash completed between sampled elapsed bounds of approximately 2.89–3.11 seconds. | Retain 3 seconds. |
| Plate return | Configured 10 seconds. Dirty plate appeared 9.5 seconds after the served physical plate disappeared; removal is not the delivery-start event. | Replace assumed 5 seconds with configured 10 seconds; exact event-to-event verification remains open. |
| Station mapping | Native-to-world spacing 1.2; native zero maps to world X=6, Z=7.2. All gridded stations fit this mapping, within 0.007 units including sink offsets. | Retain column=X+6, row=3−Z. |
| Solid obstacles | Collider centers cover previously open cells (3,7) and (9,8). | Block those cells. |
| Narrow lower passage | A chef was observed off-center in cell (3,8). Its nominal center has inadequate capsule clearance in the collider-bound approximation. | Keep the passage; exact continuous clearance cannot be represented by the current coarse grid. |

Dirty and clean stack snapshots each reported one plate. This is not a test of
larger stacks, exact scoring, orders, burning, interrupted work, all collision
edges, or every walkable tile. This single cycle is a useful calibration, not a
precise timing benchmark or a policy-transfer result.

The first v0.3.0 build incorrectly expected an `Item` indexer on the game's
`FastList` pending-return queue. It produced 100 optional-read warnings while
that queue was nonempty. Other state continued to record, including physical
dirty-plate arrival and washing. The reader now uses the backing `_items` array
confirmed in the installed assembly. A focused regression test covers a queue
without an indexer; the fix still needs a future live check. The installed build
used for this recording has not been replaced during gameplay.

Normalized observations, assembly hash, raw-recording hash, and limitations are
tracked in the simulator's
[`calibration_reference.json`](../overcooked-v2/scenarios/oc2_sushi_1_3/calibration_reference.json).
Raw evidence stays local and ignored in
[`outputs/calibration-level-1-3-session-1/`](outputs/calibration-level-1-3-session-1/),
with the paused preflight in
[`outputs/calibration-preflight-1/`](outputs/calibration-preflight-1/).
