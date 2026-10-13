---
name: analyze-overcooked-replay
description: Explain one recorded OvercookedV2 episode and investigate coordination successes and possible problems using exact replay state and action evidence. Use for replay walkthroughs and episode diagnosis, not aggregate evaluation or training.
---

# Analyze an Overcooked replay

Produce a concise, plain-language chat report of one episode: what happened, where coordination worked, the most consequential supported problems, and one concrete follow-up check. Save Markdown only when requested. This skill does not authorize training, simulator changes, run comparisons, or interactive annotations.

## Locate and read evidence

Accept a local `replay.html` or a folder containing it. When unspecified, discover replays with `rg --files --hidden --no-ignore -g replay.html`, excluding Git internals. If several exist and context does not identify one, ask the user to choose; do not silently pick the newest or highest-scoring episode. If only a GIF is available, ask for the corresponding HTML replay before attempting exact-step diagnosis.

Read [the replay reference](references/replay-format.md) for field meanings and interpretation limits. Use the bundled standard-library helper, resolving its path relative to this skill:

```sh
python3 <skill-directory>/scripts/extract_replay.py <replay-file-or-folder>
python3 <skill-directory>/scripts/extract_replay.py <replay-file-or-folder> --start 100 --end 120
```

The helper reads embedded JSON without executing HTML or importing the simulator. Default output covers every state without images. If tool output would truncate, inspect metadata/totals and read the entire episode in bounded windows; never diagnose the whole episode from a truncated prefix. A window includes its preceding state as `context_state`. The output's `episode_totals` always covers the full episode, not just the window. Stop exact analysis on validation errors and explain what evidence is missing or inconsistent.

## Investigate the episode

First inspect the whole trajectory and outcome. Then revisit timestep windows around deliveries, inventory/pot changes, repeated failed moves, unproductive interactions, backtracking, competing use of space/resources, ready soup left uncollected, and successful handoffs or task division. Prioritize a few consequential findings over a list of every repeated action. Inspect kitchen images from the replay with available browser/image tools when spatial evidence is unclear; do not require a particular browser or change the source replay.

For each finding, cite exact steps or ranges and chef IDs, describe the observed actions and state changes, then distinguish the likely explanation from the observation. Check surrounding steps, both chefs, pot timers, facing, inventory, and nearby grid cells before interpreting inactivity or interference. Failed movement alone does not establish teammate blocking. Repeated interaction may be useful. Waiting while soup cooks can be appropriate. A recipe or counter transfer may make apparently duplicated work productive.

The replay records no explicit task plans or intentions. Describe apparent roles as inferences. The spectator sees more than a chef with restricted observations; do not assume a chef knew about a distant opportunity. Do not claim proven lost reward, an exact efficiency ceiling, or a policy-wide weakness from this episode. Keep step counts distinct from the viewer's playback time.

## Report

Link the source replay using its absolute local path. Briefly identify layout, mode, seed, start settings, episode length, raw team reward, and correct/wrong deliveries. Follow with a short episode narrative, successful coordination where supported, and a few evidence-backed problems with uncertainty made clear. If there is no clear failure, say so instead of inventing one; if there is no successful coordination, do not invent that either.

End with one specific follow-up observation or check that could clarify the leading finding. Suggest checks without selecting a research intervention or launching an experiment. Keep episode totals separate from adjacent run summaries; evaluation replays show only the first test episode. Answer a user's narrower timestep question directly, retaining full-episode context where relevant.
