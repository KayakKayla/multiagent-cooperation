# Agent context

- Project: Waterloo SE 390, team Sims (5 members); research on multi-agent cooperation using Overcooked-style environments. Study coordination failures and possible improvements to teammate modeling, task allocation, communication, or planning.
- Evaluation interests: team reward, completion rate, coordination efficiency, unseen teammates, and unseen layouts. Specific intervention and experiment design remain undecided.
- Code: this repository; JaxMARL is a pinned submodule in overcooked-v2/JaxMARL. Keep upstream unchanged unless simulator edits are explicitly requested. Favor simple wrappers with brief explanatory comments; preserve useful safeguards.
- User selected OvercookedV2 as starting simulator. PPO is installed as an upstream baseline; research intervention remains undecided. Familiar-versus-unfamiliar partner evaluation is only a proposal.
- `overcooked-v2/` contains the Coordination Ring PPO baseline, the local `oc2_sushi_1_3` scenario, evaluation, and replay tools. `overcooked2-bridge/` is an optional integration with the commercial game.
- Read [the setup guide](overcooked-v2/README.md) before code/setup work. It covers CPU/GPU setup, launchers, and cross-policy evaluation, which currently requires matching network and observation settings. Frozen sushi model usage is in [the model guide](model/README.md).
- Frozen evaluation supports `--partner-run` for a second compatible policy, with separate recurrent memory and both checkpoint identities recorded. See `overcooked-v2/README.md`.
- Simulator extensions are optional future work. Choose changes to test a specific coordination capability; distinguish trained-policy transfer from reuse of a training method. Results in one simulator do not establish broader generalization.
- Do not launch training or experiment runs without explicit user direction. Short verification checks of requested code changes are allowed.
- If present, read `.local/AGENTS.md` for this user's preferences, machine settings, and local run records. Personal context, machine settings, and generated outputs stay ignored by Git.
- Historical notes moved from this file are in [project-log.md](docs/project-log.md).
- Keep this file concise; update confirmed decisions and current state, preserving the distinction between proposals and commitments.
