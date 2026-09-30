# Agent context

- Code: this repository; JaxMARL is a pinned submodule in overcooked-v2/JaxMARL. Keep upstream unchanged unless simulator edits are explicitly requested.
- Local course documents: C:\Users\Rowan\Documents\School\SE\3B\SE390.
- Project: Waterloo SE 390, team Sims (5 members); research on multi-agent cooperation using Overcooked-style environments. Study coordination failures and possible improvements to teammate modeling, task allocation, communication, or planning.
- Evaluation interests: team reward, completion rate, coordination efficiency, unseen teammates, and unseen layouts. Specific intervention and experiment design remain undecided.
- State (2026-09-30): OvercookedV2/JaxMARL setup in `overcooked-v2/`; Python 3.12 environment at `/home/ro/.venvs/se390-overcooked-v2` in Ubuntu-24.04 WSL2. RTX 3070 simulator/rendering and tiny PPO check (2 updates, checkpoint save/reload/inference) verified. Checkpoint is a setup artifact, not a useful trained baseline. No running job. See `overcooked-v2/README.md` for commands, pins, and local compatibility fixes.
- User selected OvercookedV2 as starting simulator. PPO is installed as an upstream baseline; research intervention remains undecided. Familiar-versus-unfamiliar partner evaluation is only a proposal.
- Simulator extensions are optional future work. Choose changes to test a specific coordination capability; distinguish trained-policy transfer from reuse of a training method. Results in one simulator do not establish broader generalization.
- User is new to ML; use concise plain language. Research work is hands-on and user-led: explain one next step, let them catch up/run it, and wait. Short verification checks of requested code changes are authorized; do not launch research experiments or longer training without user direction. Keep course guidance concise.
- Detailed abstract/course context: chat "Prepare for SE 390 MP1 check-in", thread `01a0d17d-dbd3-70a0-819f-c9fad2861659` (local). Course requirements are in the local PDFs.
- Keep this file concise; update confirmed decisions and current state, preserving the distinction between proposals and commitments.
