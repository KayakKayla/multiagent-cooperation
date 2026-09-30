# SE 390 multi-agent cooperation

Team Sims is studying cooperation in Overcooked-style environments. OvercookedV2 is our starting simulator; the research intervention and evaluation design are still undecided.

## Repository layout

- `overcooked-v2/smoke.py`: one random-action episode and a rendered preview.
- `overcooked-v2/train_check.py`: a tiny upstream PPO run, with checkpoint save/reload verification.
- `overcooked-v2/run.ps1` and `run.sh`: launch either check through the local Python environment.
- `overcooked-v2/setup.sh`: initialize the dependency and install pinned packages.
- `overcooked-v2/JaxMARL/`: pinned upstream Git submodule.
- `overcooked-v2/outputs/`: generated results, ignored by Git.
- `AGENTS.md`: concise project context for coding agents.

## Setup and checks

The verified local setup uses Ubuntu-24.04 in WSL2, Python 3.12, and an RTX 3070. See [the setup guide](overcooked-v2/README.md) for versions, compatibility settings, and requirements.

After cloning, run this from the repository root in Ubuntu:

```bash
bash overcooked-v2/setup.sh
```

This initializes only JaxMARL, at the pinned commit, and installs packages into a separate Linux virtual environment. It does not launch a check.

From the repository root in PowerShell:

```powershell
.\overcooked-v2\run.ps1 smoke
.\overcooked-v2\run.ps1 train-check
```

Both commands are bounded setup checks. The tiny checkpoint is not a useful trained baseline.

## JaxMARL dependency

The submodule keeps upstream source/history separate from our scripts while recording the exact version used. After pulling a change to its pin, run:

```bash
git submodule update --init
```

Our simulator and PPO algorithm currently use upstream code unchanged. Dependency updates or simulator modifications should be deliberate project changes.
