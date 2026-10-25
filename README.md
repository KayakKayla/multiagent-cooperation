# SE 390 multi-agent cooperation

Team Sims is studying cooperation in Overcooked-style environments. OvercookedV2 is our starting simulator; the research intervention and evaluation design are still undecided.

## Repository layout

- `overcooked-v2/smoke.py`: one random-action episode and a rendered preview.
- `overcooked-v2/train_check.py`: a tiny upstream PPO run, with checkpoint save/reload verification.
- `overcooked-v2/train.py` and `configs/coord_ring.yaml`: a saved single-seed baseline using the paper's Coordination Ring settings.
- `overcooked-v2/evaluate.py` and `preview.py`: frozen before/after policy tests, per-episode scores, an interactive HTML replay, and an annotated GIF.
- `overcooked-v2/run.ps1` and `run.sh`: launch checks or user-started baseline training through the local Python environment.
- `overcooked-v2/setup.sh`: initialize the dependency and install pinned packages.
- `overcooked-v2/JaxMARL/`: pinned upstream Git submodule.
- `overcooked-v2/outputs/`: generated results, ignored by Git.
- `overcooked2-bridge/`: optional, user-supervised integration checks for the installed commercial Steam game; see its [setup and test guide](overcooked2-bridge/README.md).
- `AGENTS.md`: concise project context for coding agents.

## Training options

Users can choose between two training platforms:

1. **Local NVIDIA GPU:** train on your own compatible NVIDIA GPU using Linux or Windows through WSL2. The existing setup scripts and launchers implement this path; follow the setup and checks below.
2. **Modal cloud GPU:** use Modal when you do not have a compatible local NVIDIA GPU or prefer cloud training. This is the planned alternative; the Modal launcher and setup instructions are not implemented yet.

Both paths should use the same pinned JaxMARL version, baseline configuration, and evaluation protocol so results can be compared.

## Local NVIDIA GPU setup and checks

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

The paper-aligned Coordination Ring baseline is prepared for a user-led run:

```powershell
.\overcooked-v2\run.ps1 train -DryRun
.\overcooked-v2\run.ps1 train
```

The first command inspects settings. The second starts the roughly 10-million-step run and saves its final policy; see the setup guide before launching it.

## JaxMARL dependency

The submodule keeps upstream source/history separate from our scripts while recording the exact version used. After pulling a change to its pin, run:

```bash
git submodule update --init
```

Our simulator and PPO algorithm currently use upstream code unchanged. Dependency updates or simulator modifications should be deliberate project changes.
