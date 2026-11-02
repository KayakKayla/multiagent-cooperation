#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

# Optional machine settings stay outside the shared launcher.
if [[ -f ../.local/overcooked.env ]]; then
    source ../.local/overcooked.env
fi

if [[ "$mode" == scenario ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

export MPLBACKEND=Agg
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

venv_path="${OVC_VENV:-$PWD/.venv}"
if [[ ! -x "$venv_path/bin/python" ]]; then
    echo 'Environment missing. See README.md for baseline or scenario CPU setup.' >&2
    exit 1
fi

case "$mode" in
    smoke)
        exec "$venv_path/bin/python" smoke.py "$@"
        ;;
    train-check)
        exec "$venv_path/bin/python" train_check.py "$@"
        ;;
    train)
        exec "$venv_path/bin/python" train.py "$@"
        ;;
    evaluate)
        exec "$venv_path/bin/python" evaluate.py "$@"
        ;;
    scripted)
        exec "$venv_path/bin/python" ../docs/theoretical-maximum/verify_theoretical_maximum.py --gif "$@"
        ;;
    scenario)
        exec "$venv_path/bin/python" scenario.py "$@"
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        exit 2
        ;;
esac
