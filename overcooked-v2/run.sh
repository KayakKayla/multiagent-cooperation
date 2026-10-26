#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

# Runtime defaults for the verified local WSL/GPU setup.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "$mode" == scenario ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
else
    # Keep GPU first; upstream logging callbacks also require a CPU device.
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
    # This WSL/GPU stack stalls on repeated batched steps with CUDA graphs enabled.
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
fi
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

venv_path="${OVC_VENV:-$HOME/.venvs/se390-overcooked-v2}"
if [[ "$mode" == scenario && -z "${OVC_VENV:-}" && -x .venv/bin/python ]]; then
    venv_path="$PWD/.venv"
fi
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
