#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Optional machine settings; command-line platform choices take precedence.
if [[ -f ../.local/overcooked.env ]]; then
    source ../.local/overcooked.env
fi
if [[ $# -gt 1 ]]; then
    echo 'Usage: bash setup.sh [--cpu|--gpu]' >&2
    exit 2
fi
case "${1:-}" in
    "") requirements="${OVC_REQUIREMENTS:-requirements.txt}" ;;
    --cpu) requirements=requirements.txt ;;
    --gpu) requirements=requirements-gpu.lock.txt ;;
    --help|-h) echo 'Usage: bash setup.sh [--cpu|--gpu]'; exit 0 ;;
    *) echo 'Usage: bash setup.sh [--cpu|--gpu]' >&2; exit 2 ;;
esac

# Fetch only our pinned JaxMARL dependency; its optional nested environments
# are not needed for OvercookedV2.
project_root="$(cd .. && pwd)"
git -C "$project_root" submodule update --init -- overcooked-v2/JaxMARL

revision="$(tr -d '\r\n' < UPSTREAM_REVISION)"
if [[ "$(git -C JaxMARL rev-parse HEAD)" != "$revision" ]]; then
    echo 'JaxMARL revision differs from UPSTREAM_REVISION; inspect local changes before switching.' >&2
    exit 1
fi

# Keep packages isolated; a machine profile can choose another location.
venv_path="${OVC_VENV:-$PWD/.venv}"
if [[ ! -f "$venv_path/pyvenv.cfg" ]]; then
    python3 -m venv "$venv_path"
fi
if ! "$venv_path/bin/python" -m pip --version >/dev/null 2>&1; then
    "$venv_path/bin/python" -m ensurepip --upgrade
fi

"$venv_path/bin/python" -m pip install -r "$requirements"
"$venv_path/bin/python" -m pip check
