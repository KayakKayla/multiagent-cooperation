#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Fetch only our pinned JaxMARL dependency; its optional nested environments
# are not needed for OvercookedV2.
project_root="$(cd .. && pwd)"
git -C "$project_root" submodule update --init -- overcooked-v2/JaxMARL

revision="$(tr -d '\r\n' < UPSTREAM_REVISION)"
if [[ "$(git -C JaxMARL rev-parse HEAD)" != "$revision" ]]; then
    echo 'JaxMARL revision differs from UPSTREAM_REVISION; inspect local changes before switching.' >&2
    exit 1
fi

# Keep installed packages in Linux storage, separate from the Windows checkout.
venv_path="${OVC_VENV:-$HOME/.venvs/se390-overcooked-v2}"
if [[ ! -f "$venv_path/pyvenv.cfg" ]]; then
    python3 -m venv --without-pip "$venv_path"
fi
if ! "$venv_path/bin/python" -m pip --version >/dev/null 2>&1; then
    python3 -m pip --python "$venv_path/bin/python" install pip
fi

requirements=requirements.txt
if [[ -f requirements-gpu.lock.txt ]]; then
    requirements=requirements-gpu.lock.txt
fi

"$venv_path/bin/python" -m pip install -r "$requirements"
"$venv_path/bin/python" -m pip check
