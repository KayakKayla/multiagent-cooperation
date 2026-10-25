#!/bin/bash
set -euo pipefail
bridge_root="$(cd "$(dirname "$0")" && pwd)"
game_dir="${OC2_GAME_DIR:-$HOME/Library/Application Support/Steam/steamapps/common/Overcooked! 2}"
managed="$game_dir/Overcooked2.app/Contents/Resources/Data/Managed"
sdk="$bridge_root/.deps/dotnet"
export DOTNET_CLI_HOME="$bridge_root/.deps/dotnet-home"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
if [[ ! -x "$sdk/dotnet" ]]; then
    echo 'Local .NET SDK is missing; run setup.py first.' >&2
    exit 1
fi
mkdir -p "$bridge_root/bin"
# Compile against the game's actual old Mono API, without bundling its libraries.
refs=()
for assembly in "$managed"/*.dll; do
    refs+=("-reference:$assembly")
done
refs+=("-reference:$bridge_root/.deps/bepinex/BepInEx/core/BepInEx.dll")
"$sdk/dotnet" "$sdk/sdk/8.0.426/Roslyn/bincore/csc.dll" \
    -nologo -target:library -platform:anycpu -nostdlib+ -langversion:7.3 \
    "-out:$bridge_root/bin/Sims.Overcooked2Bridge.dll" \
    "${refs[@]}" "$bridge_root"/plugin/*.cs
