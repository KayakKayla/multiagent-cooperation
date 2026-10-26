#!/bin/bash
set -euo pipefail
bridge_root="$(cd "$(dirname "$0")" && pwd)"
sdk="$bridge_root/.deps/dotnet"
test_output="$bridge_root/.deps/test"
mkdir -p "$test_output"
python3 - "$bridge_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
reference = next((root / '.deps/dotnet/packs/Microsoft.NETCore.App.Ref').glob('*/ref/net8.0'))
out = root / '.deps/test'
(out / 'references.rsp').write_text('\n'.join('"-r:' + str(p) + '"' for p in reference.glob('*.dll')))
(out / 'CoreTests.runtimeconfig.json').write_text(json.dumps({'runtimeOptions': {
    'tfm': 'net8.0', 'framework': {'name': 'Microsoft.NETCore.App', 'version': '8.0.0'}}}))
PY
"$sdk/dotnet" "$sdk/sdk/8.0.426/Roslyn/bincore/csc.dll" -nologo -target:exe \
    "-out:$test_output/CoreTests.dll" "@$test_output/references.rsp" \
    "$bridge_root/plugin/Protocol.cs" "$bridge_root/plugin/Transport.cs" \
    "$bridge_root/plugin/JsonOutput.cs" "$bridge_root/plugin/KitchenSnapshot.cs" "$bridge_root/tests/CoreTests.cs"
"$sdk/dotnet" "$test_output/CoreTests.dll"
PYTHONPYCACHEPREFIX="$test_output/pycache" python3 -m unittest discover -s "$bridge_root/tests" -p 'test_*.py' -v
