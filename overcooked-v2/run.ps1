param(
    [ValidateSet('smoke', 'train-check')]
    [string]$Mode = 'smoke',

    [string]$Layout = 'grounded_coord_simple'
)

$ErrorActionPreference = 'Stop'

# Resolve the Linux working directory from this script, not the caller's folder.
& wsl -d Ubuntu-24.04 --cd $PSScriptRoot --exec bash ./run.sh $Mode --layout $Layout

exit $LASTEXITCODE
