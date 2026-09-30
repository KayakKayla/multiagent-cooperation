param(
    [ValidateSet('smoke', 'train-check', 'train')]
    [string]$Mode = 'smoke',

    [string]$Layout = '',
    [int]$Seed = 42,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$runArguments = @('./run.sh', $Mode)

if ($Mode -eq 'train') {
    if ($Layout) {
        throw 'Training uses configs/coord_ring.yaml. Omit -Layout for this baseline.'
    }
    $runArguments += @('--seed', "$Seed")
    if ($DryRun) {
        $runArguments += '--dry-run'
    }
} else {
    if ($DryRun) {
        throw '-DryRun is available with train mode.'
    }
    if ($Layout) {
        $runArguments += @('--layout', $Layout)
    }
}

# Resolve the Linux working directory from this script, not the caller's folder.
& wsl -d Ubuntu-24.04 --cd $PSScriptRoot --exec bash @runArguments
exit $LASTEXITCODE
