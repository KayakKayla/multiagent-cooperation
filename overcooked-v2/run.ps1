param(
    [ValidateSet('smoke', 'train-check', 'train', 'evaluate')]
    [string]$Mode = 'smoke',

    [string]$Layout = '',
    [int]$Seed = 42,
    [switch]$DryRun,
    [switch]$Untrained,
    [string]$Run = '',
    [int]$Episodes = 5,
    [int]$EvalSeed = 1000
)

$ErrorActionPreference = 'Stop'
$runArguments = @('./run.sh', $Mode)

if ($Mode -ne 'evaluate' -and ($Untrained -or $Run -or $PSBoundParameters.ContainsKey('Episodes') -or $PSBoundParameters.ContainsKey('EvalSeed'))) {
    throw '-Untrained, -Run, -Episodes and -EvalSeed are for evaluate mode.'
}

if ($Mode -eq 'train') {
    if ($Layout) {
        throw 'Training uses configs/coord_ring.yaml. Omit -Layout for this baseline.'
    }
    $runArguments += @('--seed', "$Seed")
    if ($DryRun) {
        $runArguments += '--dry-run'
    }
} elseif ($Mode -eq 'evaluate') {
    if ($Layout -or $DryRun) {
        throw 'Evaluation uses the baseline or saved training settings. Omit -Layout and -DryRun.'
    }
    if ([bool]$Untrained -eq [bool]$Run) {
        throw 'Choose exactly one: -Untrained or -Run <training-directory>.'
    }
    if ($Untrained) {
        $runArguments += @('--untrained', '--seed', "$Seed")
    } else {
        if ($PSBoundParameters.ContainsKey('Seed')) {
            throw 'A saved policy uses its saved training seed. Omit -Seed.'
        }
        # Accept normal Windows paths, including spaces, from the caller's folder.
        $windowsRun = (Resolve-Path -LiteralPath $Run).Path
        $linuxRun = & wsl -d Ubuntu-24.04 --exec wslpath -a -u $windowsRun
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not convert the saved run path for WSL.'
        }
        $runArguments += @('--run', "$linuxRun")
    }
    $runArguments += @('--episodes', "$Episodes", '--eval-seed', "$EvalSeed")
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
