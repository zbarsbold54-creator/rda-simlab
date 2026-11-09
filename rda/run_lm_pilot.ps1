param([switch]$Smoke)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot
$env:NANOCHAT_DATASET = 'fineweb-edu'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$outputRoot = if ($Smoke) { Join-Path $repoRoot 'lm_pilot_smoke' } else { Join-Path $repoRoot 'lm_pilot' }
$lockPath = Join-Path $outputRoot 'driver.lock'
$statePath = Join-Path $outputRoot 'state.json'
$driverLogPath = Join-Path $outputRoot 'driver.log'
$iterations = if ($Smoke) { 20 } else { 4578 }
$checkpointEvery = if ($Smoke) { 10 } else { 250 }
$armNames = if ($Smoke) { @('A') } else { @('A', 'Bprime', 'C') }
$milestones = if ($Smoke) { @() } else { @(1500, 3000, 4578) }
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

if (-not ('NanochatProcessLineQueue' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
public sealed class NanochatProcessLineQueue {
    public readonly ConcurrentQueue<string> Lines = new ConcurrentQueue<string>();
    public void Attach(Process process) {
        process.OutputDataReceived += (sender, args) => { if (args.Data != null) Lines.Enqueue("stdout\t" + args.Data); };
        process.ErrorDataReceived += (sender, args) => { if (args.Data != null) Lines.Enqueue("stderr\t" + args.Data); };
    }
}
'@
}

function Add-LogLines {
    param([string]$Path, [string[]]$Lines)
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            [System.IO.File]::AppendAllLines($Path, $Lines, $utf8NoBom)
            return
        } catch {
            if ($attempt -eq 5) { throw }
            Start-Sleep -Milliseconds (150 * $attempt)
        }
    }
}

function Write-DriverLog {
    param([string]$Message)
    Add-LogLines -Path $driverLogPath -Lines @("$(Get-Date -Format o) $Message")
}

function Write-State {
    $state.updated_utc = [DateTime]::UtcNow.ToString('o')
    $tempPath = "$statePath.tmp"
    $json = $state | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($tempPath, $json, $utf8NoBom)
    Move-Item -LiteralPath $tempPath -Destination $statePath -Force
}

function Get-CheckpointDirectory([string]$Arm) {
    return (Join-Path (Join-Path $outputRoot $Arm) 'base_checkpoints')
}

function Test-CompleteCheckpoint([string]$Directory, [int]$Step) {
    $suffix = '{0:D6}' -f $Step
    $required = @(
        "model_$suffix.pt",
        "optim_${suffix}_rank0.pt",
        "rng_${suffix}_rank0.pt",
        "meta_$suffix.json"
    )
    foreach ($name in $required) {
        $path = Join-Path $Directory $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -le 0) { return $false }
    }
    try {
        $null = Get-Content -LiteralPath (Join-Path $Directory "meta_$suffix.json") -Raw | ConvertFrom-Json
    } catch { return $false }
    return $true
}

function Get-CompleteSteps([string]$Directory) {
    if (-not (Test-Path -LiteralPath $Directory)) { return @() }
    $steps = @()
    Get-ChildItem -LiteralPath $Directory -Filter 'model_*.pt' -File | ForEach-Object {
        if ($_.Name -match '^model_(\d+)\.pt$') {
            $step = [int]$Matches[1]
            if (Test-CompleteCheckpoint -Directory $Directory -Step $step) { $steps += $step }
        }
    }
    return @($steps | Sort-Object -Unique)
}

function Get-LatestCompleteStep([string]$Directory) {
    $steps = @(Get-CompleteSteps -Directory $Directory)
    if ($steps.Count -eq 0) { return 0 }
    return [int]$steps[-1]
}

function Prune-Checkpoints([string]$Directory) {
    $steps = @(Get-CompleteSteps -Directory $Directory)
    if ($steps.Count -le 2) { return }
    $keep = @($steps | Select-Object -Last 2) + @($milestones)
    foreach ($step in $steps) {
        if ($keep -contains $step) { continue }
        $suffix = '{0:D6}' -f $step
        foreach ($name in @("model_$suffix.pt", "meta_$suffix.json", "optim_${suffix}_rank0.pt", "rng_${suffix}_rank0.pt")) {
            $path = Join-Path $Directory $name
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
        }
    }
}

function New-ArmState {
    return [pscustomobject]@{
        status = 'pending'
        last_complete_checkpoint_step = 0
        retry_count = 0
        training_pid = 0
        notes = @()
        last_error = ''
    }
}

function Get-ArmArgs([string]$Arm) {
    $checkpointTag = Join-Path (Join-Path $outputRoot $Arm) 'base_checkpoints'
    $args = @(
        '-u', '-m', 'scripts.base_train',
        '--model-tag', $checkpointTag,
        '--seed=42', '--depth=8', '--max-seq-len=1024',
        '--device-batch-size=2', '--total-batch-size=32768',
        "--num-iterations=$iterations", "--save-every=$checkpointEvery",
        "--eval-every=$checkpointEvery", '--eval-tokens=524288',
        '--core-metric-every=-1', '--sample-every=-1'
    )
    if ($Arm -eq 'Bprime') { $args += '--rda' }
    if ($Arm -eq 'C') { $args += @('--rda', '--rda-const-state') }
    return $args
}

function ConvertTo-ProcessArgument([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' }
    return $Value
}

function Start-TrainingAttempt([string]$Arm, [int]$ResumeStep) {
    $armDir = Join-Path $outputRoot $Arm
    New-Item -ItemType Directory -Path $armDir -Force | Out-Null
    $logPath = Join-Path $armDir 'train.log'
    $valPath = Join-Path $armDir 'val_bpb.csv'
    if (-not (Test-Path -LiteralPath $valPath)) { [System.IO.File]::WriteAllText($valPath, "step,val_bpb`n", $utf8NoBom) }
    $seenSteps = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($valPath)) { if ($line -match '^(\d+),') { $seenSteps[[int]$Matches[1]] = $true } }

    $childArgs = @(Get-ArmArgs -Arm $Arm)
    if ($ResumeStep -gt 0) { $childArgs += "--resume-from-step=$ResumeStep" }
    $argumentString = ($childArgs | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join ' '
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $python
    $psi.Arguments = $argumentString
    $psi.WorkingDirectory = $repoRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    $queue = [NanochatProcessLineQueue]::new()
    $queue.Attach($process)
    if (-not $process.Start()) { throw 'Could not start training child.' }
    $state.arms.$Arm.training_pid = $process.Id
    Write-State
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    Write-DriverLog "TRAIN_START arm=$Arm pid=$($process.Id) resume_step=$ResumeStep args=$argumentString"
    $lastSeenCheckpoint = $ResumeStep
    $lastMsPerStep = 0.0
    $writer = [System.IO.StreamWriter]::new($logPath, $true, $utf8NoBom)
    $writer.AutoFlush = $true
    try {
        while (-not $process.HasExited -or -not $queue.Lines.IsEmpty) {
            $item = $null
            while ($queue.Lines.TryDequeue([ref]$item)) {
                $separator = ([string]$item).IndexOf([char]9)
                $stream = ([string]$item).Substring(0, $separator)
                $line = ([string]$item).Substring($separator + 1)
                $writer.WriteLine("[$stream] $line")
                if ($stream -eq 'stdout' -and $line -match '^Step\s+(\d+)\s+\|\s+Validation bpb:\s*([0-9]+(?:\.[0-9]+)?)') {
                    $valStep = [int]$Matches[1]
                    $bpb = $Matches[2]
                    if (-not $seenSteps.ContainsKey($valStep)) {
                        Add-LogLines -Path $valPath -Lines @("$valStep,$bpb")
                        $seenSteps[$valStep] = $true
                    }
                }
                if ($stream -eq 'stdout' -and $line -match 'dt:\s*([0-9]+(?:\.[0-9]+)?)ms') { $lastMsPerStep = [double]$Matches[1] }
            }
            $latest = Get-LatestCompleteStep -Directory (Get-CheckpointDirectory -Arm $Arm)
            if ($latest -gt $lastSeenCheckpoint) {
                $lastSeenCheckpoint = $latest
                $armState = $state.arms.$Arm
                $armState.last_complete_checkpoint_step = $latest
                $armState.notes = @($armState.notes | Where-Object { $_ -notmatch '^data position approximate after resume$' })
                Write-State
                Prune-Checkpoints -Directory (Get-CheckpointDirectory -Arm $Arm)
                Write-DriverLog "CHECKPOINT_COMPLETE arm=$Arm step=$latest"
            }
            if (-not $process.HasExited) { Start-Sleep -Milliseconds 500 }
        }
        $exitCode = $process.ExitCode
        $state.arms.$Arm.training_pid = 0
        Write-State
    } finally {
        $writer.Dispose()
        $process.Dispose()
    }
    return [pscustomobject]@{ ExitCode=$exitCode; LastCheckpoint=$lastSeenCheckpoint; MsPerStep=$lastMsPerStep; LogPath=$logPath }
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
if (Test-Path -LiteralPath $lockPath) {
    $lockedPid = 0
    [void][int]::TryParse((Get-Content -LiteralPath $lockPath -Raw).Trim(), [ref]$lockedPid)
    $lockedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$lockedPid" -ErrorAction SilentlyContinue
    if ($lockedProcess -and $lockedProcess.Name -match '^powershell' -and $lockedProcess.CommandLine -match 'run_lm_pilot\.ps1') {
        Write-Output "Driver already running as PID $lockedPid; exiting."
        exit 0
    }
    Write-DriverLog "STALE_LOCK takeover old_pid=$lockedPid"
}
[System.IO.File]::WriteAllText($lockPath, "$PID`n", $utf8NoBom)

try {
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } else {
        $state = [pscustomobject]@{ updated_utc=[DateTime]::UtcNow.ToString('o'); arms=[pscustomobject]@{} }
    }
    foreach ($arm in $armNames) {
        if (-not $state.arms.PSObject.Properties[$arm]) { $state.arms | Add-Member -NotePropertyName $arm -NotePropertyValue (New-ArmState) }
        if (-not $state.arms.$arm.PSObject.Properties['training_pid']) { $state.arms.$arm | Add-Member -NotePropertyName training_pid -NotePropertyValue 0 }
    }
    Write-State

    $msByArm = @{}
    foreach ($arm in $armNames) {
        $armState = $state.arms.$arm
        if ($armState.status -eq 'done' -or $armState.status -eq 'failed') {
            Write-DriverLog "SKIP arm=$arm status=$($armState.status)"
            continue
        }
        $checkpointDir = Get-CheckpointDirectory -Arm $arm
        $latest = Get-LatestCompleteStep -Directory $checkpointDir
        $orphanPid = 0
        if ($armState.training_pid) { $orphanPid = [int]$armState.training_pid }
        if ($orphanPid -gt 0) {
            $orphan = Get-CimInstance Win32_Process -Filter "ProcessId=$orphanPid" -ErrorAction SilentlyContinue
            if ($orphan -and $orphan.CommandLine -match 'scripts\.base_train') {
                Write-DriverLog "ATTACH_EXISTING_TRAINING arm=$arm pid=$orphanPid"
                while (Get-Process -Id $orphanPid -ErrorAction SilentlyContinue) {
                    $latest = Get-LatestCompleteStep -Directory $checkpointDir
                    if ($latest -gt [int]$armState.last_complete_checkpoint_step) {
                        $armState.last_complete_checkpoint_step = $latest
                        Write-State
                        Prune-Checkpoints -Directory $checkpointDir
                    }
                    Start-Sleep -Seconds 2
                }
                $armState.training_pid = 0
                $latest = Get-LatestCompleteStep -Directory $checkpointDir
                Write-State
            } else { $armState.training_pid = 0 }
        }
        $savedCheckpointStep = [int]$armState.last_complete_checkpoint_step
        $armState.last_complete_checkpoint_step = $latest
        if ($latest -gt $savedCheckpointStep) { $armState.retry_count = 0 }
        $armState.status = 'running'
        if ($latest -gt 0 -and -not ($armState.notes -contains 'data position approximate after resume')) {
            $armState.notes = @($armState.notes) + @('data position approximate after resume')
        }
        Write-State
        $failuresWithoutProgress = [int]$armState.retry_count
        $progressBeforeFailure = $latest
        while ($true) {
            $result = Start-TrainingAttempt -Arm $arm -ResumeStep $latest
            $latest = Get-LatestCompleteStep -Directory $checkpointDir
            $armState.last_complete_checkpoint_step = $latest
            if ($result.MsPerStep -gt 0) { $msByArm[$arm] = $result.MsPerStep }
            if ($result.ExitCode -eq 0 -and $latest -ge $iterations) {
                $armState.status = 'done'
                $armState.last_error = ''
                if (-not ($armState.notes -contains 'sampling skipped: no existing checkpoint-generation helper supports RDA checkpoints')) {
                    $armState.notes = @($armState.notes) + @('sampling skipped: no existing checkpoint-generation helper supports RDA checkpoints')
                }
                Write-State
                Write-DriverLog "TRAIN_DONE arm=$arm step=$latest"
                Write-DriverLog "SAMPLES_SKIPPED arm=$arm reason=no_existing_RDA_checkpoint_generation_helper"
                Prune-Checkpoints -Directory $checkpointDir
                break
            }
            $lastError = 'Training process exited with code {0}; latest complete checkpoint step {1}.' -f $result.ExitCode, $latest
            try { $lastError = (Get-Content -LiteralPath $result.LogPath -Tail 4 | Out-String).Trim() } catch { }
            $armState.last_error = $lastError
            $armState.retry_count = [int]$armState.retry_count + 1
            if ($latest -gt $progressBeforeFailure) {
                $failuresWithoutProgress = 0
                $armState.retry_count = 0
                $progressBeforeFailure = $latest
            } else { $failuresWithoutProgress++ }
            if ($failuresWithoutProgress -ge 3) {
                $armState.status = 'failed'
                Write-State
                Write-DriverLog "TRAIN_FAILED arm=$arm reason=3 failures without checkpoint"
                break
            }
            $armState.status = 'pending'
            Write-State
            Write-DriverLog "TRAIN_RETRY arm=$arm next_attempt=$($armState.retry_count + 1) resume_step=$latest"
            Start-Sleep -Seconds 60
            $armState.status = 'running'
            Write-State
        }
    }

    $allTerminal = $true
    foreach ($arm in $armNames) { if ($state.arms.$arm.status -notin @('done', 'failed')) { $allTerminal = $false } }
    if ($allTerminal -and -not $Smoke) {
        [System.IO.File]::WriteAllText((Join-Path $outputRoot 'DONE'), "$(Get-Date -Format o)`n", $utf8NoBom)
    }
    Write-DriverLog "DRIVER_FINISHED smoke=$Smoke all_terminal=$allTerminal"
    foreach ($arm in $msByArm.Keys) { Write-Output "ARM_MS_PER_STEP $arm $($msByArm[$arm])" }
} finally {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
