$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repoRoot = 'C:\AI\nanochat'
$runTag = 'recall-A-nomask-s43-w20'
$runDir = Join-Path $repoRoot $runTag
$lockPath = Join-Path $repoRoot 'run_manip_check.pid'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$textPath = Join-Path $runDir 'driver.log'
$evalResultPath = Join-Path $runDir 'eval_result.txt'
$checkpointBase = if ($env:NANOCHAT_BASE_DIR) {
    $env:NANOCHAT_BASE_DIR
} else {
    Join-Path $env:USERPROFILE '.cache\nanochat'
}
$checkpointDir = $runDir
$maxResumes = 3
$DryRun = $false

Set-Location -LiteralPath $repoRoot
$env:NANOCHAT_DATASET = 'fineweb-edu'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python"
}
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

# Hold the PID file open for the full run. FileShare.Read lets a contender
# inspect the PID while preventing it from acquiring a second write lock.
$lockStream = $null
try {
    try {
        $lockStream = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::Read
        )
    } catch [System.IO.IOException] {
        $oldPid = 0
        try { $oldPid = [int](Get-Content -LiteralPath $lockPath -ErrorAction Stop | Select-Object -First 1) } catch {}
        if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            throw "Another manipulation check is live (PID $oldPid); refusing a second instance."
        }
        throw "Could not acquire the manipulation-check PID lock: $($_.Exception.Message)"
    }

    $lockStream.Position = 0
    $reader = [System.IO.StreamReader]::new(
        $lockStream, [System.Text.Encoding]::UTF8, $true, 1024, $true
    )
    $previousPidText = $reader.ReadToEnd().Trim()
    $reader.Dispose()
    $previousPid = 0
    if ([int]::TryParse($previousPidText, [ref]$previousPid) -and
        $previousPid -gt 0 -and $previousPid -ne $PID -and
        (Get-Process -Id $previousPid -ErrorAction SilentlyContinue)) {
        throw "PID file names live process $previousPid; refusing to start."
    }
    $lockStream.SetLength(0)
    $lockWriter = [System.IO.StreamWriter]::new(
        $lockStream, [System.Text.UTF8Encoding]::new($false), 1024, $true
    )
    $lockWriter.WriteLine([string]$PID)
    $lockWriter.Flush()
    $lockWriter.Dispose()

    function Add-LogLines([string[]]$lines) {
        if ($DryRun -or $lines.Count -eq 0) { return }
        $stream = $null
        for ($openAttempt = 1; $openAttempt -le 5; $openAttempt++) {
            try {
                $stream = [System.IO.File]::Open(
                    $textPath,
                    [System.IO.FileMode]::Append,
                    [System.IO.FileAccess]::Write,
                    [System.IO.FileShare]::ReadWrite
                )
                break
            } catch [System.IO.IOException] {
                if ($openAttempt -eq 5) { throw }
                Start-Sleep -Milliseconds (100 * $openAttempt)
            }
        }
        $writer = $null
        try {
            $writer = [System.IO.StreamWriter]::new(
                $stream, [System.Text.UTF8Encoding]::new($false)
            )
            foreach ($line in $lines) { $writer.WriteLine([string]$line) }
            $writer.Flush()
        } finally {
            if ($null -ne $writer) { $writer.Dispose() }
            elseif ($null -ne $stream) { $stream.Dispose() }
        }
    }

    function Write-LogLine([string]$line) { Add-LogLines @($line) }

    function Get-LatestCompleteCheckpoint {
        if (-not (Test-Path -LiteralPath $checkpointDir -PathType Container)) { return $null }
        $candidates = Get-ChildItem -LiteralPath $checkpointDir -Filter 'model_*.pt' -File |
            ForEach-Object {
                if ($_.Name -match '^model_(\d+)\.pt$') { [int]$Matches[1] }
            } | Sort-Object -Descending
        foreach ($step in $candidates) {
            $suffix = '{0:D6}' -f $step
            $required = @(
                "model_$suffix.pt", "meta_$suffix.json",
                "optim_${suffix}_rank0.pt", "rng_${suffix}_rank0.pt"
            )
            if (@($required | Where-Object {
                -not (Test-Path -LiteralPath (Join-Path $checkpointDir $_) -PathType Leaf)
            }).Count -ne 0) { continue }
            try {
                $meta = Get-Content -LiteralPath (Join-Path $checkpointDir "meta_$suffix.json") -Raw -Encoding UTF8 |
                    ConvertFrom-Json
                if ([int]$meta.step -eq $step) { return $step }
            } catch {}
        }
        return $null
    }

    $baseArgs = @(
        '-m', 'scripts.base_train',
        '--model-tag', $runDir,
        '--seed', '43',
        '--depth=8', '--aspect-ratio=96', '--head-dim=128',
        '--max-seq-len=1024', '--window-pattern=L',
        '--device-batch-size=2', '--total-batch-size=32768',
        '--num-iterations=1200', '--eval-every=300',
        '--eval-tokens=524288', '--core-metric-every=-1',
        '--sample-every=-1', '--synthetic-recall-frac=0.5',
        '--recall-answer-weight=20'
    )
    Write-LogLine "RUN_START timestamp=$(Get-Date -Format o) pid=$PID run_tag=$runTag"
    Write-LogLine "ENV NANOCHAT_DATASET=$env:NANOCHAT_DATASET PYTHONIOENCODING=$env:PYTHONIOENCODING PYTHONUTF8=$env:PYTHONUTF8 NANOCHAT_BASE_DIR=$env:NANOCHAT_BASE_DIR"
    Write-LogLine "TRAIN_TEMPLATE $python $($baseArgs -join ' ')"
    Write-LogLine "CHECKPOINT_DIR=$checkpointDir checkpoint_base=$checkpointBase"

    $resumeCount = 0
    $resumeStep = Get-LatestCompleteCheckpoint
    $trainExit = -1
    if ($null -ne $resumeStep -and $resumeStep -ge 1200) {
        Write-LogLine "TRAIN_ALREADY_COMPLETE step=$resumeStep"
        $trainExit = 0
    }
    for ($attempt = 1; $trainExit -ne 0 -and $attempt -le ($maxResumes + 1); $attempt++) {
        $childArgs = @($baseArgs)
        if ($null -ne $resumeStep) { $childArgs += "--resume-from-step=$resumeStep" }
        $stdoutPath = Join-Path $runDir "train_attempt_$attempt.stdout.log"
        $stderrPath = Join-Path $runDir "train_attempt_$attempt.stderr.log"
        Write-LogLine "TRAIN_START attempt=$attempt resume_step=$(if($null -eq $resumeStep){'fresh'}else{$resumeStep}) timestamp=$(Get-Date -Format o)"
        Write-LogLine "TRAIN_COMMAND $python $($childArgs -join ' ')"
        $child = Start-Process -FilePath $python -ArgumentList ($childArgs -join ' ') `
            -WorkingDirectory $repoRoot -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        Write-LogLine "TRAIN_CHILD_PID attempt=$attempt pid=$($child.Id)"
        while (-not $child.HasExited) { Start-Sleep -Seconds 1; $child.Refresh() }
        $trainExit = $child.ExitCode
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { @(Get-Content -LiteralPath $stdoutPath -Encoding UTF8) } else { @() }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { @(Get-Content -LiteralPath $stderrPath -Encoding UTF8) } else { @() }
        Add-LogLines $stdout
        Add-LogLines $stderr
        Write-LogLine "TRAIN_END attempt=$attempt exit_code=$trainExit timestamp=$(Get-Date -Format o)"
        if ($trainExit -eq 0) { break }
        if ($resumeCount -ge $maxResumes) { throw "Training child failed after $resumeCount resume attempts (exit=$trainExit)." }
        $resumeStep = Get-LatestCompleteCheckpoint
        if ($null -eq $resumeStep) { throw "Training child failed and no complete checkpoint was found (exit=$trainExit)." }
        if ($resumeStep -ge 1200) {
            Write-LogLine "TRAIN_CHECKPOINT_COMPLETE_AFTER_CHILD_EXIT step=$resumeStep"
            $trainExit = 0
            break
        }
        $resumeCount++
        Write-LogLine "TRAIN_RETRY next_attempt=$($attempt + 1) resume_step=$resumeStep resume_count=$resumeCount"
    }
    if ($trainExit -ne 0) { throw "Training failed with exit code $trainExit." }
    $finalStep = Get-LatestCompleteCheckpoint
    if ($null -eq $finalStep -or $finalStep -ne 1200) {
        throw "Training exited successfully but final complete checkpoint is step $finalStep, expected 1200."
    }
    Write-LogLine "TRAIN_COMPLETE step=$finalStep resume_count=$resumeCount"

    $evalWrapper = Join-Path $runDir 'eval_nomask_wrapper.py'
    @'
import os, sys
sys.path.insert(0, r"C:\AI\nanochat")
from diagnostics import recall_models
_real_load_model = recall_models.load_model
def _load_model(checkpoint_dir, step, device, seed, segment_len=128, decay_override=None):
    return _real_load_model(checkpoint_dir, step, device, seed, segment_len=0,
                            decay_override=decay_override)
import diagnostics.eval_recall as _eval
_eval.load_model = _load_model
_eval.main()
'@ | Set-Content -LiteralPath $evalWrapper -Encoding UTF8

    $evalArgs = @(
        $evalWrapper,
        '--checkpoint-dir', $checkpointDir,
        '--step', '1200', '--device-batch-size', '2', '--n-rows', '8192',
        '--seed', '43', '--eval-seed-base', '3000000000',
        '--allow-cross-above-0.09'
    )
    $evalOut = Join-Path $runDir 'eval.stdout.log'
    $evalErr = Join-Path $runDir 'eval.stderr.log'
    Write-LogLine "EVAL_START timestamp=$(Get-Date -Format o) command=$python $($evalArgs -join ' ') segment_len=0"
    $evalChild = Start-Process -FilePath $python -ArgumentList ($evalArgs -join ' ') `
        -WorkingDirectory $repoRoot -PassThru -Wait `
        -RedirectStandardOutput $evalOut -RedirectStandardError $evalErr

    # First action after the evaluation child exits: preserve its complete output.
    $evalText = [System.IO.File]::ReadAllText($evalOut, [System.Text.Encoding]::UTF8) +
        "`r`n===== STDERR =====`r`n" +
        [System.IO.File]::ReadAllText($evalErr, [System.Text.Encoding]::UTF8)
    [System.IO.File]::WriteAllText(
        $evalResultPath, $evalText, [System.Text.UTF8Encoding]::new($false)
    )
    $evalExit = $evalChild.ExitCode
    Add-LogLines @(($evalText -split "`r?`n"))
    Write-LogLine "EVAL_END exit_code=$evalExit output=$evalResultPath timestamp=$(Get-Date -Format o)"
    if ($evalExit -ne 0) { throw "Evaluation failed with exit code $evalExit; output saved to $evalResultPath." }
    Write-LogLine "RUN_COMPLETE timestamp=$(Get-Date -Format o)"
} finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
}
