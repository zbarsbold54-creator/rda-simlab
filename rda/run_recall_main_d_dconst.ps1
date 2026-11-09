param(
    [switch]$DryRun,
    [string]$StateDirectory,
    [switch]$TrainingSmoke,
    [int]$SmokeSeed = 101,
    [int]$SmokeIterations = 3,
    [string]$SmokeTag = 'smoke-driverpath-s101'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$outputDir = if ($TrainingSmoke) {
    if (-not $StateDirectory) {
        throw "TrainingSmoke requires an isolated -StateDirectory."
    }
    [System.IO.Path]::GetFullPath($StateDirectory)
}
elseif ($StateDirectory) {
    [System.IO.Path]::GetFullPath($StateDirectory)
}
else {
    Join-Path $repoRoot 'results\recall_benchmark\main'
}
$checkpointBase = if ($env:NANOCHAT_BASE_DIR) {
    $env:NANOCHAT_BASE_DIR
}
else {
    Join-Path $env:USERPROFILE '.cache\nanochat'
}
$jsonPath = Join-Path $outputDir 'main_d_dconst.json'
$textPath = Join-Path $outputDir 'main_d_dconst.txt'

function Add-LogLines([string[]]$lines) {
    if ($DryRun -or $lines.Count -eq 0) {
        return
    }

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
        }
        catch [System.IO.IOException] {
            if ($openAttempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds (100 * $openAttempt)
        }
    }

    $writer = $null
    try {
        $writer = [System.IO.StreamWriter]::new($stream, [System.Text.UTF8Encoding]::new($false))
        foreach ($line in $lines) {
            $writer.WriteLine([string]$line)
        }
        $writer.Flush()
    }
    finally {
        if ($null -ne $writer) {
            $writer.Dispose()
        }
        elseif ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

$checkpointRoot = if ($DryRun -and $StateDirectory) {
    Join-Path $outputDir 'base_checkpoints'
}
else {
    Join-Path $checkpointBase 'base_checkpoints'
}
$evalSeedBase = 3000000000
$nRows = 8192
$step = 1200
$deviceBatchSize = 2
$trainingSeeds = @(43, 101, 102, 103, 104)
$crossDelays = @('192', '320', '448', '576', '704', '832')

if ($evalSeedBase -le 104 -or $evalSeedBase -in @(1000000000, 2000000000) -or $nRows -ge $evalSeedBase) {
    throw "Eval seed base failed the seed-disjointness preflight."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python"
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$jsonExists = Test-Path -LiteralPath $jsonPath -PathType Leaf
$textExists = Test-Path -LiteralPath $textPath -PathType Leaf

function Save-Json {
    $temporaryPath = "$jsonPath.tmp"
    $json = $script:data | ConvertTo-Json -Depth 20
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($temporaryPath, $json, $encoding)
    if (Test-Path -LiteralPath $jsonPath -PathType Leaf) {
        [System.IO.File]::Replace($temporaryPath, $jsonPath, $null)
    }
    else {
        [System.IO.File]::Move($temporaryPath, $jsonPath)
    }
}

if ($jsonExists) {
    if (-not $DryRun -and -not $textExists) {
        throw "JSON exists but text log is missing; refusing to replace or desynchronize prior results."
    }
    try {
        $script:data = Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        $corruptPath = "$jsonPath.corrupt-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
        Copy-Item -LiteralPath $jsonPath -Destination $corruptPath
        throw "Existing JSON failed to parse; preserved it at $corruptPath. Original was not overwritten. $($_.Exception.Message)"
    }
    if ($script:data.schema_version -ne 1 -or $script:data.eval_seed_base -ne $evalSeedBase) {
        throw "Existing JSON has an unexpected schema or eval seed base."
    }
    if ($null -eq $script:data.entries) {
        throw "Existing JSON is missing its entries array."
    }
    if (-not $script:data.PSObject.Properties['completed']) {
        $script:data | Add-Member -NotePropertyName completed -NotePropertyValue $false
    }
    if (-not $script:data.PSObject.Properties['completed_at']) {
        $script:data | Add-Member -NotePropertyName completed_at -NotePropertyValue $null
    }
    if (-not $DryRun) {
        Add-LogLines @("`r`nRESUME timestamp=$(Get-Date -Format o)")
    }
}
else {
    if ($textExists -and -not $DryRun) {
        $headerLines = @(Get-Content -LiteralPath $textPath -TotalCount 2 -Encoding UTF8)
        if ($headerLines.Count -lt 2 -or $headerLines[0] -ne 'Recall main D / D_const grid' -or
            $headerLines[1] -ne 'seed 43 also informed calibration choices (answer weight 20, no-context stop lift); seeds 101-104 did not.') {
            throw "Text log exists without JSON and does not match the expected fresh header."
        }
    }
    $script:data = [pscustomobject]@{
        schema_version = 1
        eval_seed_base = $evalSeedBase
        n_rows = $nRows
        entries = @()
        completed = $false
        completed_at = $null
    }
    if (-not $textExists -and -not $DryRun) {
        $header = @(
            'Recall main D / D_const grid'
            'seed 43 also informed calibration choices (answer weight 20, no-context stop lift); seeds 101-104 did not.'
            "SETTINGS eval_seed_base=$evalSeedBase N=$nRows step=$step segment_len=128 recall_fraction=0.5 answer_weight=20 iterations=1200 device_batch_size=$deviceBatchSize scoring=16-way CROSS=192,320,448,576,704,832"
            'No D-vs-D_const or across-seed hypothesis tests are run here. The no-context control applies the spec pooled check plus Holm correction across 16 delay/class cells separately per model.'
            "DRIVER_COMMAND=powershell -NoProfile -ExecutionPolicy Bypass -File .\run_recall_main_d_dconst.ps1"
        )
        Set-Content -LiteralPath $textPath -Value $header -Encoding UTF8
    }
    if (-not $DryRun) {
        Save-Json
    }
    if ($textExists -and -not $DryRun) {
        Add-LogLines @("RESUME after_initialization timestamp=$(Get-Date -Format o)")
    }
}

function Write-LogLine([string]$line) {
    Write-Host $line
    Add-LogLines @($line)
}

function Add-Entry($entry) {
    $script:data.entries = @($script:data.entries) + @($entry)
    if (-not $DryRun) {
        Save-Json
    }
}

function Test-CompleteRunEntry($entry, [string]$runId) {
    if ($null -eq $entry -or $entry.type -ne 'run' -or $entry.run_id -ne $runId) {
        return $false
    }
    if ($entry.eval_seed_base -ne $evalSeedBase -or $entry.n_rows -ne $nRows) {
        return $false
    }
    if ($null -eq $entry.per_delay -or
        $null -eq $entry.cross_pooled -or
        $null -eq $entry.cross_pooled.salient_accuracy -or
        $null -eq $entry.cross_pooled.ordinary_accuracy -or
        $null -eq $entry.cross_pooled.gap -or
        $null -eq $entry.no_context -or
        $null -eq $entry.no_context.cells -or
        @($entry.no_context.cells).Count -ne 16 -or
        $null -eq $entry.no_context.holm_cells -or
        @($entry.no_context.holm_cells).Count -ne 16) {
        return $false
    }
    foreach ($delay in @('16', '48') + $crossDelays) {
        if ($null -eq $entry.per_delay.PSObject.Properties[$delay]) {
            return $false
        }
        $cell = $entry.per_delay.PSObject.Properties[$delay].Value
        if ($null -eq $cell.SALIENT -or $null -eq $cell.ORDINARY) {
            return $false
        }
    }
    return $true
}

function Get-RunEntry([string]$runId) {
    $matches = @($script:data.entries | Where-Object {
        $_.type -eq 'run' -and $_.run_id -eq $runId
    })
    for ($index = $matches.Count - 1; $index -ge 0; $index--) {
        if (Test-CompleteRunEntry $matches[$index] $runId) {
            return $matches[$index]
        }
    }
    return $null
}

function Get-RunPaths([string]$arm, [int]$seed) {
    $tagArm = if ($arm -eq 'D') { 'D' } else { 'Dconst' }
    $tag = "recall-$tagArm-s$seed-w20"
    $directory = Join-Path $checkpointRoot $tag
    return [pscustomobject]@{
        tag = $tag
        checkpoint_dir = $directory
        checkpoint = Join-Path $directory "model_$('{0:D6}' -f $step).pt"
        marker = Join-Path $checkpointRoot "$tag.train_done"
    }
}

function Read-TrainMarker([string]$arm, [int]$seed, $paths) {
    if (-not (Test-Path -LiteralPath $paths.marker -PathType Leaf) -or
        -not (Test-Path -LiteralPath $paths.checkpoint -PathType Leaf)) {
        return $null
    }
    try {
        $marker = Get-Content -LiteralPath $paths.marker -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
        $checkpointSize = (Get-Item -LiteralPath $paths.checkpoint).Length
        if ($marker.run_id -ne "$arm-$seed" -or
            $marker.process_exit_code -ne 0 -or
            $marker.checkpoint_bytes -le 0 -or
            $marker.checkpoint_bytes -ne $checkpointSize) {
            return $null
        }
        return $marker
    }
    catch {
        return $null
    }
}

function Test-TrainingComplete([string]$arm, [int]$seed, $paths) {
    return $null -ne (Read-TrainMarker $arm $seed $paths)
}

function Remove-IncompleteRun([string]$arm, [int]$seed, $paths) {
    if ($seed -eq 43) {
        throw "Refusing to modify seed-43 calibration checkpoint folder."
    }
    if ($paths.tag -notmatch '^recall-(D|Dconst)-s(101|102|103|104)-w20$') {
        throw "Refusing to delete an out-of-scope checkpoint path: $($paths.tag)"
    }
    $expected = [System.IO.Path]::GetFullPath((Join-Path $checkpointRoot $paths.tag))
    $root = [System.IO.Path]::GetFullPath($checkpointRoot).TrimEnd('\') + '\'
    if (-not $expected.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete a checkpoint folder outside $checkpointRoot."
    }
    if (Test-Path -LiteralPath $paths.marker -PathType Leaf) {
        Remove-Item -LiteralPath $paths.marker -Force
        Write-LogLine "DELETED_STALE_TRAIN_MARKER path=$($paths.marker) reason=checkpoint_missing_or_marker_invalid"
    }
    if (Test-Path -LiteralPath $expected -PathType Container) {
        Remove-Item -LiteralPath $expected -Recurse -Force
        Write-LogLine "DELETED_INCOMPLETE_CHECKPOINT_FOLDER path=$expected reason=missing_valid_train_done_marker"
    }
}

function Test-StableCheckpoint([string]$checkpointPath) {
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        return $null
    }
    $first = Get-Item -LiteralPath $checkpointPath
    Start-Sleep -Seconds 2
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        return $null
    }
    $second = Get-Item -LiteralPath $checkpointPath
    if ($first.Length -le 0 -or $first.Length -ne $second.Length) {
        return $null
    }
    return [long]$second.Length
}

function Write-TrainMarker([string]$arm, [int]$seed, $paths, [long]$checkpointBytes, $trainingInfo) {
    $markerData = [pscustomobject]@{
        run_id = "$arm-$seed"
        process_exit_code = 0
        checkpoint_path = $paths.checkpoint
        checkpoint_bytes = $checkpointBytes
        training_wall_seconds = [double]$trainingInfo.wall_seconds
        peak_vram_mib = $trainingInfo.peak_vram_mib
        attempts = [int]$trainingInfo.attempts
        completed_at = (Get-Date).ToString('o')
    }
    $temporaryMarker = "$($paths.marker).tmp"
    $markerJson = $markerData | ConvertTo-Json -Depth 5
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($temporaryMarker, $markerJson, $encoding)
    if (Test-Path -LiteralPath $paths.marker -PathType Leaf) {
        [System.IO.File]::Replace($temporaryMarker, $paths.marker, $null)
    }
    else {
        [System.IO.File]::Move($temporaryMarker, $paths.marker)
    }
    return $markerData
}

function Get-PairDecision([int]$seed) {
    $pairId = "seed-$seed"
    $existing = @($script:data.entries | Where-Object {
        $_.type -eq 'no_context_pair_decision' -and $_.pair_id -eq $pairId
    })
    if ($existing.Count -gt 1) {
        throw "Duplicate no-context pair decisions found for $pairId."
    }
    if ($existing.Count -eq 1) {
        return
    }

    $d = Get-RunEntry "D-$seed"
    $dconst = Get-RunEntry "D_const-$seed"
    if ($null -eq $d -or $null -eq $dconst) {
        return
    }
    $lift = $d.no_context.arm_eligible -and $dconst.no_context.arm_eligible
    $outcome = if ($lift) { 'STOP_LIFTED' } else { 'STOP_STANDS' }
    $entry = [pscustomobject]@{
        type = 'no_context_pair_decision'
        pair_id = $pairId
        seed = $seed
        d_arm_eligible = [bool]$d.no_context.arm_eligible
        dconst_arm_eligible = [bool]$dconst.no_context.arm_eligible
        outcome = $outcome
        rule = 'Lift only if neither arm has any Holm-significant cell and both pooled checks are WITHIN_NULL_CI.'
        completed_at = (Get-Date).ToString('o')
    }
    Add-Entry $entry
    Write-LogLine "NO_CONTEXT_PAIR seed=$seed d_eligible=$($entry.d_arm_eligible) dconst_eligible=$($entry.dconst_arm_eligible) outcome=$outcome"
}

function Invoke-PythonLogged(
    [string[]]$arguments,
    [string]$label,
    [switch]$TrainingChild,
    [string]$RunTag,
    [int]$Attempt = 0
) {
    Write-LogLine "SUBPROCESS_BEGIN label=$label command=$python $($arguments -join ' ')"
    $stdoutPath = Join-Path $env:TEMP "recall-main-$PID-$label.stdout.tmp"
    $stderrPath = Join-Path $env:TEMP "recall-main-$PID-$label.stderr.tmp"
    $quotedArguments = @($arguments | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    })
    $argumentString = $quotedArguments -join ' '
    $originalWandbMode = $env:WANDB_MODE
    $originalWandbConsole = $env:WANDB_CONSOLE
    if ($TrainingChild) {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        $env:PYTHONUNBUFFERED = '1'
        $env:WANDB_MODE = 'disabled'
        $env:WANDB_CONSOLE = 'off'
    }
    try {
        $process = Start-Process -FilePath $python -ArgumentList $argumentString `
            -WorkingDirectory $repoRoot -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    }
    catch {
        Write-LogLine "SUBPROCESS_END label=$label exit_code=none failure=no exit code (process killed or interrupted)"
        throw
    }
    finally {
        if ($TrainingChild) {
            $env:WANDB_MODE = $originalWandbMode
            $env:WANDB_CONSOLE = $originalWandbConsole
        }
    }
    $stdout = @()
    $stderr = @()
    if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
        $stdout = @(Get-Content -LiteralPath $stdoutPath -Encoding UTF8)
    }
    if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
        $stderr = @(Get-Content -LiteralPath $stderrPath -Encoding UTF8)
    }
    $output = @($stdout) + @($stderr)
    $exitCode = $process.ExitCode
    Add-LogLines $output
    if ($TrainingChild -and $exitCode -ne 0) {
        $failedDirectory = Join-Path $repoRoot 'results\recall_benchmark\main\failed_attempts'
        New-Item -ItemType Directory -Path $failedDirectory -Force | Out-Null
        $failureBase = "$RunTag-attempt-$Attempt"
        $failedStdout = Join-Path $failedDirectory "$failureBase.stdout.txt"
        $failedStderr = Join-Path $failedDirectory "$failureBase.stderr.txt"
        if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
            Copy-Item -LiteralPath $stdoutPath -Destination $failedStdout -Force
        }
        if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
            Copy-Item -LiteralPath $stderrPath -Destination $failedStderr -Force
        }
        Write-LogLine "FAILED_ATTEMPT_FILES exit_code=$exitCode stdout=$failedStdout stderr=$failedStderr"
        Write-Host "TRAINING STDERR TAIL label=$label exit_code=$exitCode"
        $stderr | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
    }
    Write-LogLine "SUBPROCESS_END label=$label exit_code=$exitCode"
    if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
        Remove-Item -LiteralPath $stdoutPath -Force
    }
    if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
        Remove-Item -LiteralPath $stderrPath -Force
    }
    return [pscustomobject]@{
        output = @($output | ForEach-Object { [string]$_ })
        stdout = @($stdout | ForEach-Object { [string]$_ })
        stderr = @($stderr | ForEach-Object { [string]$_ })
        exit_code = $exitCode
    }
}

function Get-TrainingArguments([string]$arm, [int]$seed, [string]$tag) {
    $common = @(
        '-m', 'scripts.base_train',
        "--model-tag", $tag,
        "--seed", "$seed",
        '--depth=8',
        '--aspect-ratio=96',
        '--head-dim=128',
        '--max-seq-len=1024',
        '--window-pattern=L',
        '--device-batch-size=2',
        '--total-batch-size=32768',
        '--num-iterations=1200',
        '--eval-every=300',
        '--eval-tokens=524288',
        '--core-metric-every=-1',
        '--sample-every=-1',
        '--segment-len=128',
        '--synthetic-recall-frac=0.5',
        '--recall-answer-weight=20',
        '--rda',
        '--rda-decays',
        '0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875'
    )
    if ($arm -eq 'D_const') {
        return @($common) + @('--rda-const-state')
    }
    return $common
}

function Get-ExistingTrainingPeak([string]$arm) {
    $suffix = if ($arm -eq 'D') { 'D' } else { 'Dconst' }
    $path = Join-Path $repoRoot "results\recall_benchmark\calib\train_recall-$suffix-s43-w20.txt"
    $match = Select-String -LiteralPath $path -Pattern '^Peak memory usage: ([0-9.]+)MiB$' | Select-Object -Last 1
    if ($null -eq $match) {
        return $null
    }
    return [double]$match.Matches[0].Groups[1].Value
}

function Get-TrainingPeak($output) {
    $peaks = @()
    foreach ($line in $output) {
        if ([string]$line -match 'Peak memory usage:\s*([0-9.]+)\s*MiB') {
            $peaks += [double]$Matches[1]
        }
    }
    if ($peaks.Count -eq 0) {
        return $null
    }
    return ($peaks | Measure-Object -Maximum).Maximum
}

function Invoke-Training([string]$arm, [int]$seed, [string]$tag, [string]$checkpointPath) {
    $arguments = Get-TrainingArguments $arm $seed $tag
    $command = "$python -m scripts.base_train " + ($arguments[2..($arguments.Count - 1)] -join ' ')
    Write-LogLine "TRAIN_COMMAND arm=$arm seed=$seed $command"
    $allOutput = @()
    $trainingWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $paths = Get-RunPaths $arm $seed
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        Remove-IncompleteRun $arm $seed $paths
        Write-LogLine "TRAIN_BEGIN arm=$arm seed=$seed attempt=$attempt"
        Write-LogLine "TRAIN_START_TIMESTAMP timestamp=$(Get-Date -Format o) arm=$arm seed=$seed attempt=$attempt"
        Write-LogLine "TRAIN_CHILD_ENV arm=$arm seed=$seed attempt=$attempt PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled WANDB_CONSOLE=off"
        $attemptStarted = [DateTimeOffset]::Now
        Write-LogLine ("ATTEMPT {0}/2 begin timestamp={1} arm={2} seed={3}" -f $attempt, $attemptStarted.ToString('o'), $arm, $seed)
        $result = $null
        try {
            $result = Invoke-PythonLogged $arguments "train-$arm-$seed-attempt-$attempt" `
                -TrainingChild -RunTag $tag -Attempt $attempt
        }
        finally {
            $attemptEnded = [DateTimeOffset]::Now
            $attemptElapsed = ($attemptEnded - $attemptStarted).TotalSeconds
            $attemptExit = if ($null -ne $result) {
                "exit_code=$($result.exit_code)"
            }
            else {
                'exit_code=none failure=no exit code (process killed or interrupted)'
            }
            Write-LogLine ("ATTEMPT {0}/2 end timestamp={1} elapsed_seconds={2:F3} {3} arm={4} seed={5}" -f $attempt, $attemptEnded.ToString('o'), $attemptElapsed, $attemptExit, $arm, $seed)
        }
        $allOutput += $result.output
        $hasNanLoss = $false
        foreach ($line in $result.output) {
            if ($line -match '(?i)\bloss(?:_value)?\s*[:=]\s*nan\b') {
                $hasNanLoss = $true
                break
            }
        }
        if ($hasNanLoss) {
            Write-LogLine "TRAIN_ATTEMPT_RESULT arm=$arm seed=$seed attempt=$attempt exit_code=$($result.exit_code) nan_loss=True"
            $trainingWatch.Stop()
            throw "Training loss is NaN for $arm seed $seed; stopping without retry."
        }
        $stableBytes = if ($result.exit_code -eq 0) {
            Test-StableCheckpoint $checkpointPath
        }
        else {
            $null
        }
        if ($result.exit_code -eq 0 -and $null -ne $stableBytes) {
            $trainingWatch.Stop()
            $trainingInfo = [pscustomobject]@{
                wall_seconds = $trainingWatch.Elapsed.TotalSeconds
                peak_vram_mib = Get-TrainingPeak $allOutput
                attempts = $attempt
            }
            $null = Write-TrainMarker $arm $seed $paths $stableBytes $trainingInfo
            Write-LogLine "TRAIN_ATTEMPT_RESULT arm=$arm seed=$seed attempt=$attempt exit_code=0 checkpoint_bytes=$stableBytes stable_two_reads=True train_done_marker=written"
            return $trainingInfo
        }
        $attemptStatus = if ($result.exit_code -eq 0) { 'checkpoint_missing_empty_or_unstable=True' } else { 'checkpoint_not_verified=True' }
        Write-LogLine "TRAIN_ATTEMPT_RESULT arm=$arm seed=$seed attempt=$attempt exit_code=$($result.exit_code) $attemptStatus"
        if ($attempt -lt 2) {
            $reason = if ($result.exit_code -eq 0) { 'checkpoint_missing_empty_or_unstable' } else { "process_exit_$($result.exit_code)" }
            Write-LogLine "TRAIN_RETRY arm=$arm seed=$seed reason=$reason"
        }
    }
    $trainingWatch.Stop()
    throw "Training failed twice for $arm seed $seed."
}

function Parse-EvalResult([string[]]$output, [string]$arm, [int]$seed, [double]$wallSeconds, $trainingPeakMiB, [int]$retries) {
    $resultJsonLine = $output | Where-Object { $_ -match '^RESULT_JSON=' } | Select-Object -Last 1
    if ($null -eq $resultJsonLine) {
        throw "Eval output omitted RESULT_JSON for $arm seed $seed."
    }
    $perDelay = ($resultJsonLine -replace '^RESULT_JSON=', '') | ConvertFrom-Json
    $salientSum = 0.0
    $ordinarySum = 0.0
    $pairedSum = 0.0
    foreach ($delay in $crossDelays) {
        $cell = $perDelay.PSObject.Properties[$delay].Value
        if ($null -eq $cell) {
            throw "Eval output omitted CROSS delay $delay for $arm seed $seed."
        }
        $salientSum += [double]$cell.SALIENT
        $ordinarySum += [double]$cell.ORDINARY
        $pairedSum += [double]$cell.paired_difference
    }
    $pooledSalient = $salientSum / $crossDelays.Count
    $pooledOrdinary = $ordinarySum / $crossDelays.Count
    $noContextCells = @()
    $holmCells = @()
    foreach ($line in $output) {
        if ($line -match '^NO_CONTEXT delay=(\d+) class=(SALIENT|ORDINARY) .* pvalue=([0-9.]+) flag=(\S+)$') {
            $noContextCells += [pscustomobject]@{
                delay = [int]$Matches[1]
                class = $Matches[2]
                pvalue = [double]$Matches[3]
                flag = $Matches[4]
            }
        }
        elseif ($line -match '^HOLM delay=(\d+) class=(SALIENT|ORDINARY) pvalue=([0-9.]+) holm_pvalue=([0-9.]+) significant=(YES|NO)$') {
            $holmCells += [pscustomobject]@{
                delay = [int]$Matches[1]
                class = $Matches[2]
                pvalue = [double]$Matches[3]
                holm_pvalue = [double]$Matches[4]
                significant = ($Matches[5] -eq 'YES')
            }
        }
    }
    $pooledLine = $output | Where-Object { $_ -match '^NO_CONTEXT_POOLED ' } | Select-Object -Last 1
    if ($null -eq $pooledLine -or $pooledLine -notmatch 'accuracy=([0-9.]+) .*permutation_null=([0-9.]+) .*flag=(\S+)$') {
        throw "Eval output omitted the pooled no-context result for $arm seed $seed."
    }
    $pooledNoContextAccuracy = [double]$Matches[1]
    $pooledNoContextNull = [double]$Matches[2]
    $pooledNoContextFlag = $Matches[3]
    if ($holmCells.Count -ne 16 -or $noContextCells.Count -ne 16) {
        throw "Expected 16 no-context and Holm cells; got $($noContextCells.Count), $($holmCells.Count) for $arm seed $seed."
    }
    $hasHolmSignificance = [bool](@($holmCells | Where-Object { $_.significant }).Count -gt 0)
    $armEligible = (-not $hasHolmSignificance) -and ($pooledNoContextFlag -eq 'WITHIN_NULL_CI')
    return [pscustomobject]@{
        type = 'run'
        run_id = "$arm-$seed"
        arm = $arm
        seed = $seed
        checkpoint_step = $step
        checkpoint_dir = "C:\Users\HiTech\.cache\nanochat\base_checkpoints\recall-$($arm -replace '_const','const')-s$seed-w20"
        eval_seed_base = $evalSeedBase
        n_rows = $nRows
        per_delay = $perDelay
        cross_pooled = [pscustomobject]@{
            salient_accuracy = $pooledSalient
            ordinary_accuracy = $pooledOrdinary
            gap = $pairedSum / $crossDelays.Count
        }
        no_context = [pscustomobject]@{
            pooled_accuracy = $pooledNoContextAccuracy
            pooled_permutation_null = $pooledNoContextNull
            pooled_flag = $pooledNoContextFlag
            holm_cells = $holmCells
            cells = $noContextCells
            any_holm_significant = $hasHolmSignificance
            arm_eligible = $armEligible
        }
        wall_seconds = $wallSeconds
        training_peak_vram_mib = $trainingPeakMiB
        training_retries = $retries
        completed_at = (Get-Date).ToString('o')
    }
}

function Invoke-Evaluation([string]$arm, [int]$seed, [string]$tag, [string]$checkpointPath, $trainingInfo) {
    $runId = "$arm-$seed"
    $existingRun = Get-RunEntry $runId
    if ($null -ne $existingRun) {
        Write-LogLine "SKIP_EVAL arm=$arm seed=$seed reason=complete_json_entry_exists"
        Get-PairDecision $seed
        return
    }
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "Required checkpoint is missing: $checkpointPath"
    }
    if ($seed -eq 43 -and $null -eq $trainingInfo) {
        $trainingInfo = [pscustomobject]@{
            wall_seconds = 0.0
            peak_vram_mib = Get-ExistingTrainingPeak $arm
            attempts = 0
        }
    }
    elseif ($null -eq $trainingInfo) {
        $paths = Get-RunPaths $arm $seed
        $marker = Read-TrainMarker $arm $seed $paths
        if ($null -eq $marker) {
            throw "Cannot evaluate $runId without a valid train_done marker."
        }
        $trainingInfo = [pscustomobject]@{
            wall_seconds = [double]$marker.training_wall_seconds
            peak_vram_mib = $marker.peak_vram_mib
            attempts = [int]$marker.attempts
        }
    }

    $evalArgs = @(
        '-m', 'diagnostics.eval_recall',
        '--checkpoint-dir', (Split-Path -Parent $checkpointPath),
        '--step', "$step",
        '--device-batch-size', "$deviceBatchSize",
        '--n-rows', "$nRows",
        '--seed', "$seed",
        '--eval-seed-base', "$evalSeedBase",
        '--allow-cross-above-0.09'
    )
    $evalWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-PythonLogged $evalArgs "eval-$arm-$seed"
    $evalWatch.Stop()
    if ($result.exit_code -ne 0) {
        throw "Evaluation failed for $arm seed $seed with exit code $($result.exit_code)."
    }
    $totalWall = [double]$trainingInfo.wall_seconds + $evalWatch.Elapsed.TotalSeconds
    $peak = $trainingInfo.peak_vram_mib
    $retryCount = [math]::Max(0, [int]$trainingInfo.attempts - 1)
    $entry = Parse-EvalResult $result.output $arm $seed $totalWall $peak $retryCount
    Add-Entry $entry
    Write-LogLine ("RUN_COMPLETE arm={0} seed={1} wall_seconds={2:F3} training_peak_vram_mib={3} retries={4}" -f $arm, $seed, $totalWall, $peak, $retryCount)
    Write-LogLine ("COMPLETED {0} s{1} {2:F3}s" -f $arm, $seed, $totalWall)
    Get-PairDecision $seed
}

if ($TrainingSmoke) {
    if ($DryRun -or $SmokeSeed -eq 43 -or $SmokeIterations -le 0) {
        throw "TrainingSmoke requires a positive iteration count, a non-calibration seed, and no -DryRun."
    }
    if ($SmokeTag -notmatch '^smoke-driverpath-s\d+$') {
        throw "TrainingSmoke tag must match smoke-driverpath-s<seed>."
    }
    $smokeArguments = @(Get-TrainingArguments 'D' $SmokeSeed $SmokeTag)
    $iterationFlag = "--num-iterations=$SmokeIterations"
    $smokeArguments = @($smokeArguments | ForEach-Object {
        if ($_ -eq '--num-iterations=1200') { $iterationFlag } else { $_ }
    })
    $smokeCheckpointDirectory = Join-Path $checkpointRoot $SmokeTag
    $smokeCheckpoint = Join-Path $smokeCheckpointDirectory "model_$('{0:D6}' -f $SmokeIterations).pt"
    Write-LogLine "TRAIN_COMMAND smoke=true arm=D seed=$SmokeSeed $python -m scripts.base_train $($smokeArguments[2..($smokeArguments.Count - 1)] -join ' ')"
    Write-LogLine "TRAIN_BEGIN arm=D seed=$SmokeSeed attempt=1 smoke=true"
    Write-LogLine "TRAIN_START_TIMESTAMP timestamp=$(Get-Date -Format o) arm=D seed=$SmokeSeed attempt=1 smoke=true"
    Write-LogLine "TRAIN_CHILD_ENV arm=D seed=$SmokeSeed attempt=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 WANDB_MODE=disabled WANDB_CONSOLE=off"
    $smokeResult = Invoke-PythonLogged $smokeArguments `
        "smoke-D-$SmokeSeed-attempt-1" -TrainingChild -RunTag $SmokeTag -Attempt 1
    if ($smokeResult.exit_code -ne 0) {
        throw "Training smoke failed for $SmokeTag with exit code $($smokeResult.exit_code)."
    }
    $checkpointBytes = if (Test-Path -LiteralPath $smokeCheckpoint -PathType Leaf) {
        (Get-Item -LiteralPath $smokeCheckpoint).Length
    }
    else {
        0
    }
    Write-LogLine "TRAIN_SMOKE_RESULT tag=$SmokeTag iterations=$SmokeIterations checkpoint=$smokeCheckpoint checkpoint_bytes=$checkpointBytes"
    return
}

$dTemplateCommand = "$python -m scripts.base_train --model-tag recall-D-s<seed>-w20 --seed <seed> --depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024 --window-pattern=L --device-batch-size=2 --total-batch-size=32768 --num-iterations=1200 --eval-every=300 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --segment-len=128 --synthetic-recall-frac=0.5 --recall-answer-weight=20 --rda --rda-decays 0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875"
$dconstTemplateCommand = "$python -m scripts.base_train --model-tag recall-Dconst-s<seed>-w20 --seed <seed> --depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024 --window-pattern=L --device-batch-size=2 --total-batch-size=32768 --num-iterations=1200 --eval-every=300 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --segment-len=128 --synthetic-recall-frac=0.5 --recall-answer-weight=20 --rda --rda-decays 0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875 --rda-const-state"
if (-not $DryRun) {
    Write-LogLine "CALIBRATION_TRAIN_COMMAND_D=$dTemplateCommand"
    Write-LogLine "CALIBRATION_TRAIN_COMMAND_DCONST=$dconstTemplateCommand"
}

foreach ($seed in $trainingSeeds) {
    foreach ($arm in @('D', 'D_const')) {
        $paths = Get-RunPaths $arm $seed
        $runId = "$arm-$seed"
        $evalComplete = $null -ne (Get-RunEntry $runId)
        $trainComplete = Test-TrainingComplete $arm $seed $paths

        if ($DryRun) {
            if ($seed -eq 43) {
                $action = if ($evalComplete) { 'SKIP' } else { 'EVAL' }
                $reason = if ($evalComplete) { 'complete_eval_entry; calibration_checkpoint_reused' } else { 'calibration_checkpoint_eval_only' }
            }
            elseif (-not $trainComplete) {
                $action = 'TRAIN'
                $reason = if (Test-Path -LiteralPath $paths.checkpoint_dir -PathType Container) { 'missing_valid_train_done_marker; incomplete_folder_would_be_deleted' } else { 'no_valid_checkpoint_and_train_done_marker' }
            }
            elseif (-not $evalComplete) {
                $action = 'EVAL'
                $reason = 'train_done_checkpoint_exists; eval_entry_missing_or_incomplete'
            }
            else {
                $action = 'SKIP'
                $reason = 'train_done_checkpoint_and_complete_eval_entry_exist'
            }
            Write-Host ("{0,-8} {1,-5} {2,-5} {3}" -f $action, $arm, "s$seed", $reason)
            continue
        }

        $checkpointExists = Test-Path -LiteralPath $paths.checkpoint -PathType Leaf
        $trainingInfo = $null
        if ($seed -ne 43 -and -not $trainComplete) {
            $trainingInfo = Invoke-Training $arm $seed $paths.tag $paths.checkpoint
        }
        elseif ($seed -eq 43 -and -not $checkpointExists) {
            if ($evalComplete) {
                Write-LogLine "SKIP_EVAL arm=$arm seed=$seed reason=complete_json_entry_exists"
                continue
            }
            throw "Seed-43 calibration checkpoint missing; seed 43 must not be retrained: $($paths.checkpoint)"
        }
        elseif ($seed -ne 43 -and $trainComplete -and -not $evalComplete) {
            Write-LogLine "SKIP_TRAIN arm=$arm seed=$seed reason=step1200_checkpoint_and_train_done_marker_exist"
        }
        Invoke-Evaluation $arm $seed $paths.tag $paths.checkpoint $trainingInfo
    }
}

if (-not $DryRun) {
    $script:data.completed = $true
    $script:data.completed_at = (Get-Date).ToString('o')
    Save-Json
    Write-LogLine "GRID_COMPLETE run_entries=$(@($script:data.entries | Where-Object { $_.type -eq 'run' }).Count) pair_decisions=$(@($script:data.entries | Where-Object { $_.type -eq 'no_context_pair_decision' }).Count)"
}
