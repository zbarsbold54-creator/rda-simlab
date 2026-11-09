param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("A", "D", "D_const")]
    [string]$Arm,
    [Parameter(Mandatory=$true)]
    [ValidateSet(5, 20)]
    [int]$AnswerWeight,
    [Parameter(Mandatory=$true)]
    [string]$ModelTag
)

$ErrorActionPreference = "Stop"
Set-Location "C:\AI\nanochat"
$py = ".\.venv\Scripts\python.exe"
$timeoutSeconds = 5400
$timeoutLabel = "TIMEOUT after 90 min"
$decays = "0.9,0.95,0.975,0.9875,0.99375,0.996875,0.9984375,0.99921875"
$resultDir = "results\recall_benchmark\calib"
$log = Join-Path $resultDir ("train_{0}.txt" -f $ModelTag)
$checkpointDir = Join-Path "C:\Users\HiTech\.cache\nanochat\base_checkpoints" $ModelTag

$env:NANOCHAT_DATASET = "fineweb-edu"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
if (Test-Path -LiteralPath $log) {
    throw "Refusing to overwrite existing training log: $log"
}
if (Test-Path -LiteralPath $checkpointDir) {
    throw "Refusing to train into existing checkpoint directory: $checkpointDir"
}

$argv = @(
    "-m", "scripts.base_train",
    "--model-tag", $ModelTag,
    "--seed", "43",
    "--depth=8", "--aspect-ratio=96", "--head-dim=128",
    "--max-seq-len=1024", "--window-pattern=L",
    "--device-batch-size=2", "--total-batch-size=32768",
    "--num-iterations=1200", "--eval-every=300",
    "--eval-tokens=524288", "--core-metric-every=-1",
    "--sample-every=-1", "--segment-len=128",
    "--synthetic-recall-frac=0.5",
    "--recall-answer-weight=$AnswerWeight"
)
if ($Arm -eq "D" -or $Arm -eq "D_const") {
    $argv += @("--rda", "--rda-decays", $decays)
}
if ($Arm -eq "D_const") {
    $argv += "--rda-const-state"
}

$head = (& git rev-parse HEAD).Trim()
$stamp = @(
    "GIT_COMMIT: $head",
    "ARM: $Arm",
    "TAG: $ModelTag",
    "SEED: 43",
    "ANSWER_WEIGHT: $AnswerWeight",
    "START: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')",
    ("COMMAND: {0} {1}" -f (Resolve-Path $py).Path, ($argv -join " "))
)
Set-Content -LiteralPath $log -Value $stamp -Encoding utf8

$query = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)
$queryExit = $LASTEXITCODE
if ($queryExit -ne 0) {
    throw ("Unable to query GPU memory. Raw output: {0}; LASTEXITCODE={1}" -f `
        ($query -join [Environment]::NewLine), $queryExit)
}
$usedMiB = [int]($query | Select-Object -First 1).ToString().Trim()
if ($usedMiB -gt 2500) {
    $holders = (& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv) -join "; "
    Write-Output ("STOP: GPU memory is {0} MiB (>2500 MiB); holders: {1}" -f $usedMiB, $holders)
    exit 2
}
Write-Output ("GPU memory preflight: {0} MiB" -f $usedMiB)

$out = "$log.out"
$err = "$log.err"
$process = Start-Process -FilePath (Resolve-Path $py).Path `
    -ArgumentList ($argv -join " ") -NoNewWindow -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
$null = $process.Handle
$deadline = (Get-Date).AddSeconds($timeoutSeconds)
$timedOut = $false
while (-not $process.HasExited) {
    if ((Get-Date) -ge $deadline) {
        $timedOut = $true
        break
    }
    Start-Sleep -Milliseconds 500
}
if ($timedOut) {
    Write-Output "$timeoutLabel -- killing process tree for PID $($process.Id)"
    try { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue } catch {}
    try { & taskkill.exe /PID $process.Id /T /F 2>&1 | Out-Null } catch {}
    Start-Sleep -Seconds 3
    $exitCode = 124
} else {
    $exitCode = $process.ExitCode
}

Add-Content -LiteralPath $log -Value "" -Encoding utf8
Add-Content -LiteralPath $log -Value "===== STDOUT =====" -Encoding utf8
if (Test-Path -LiteralPath $out) {
    Get-Content -LiteralPath $out -Encoding utf8 | Add-Content -LiteralPath $log -Encoding utf8
}
Add-Content -LiteralPath $log -Value "===== STDERR =====" -Encoding utf8
if (Test-Path -LiteralPath $err) {
    Get-Content -LiteralPath $err -Encoding utf8 | Add-Content -LiteralPath $log -Encoding utf8
}
Add-Content -LiteralPath $log -Value "===== END (exit=$exitCode) =====" -Encoding utf8
Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue

$finalLine = Get-Content -LiteralPath $log -Encoding utf8 |
    Select-String -Pattern '^Step 01200 \| Validation bpb: ([0-9]+(\.[0-9]+)?)'
$bpb = $null
if ($finalLine) {
    $bpb = [double]$finalLine.Matches[0].Groups[1].Value
}
if ($timedOut -or $exitCode -ne 0 -or $null -eq $bpb -or
    [double]::IsNaN($bpb) -or [double]::IsInfinity($bpb)) {
    Write-Output ("ABORT: tag={0} exit={1} timeout={2} final_val_bpb={3} log={4}" -f `
        $ModelTag, $exitCode, $timedOut, $bpb, $log)
    Get-Content -LiteralPath $log -Tail 15
    exit $(if ($timedOut) { 124 } elseif ($exitCode -ne 0) { $exitCode } else { 1 })
}
Write-Output ("TRAIN_COMPLETE tag={0} final_val_bpb={1:F6} log={2}" -f `
    $ModelTag, $bpb, $log)
