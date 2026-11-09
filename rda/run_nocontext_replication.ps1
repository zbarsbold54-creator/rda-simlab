param(
    [string]$CheckpointDir,
    [string]$OutFile,
    [string]$RdaDecays = "",
    [int]$Step = 1200,
    [int]$NRows = 16384,
    [int]$DeviceBatchSize = 2,
    [int]$EvalSeedBase = 2000000000
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

$free = (Get-PSDrive C).Free
if ($free / 1GB -le 15) {
    throw ("DISK GATE FAILED: {0:N2} GB free (need >15 GB)" -f ($free / 1GB))
}
Write-Output ("DISK_GATE_OK free={0:N2} GB" -f ($free / 1GB))

$query = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)
if ($LASTEXITCODE -ne 0 -or -not $query) {
    throw ("Unable to query GPU memory. Raw output: {0}; LASTEXITCODE={1}" -f ($query -join "; "), $LASTEXITCODE)
}
$usedMiB = [int]($query[0].Trim())
if ($usedMiB -gt 2500) {
    $holders = (& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv) -join "; "
    Write-Output ("STOP: GPU memory is {0} MiB (>2500 MiB); holders: {1}" -f $usedMiB, $holders)
    exit 1
}
Write-Output ("GPU_MEMORY_PREFLIGHT {0} MiB" -f $usedMiB)

$env:NANOCHAT_DATASET = "fineweb-edu"
$env:PYTHONUNBUFFERED = "1"

$argList = @(
    "-m", "diagnostics.eval_recall",
    "--checkpoint-dir", $CheckpointDir,
    "--step", $Step.ToString(),
    "--device-batch-size", $DeviceBatchSize.ToString(),
    "--n-rows", $NRows.ToString(),
    "--seed", "43",
    "--eval-seed-base", $EvalSeedBase.ToString()
)
if ($RdaDecays) {
    $argList += @("--rda-decays", $RdaDecays)
}

Write-Output ("COMMAND: {0} {1}" -f $py, ($argList -join " "))

$process = Start-Process -FilePath (Resolve-Path $py).Path `
    -ArgumentList $argList `
    -WorkingDirectory (Get-Location).Path `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $OutFile `
    -RedirectStandardError ([System.IO.Path]::ChangeExtension($OutFile, ".err"))

Write-Output ("EXIT_CODE={0}" -f $process.ExitCode)
exit $process.ExitCode
