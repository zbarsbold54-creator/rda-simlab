# Sample nvidia-smi every N seconds into a CSV while a training run is in
# progress. Pure observation: it starts no training, reads no model state and
# writes only its own CSV. Used to answer whether VRAM grows over a run and
# whether the GPU is memory-starved.
#
# Usage: diag_gpu_sample.ps1 -Out <csv> -Seconds <n> [-IntervalSec 15]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][int]$Seconds,
    [int]$IntervalSec = 15
)

$ErrorActionPreference = "Stop"
$end = (Get-Date).AddSeconds($Seconds)
$path = Join-Path (Get-Location) $Out
# Write each row IMMEDIATELY. An earlier version buffered everything and wrote
# the file only after the loop, so stopping the sampler early (which is what
# happens when training finishes first) discarded every sample. Append-per-row
# costs nothing at a 15 s cadence and makes the CSV usable at any instant.
$header = "timestamp,elapsed_s,mem_used_mib,mem_total_mib,util_pct,sm_clock_mhz,temp_c,power_w"
[System.IO.File]::WriteAllText($path, $header + "`r`n",
                              (New-Object System.Text.UTF8Encoding $false))

$sw = [System.Diagnostics.Stopwatch]::StartNew()
while ($true) {
    $now = Get-Date
    $q = & nvidia-smi `
        --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu,power.draw `
        --format=csv,noheader,nounits 2>$null
    if ($q) {
        $parts = ($q -split ',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 6) {
            $row = ("{0},{1:N0},{2},{3},{4},{5},{6},{7}" -f `
                $now.ToString("yyyy-MM-ddTHH:mm:ss"), $sw.Elapsed.TotalSeconds,
                $parts[0], $parts[1], $parts[2], $parts[3], $parts[4], $parts[5])
            [System.IO.File]::AppendAllText($path, $row + "`r`n",
                                           (New-Object System.Text.UTF8Encoding $false))
        }
    }
    if ((Get-Date) -ge $end) { break }
    Start-Sleep -Seconds $IntervalSec
}

$n = ([System.IO.File]::ReadAllLines($path)).Count - 1
Write-Host ("wrote {0} ({1} samples)" -f $Out, $n)
