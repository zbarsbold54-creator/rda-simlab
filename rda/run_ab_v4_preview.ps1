# 200-step preview, seed 42, three arms, under a hard wall-clock budget.
#
# NO CODE CHANGES to the trainer: this driver only launches base_train with
# the ab_v2 flags plus --num-iterations=200 and enforces two deadlines:
#   - a per-run cap (25 min for C, 15 min for the others), and
#   - a whole-sequence deadline (35 min), after which nothing further starts.
#
# Timeout handling mirrors run_ab_v3.ps1's verified guard: poll HasExited to
# an exact deadline, kill the TREE (the venv shim spawns the real interpreter
# as a child, so killing only the parent orphans it and keeps the GPU held),
# and cache $p.Handle so ExitCode is readable for an already-exited process.
#
# C is SKIPPED if the first 20 steps project past its cap -- see $SkipC.

param([int]$BudgetMinutes = 35)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
Set-Location "C:\AI\nanochat"

$env:NANOCHAT_DATASET   = "fineweb-edu"
$env:PYTHONIOENCODING    = "utf-8"
$env:PYTHONUNBUFFERED    = "1"

$Deadline = (Get-Date).AddMinutes($BudgetMinutes)
$SkipC = $false

$BASE = @(
    "--depth=8", "--aspect-ratio=96", "--head-dim=128",
    "--max-seq-len=1024", "--window-pattern=L",
    "--device-batch-size=4", "--total-batch-size=32768",
    "--num-iterations=200",
    "--eval-every=50", "--eval-tokens=524288",
    "--core-metric-every=-1", "--sample-every=-1",
    "--seed=42"
)

function Invoke-Run([string]$py, [string[]]$argv, [string]$log, [int]$CapSeconds, [string]$CapLabel) {
    $out = "$log.out"; $err = "$log.err"
    $exe = (Resolve-Path $py).Path
    Write-Host ("EXEC: {0}" -f ($argv -join ' '))
    $p = Start-Process -FilePath $exe -ArgumentList ($argv -join ' ') `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle
    $dl = (Get-Date).AddSeconds($CapSeconds)
    $timedOut = $false
    while (-not $p.HasExited) {
        if ((Get-Date) -ge $dl) { $timedOut = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if ($timedOut) {
        Write-Host ("{0} -- killing process tree for PID {1}" -f $CapLabel, $p.Id)
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        try { & taskkill.exe /PID $p.Id /T /F 2>&1 | Out-Null } catch {}
        Start-Sleep -Seconds 3
        try { $p.WaitForExit(15000) | Out-Null } catch {}
        Add-Content -Path $log -Encoding utf8 -Value $CapLabel
        return [int]124
    }
    $p.WaitForExit()
    return $p.ExitCode
}

# Fold the child's stdout/stderr into the header log, same shape as ab_v3.
function Complete-Log([string]$log, [int]$code) {
    Add-Content -Path $log -Encoding utf8 -Value ""
    Add-Content -Path $log -Encoding utf8 -Value "===== STDOUT ====="
    if (Test-Path "$log.out") { Get-Content "$log.out" -Encoding utf8 | Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Encoding utf8 -Value "===== STDERR ====="
    if (Test-Path "$log.err") { Get-Content "$log.err" -Encoding utf8 | Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Encoding utf8 -Value ("===== END (exit={0}) =====" -f $code)
}

# Final BPB from the LAST eval line, anchored on the run's own step count.
function Get-Bpb([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $t = [System.IO.File]::ReadAllText($path)
    $m = [regex]::Matches($t, 'Step (\d{6}) \| Validation bpb:\s*([0-9]+\.[0-9]+|nan|inf|NaN|Inf)')
    if ($m.Count -eq 0) { return $null }
    return [double]$m[$m.Count - 1].Groups[2].Value
}

function Get-TrainMin([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $m = [regex]::Matches([System.IO.File]::ReadAllText($path), 'Total training time:\s*([0-9.]+)m')
    if ($m.Count -eq 0) { return $null }
    return [double]$m[$m.Count - 1].Groups[1].Value
}

function Get-SecPerStep([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $m = [regex]::Matches([System.IO.File]::ReadAllText($path),
          'step\s+(\d+)/\d+.*?dt:\s+([0-9.]+)ms')
    $vals = @()
    foreach ($x in $m) {
        if ([int]$x.Groups[1].Value -ge 5) { $vals += [double]$x.Groups[2].Value }
    }
    if ($vals.Count -eq 0) { return $null }
    return ($vals | Measure-Object -Average).Average / 1000.0
}

$RESULTS = [ordered]@{
    label = "ab_v4_preview"; seed = 42; num_iterations = 200
    budget_minutes = $BudgetMinutes
    started = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
    runs = [ordered]@{}; skipped = @(); notes = @()
}

# arm, extra flags, cap seconds, cap label
$PLAN = @(
    @{ arm = "Bprime";  flags = @("--rda"); cap = 900;  label = "TIMEOUT after 15 min" },
    @{ arm = "C";       flags = @("--rda", "--rda-const-state"); cap = 1500; label = "TIMEOUT after 25 min" },
    @{ arm = "baseline"; flags = @(); cap = 900; label = "TIMEOUT after 15 min" }
)

foreach ($r in $PLAN) {
    $tag = "ab-v4-preview-$($r.arm)"
    $log = Join-Path "logs" ("ab_v4_preview_{0}.txt" -f $r.arm)

    if ($r.arm -eq "C" -and $SkipC) {
        Write-Host "SKIP C: projected past its cap from the first 20 steps."
        $RESULTS.skipped += "C"; continue
    }
    if ((Get-Date) -ge $Deadline) {
        Write-Host ("SKIP {0}: sequence budget of {1} min exhausted." -f $r.arm, $BudgetMinutes)
        $RESULTS.skipped += $r.arm; continue
    }
    if (Test-Path $log) {
        Write-Host ("ABORT: log exists, refusing to append: {0}" -f $log)
        $RESULTS.notes += "log collision: $log"; break
    }

    $argv = @("-m", "scripts.base_train", "--model-tag", $tag) + $BASE + $r.flags
    Write-Host ""
    Write-Host ("=== START {0} {1} ===" -f $r.arm, (Get-Date -Format "HH:mm:ss"))
    Set-Content -Path $log -Encoding utf8 -Value @(
        "GIT_COMMIT: $((git rev-parse HEAD).Trim())",
        "AB_V4_ARM:   $($r.arm)",
        "AB_V4_SEED:  42",
        "AB_V4_START: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')",
        ("COMMAND: " + ($argv -join " "))
    )

    $t0 = Get-Date
    $code = Invoke-Run $py $argv $log $r.cap $r.label
    $wall = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
    Complete-Log $log $code

    $bpb = Get-Bpb $log
    $train = Get-TrainMin $log
    $sps = Get-SecPerStep $log
    $timedOut = ($code -eq 124)
    Write-Host ("--- {0}: exit={1} wall={2}m train={3}m s/step={4} bpb={5}" -f `
        $r.arm, $code, $wall, $train, $(if($sps){"{0:N3}" -f $sps}else{"n/a"}), $bpb)

    $RESULTS.runs[$r.arm] = [ordered]@{
        arm = $r.arm; exit = $code; timed_out = $timedOut
        bpb_last_eval = $bpb; wall_minutes = $wall
        train_minutes = $train; sec_per_step = $sps; log = $log
    }

    # Decide whether C is worth attempting, from its first 20 steps.
    if ($r.arm -eq "C" -and -not $timedOut) {
        $raw = [System.IO.File]::ReadAllText("$log.out")
        $m = [regex]::Matches($raw, 'step\s+(\d+)/200.*?dt:\s+([0-9.]+)ms')
        $v = @(); foreach ($x in $m) { if ([int]$x.Groups[1].Value -le 20) { $v += [double]$x.Groups[2].Value } }
        if ($v.Count -gt 0) {
            $avg20 = ($v | Measure-Object -Average).Average / 1000.0
            $proj = 200 * $avg20 / 60.0
            Write-Host ("    C first-20 mean {0:N3} s/step -> projects {1:N1} min for 200 (cap 25)" -f $avg20, $proj)
            if ($proj -gt 25) {
                Write-Host "    C projected OVER its 25-min cap -- would have been skipped."
                $RESULTS.notes += ("C projected {0:N1} min > 25 min cap" -f $proj)
            }
        }
    }
}

$RESULTS.finished = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ss')
[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) "logs\ab_v4_preview_results.json"),
    (ConvertTo-Json -InputObject $RESULTS -Depth 6),
    (New-Object System.Text.UTF8Encoding $false))
Write-Host ""
Write-Host "wrote logs\ab_v4_preview_results.json"
Write-Host "=== PREVIEW COMPLETE ==="
