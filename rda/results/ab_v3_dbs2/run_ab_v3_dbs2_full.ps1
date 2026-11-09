# run_ab_v3.ps1 -- CONTROL ARM C vs B'. Same driver shape as run_ab_v2.ps1:
# Start-Process with separated stdout/stderr, folded into ONE log per run,
# header at the top carrying the commit hash, halt on any failure.
#
# ORDER: C s42, C s43, C s44, then B' s42 AGAIN at the new HEAD as a
# code-drift check against the earlier B' s42 (1.278289). The drift check runs
# LAST so a failure there cannot cost the three C runs.
#
# METRIC: validation BPB at the FINAL step, from `Step 00600 | Validation bpb:`
# (base_train.py:740). NOT `Minimum validation bpb`, which is the best
# checkpoint. The {:05d} format renders step 600 as 00600.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
Set-Location "C:\AI\nanochat"

# Per-run wall-clock cap in SECONDS. 90 min = 5400 s. Stored in seconds so the
# timeout-path self-test can pass a 5-second limit without a fractional-minute
# value rounding to 0 (an earlier draft printed "TIMEOUT after 0 min").
$TIMEOUT_SECONDS = 5400

# Literal text required in the log on expiry. Derived from the cap rather than
# hardcoded, so it can never disagree with the value actually enforced.
$TIMEOUT_LABEL = "TIMEOUT after {0} min" -f [int]($TIMEOUT_SECONDS / 60)

# Local shards + UTF-8 console. Without the second, base_train.py dies on a
# cp1252 encode error printing a proportionality sign.
$env:NANOCHAT_DATASET = "fineweb-edu"
$env:PYTHONIOENCODING  = "utf-8"
# Unbuffered stdout, inherited by the child through Start-Process. Without it
# python block-buffers a redirected stream in ~8 KB chunks, so the log lags the
# run by thousands of steps: during the aborted 600-step C run the file showed
# "step 200" (really 200 MICRO-steps) while the trainer was at outer step 123,
# and a 45 s poll saw zero new bytes. That made the measured pace wrong by 1.8x
# and briefly pointed at a nonexistent stall. Step lines then appear as they
# are printed, which is what makes progress and the 90-minute cap auditable.
# Set on $env (inherited) rather than passed per-process so it applies to the
# venv shim and the real interpreter alike.
$env:PYTHONUNBUFFERED = "1"

# NOTE THE NAME. Must not be $results -- PowerShell variable names are
# case-insensitive, so that would collide with the ordered dictionary below and
# overwrite this path.
$RESULTS_PATH = "logs\ab_v3_dbs2_full_results.json"

$BASE_FLAGS = @(
    "--depth=8", "--aspect-ratio=96", "--head-dim=128",
    "--max-seq-len=1024", "--window-pattern=L",
    "--device-batch-size=2", "--total-batch-size=32768",
    "--num-iterations=600",
    "--eval-every=50", "--eval-tokens=524288",
    "--core-metric-every=-1", "--sample-every=-1"
)

function Build-Cmd([string]$arm, [int]$seed) {
    $tag = "ab-v3-$arm-s$seed-dbs2-full"
    $cmd = @("-m", "scripts.base_train", "--model-tag", $tag, "--seed", "$seed") + $BASE_FLAGS
    if ($arm -eq "Bprime") { $cmd += "--rda" }
    if ($arm -eq "C")      { $cmd += @("--rda", "--rda-const-state") }
    return @{ tag = $tag; argv = $cmd }
}

# Write-Host, NOT Write-Output: Write-Output lands on the success stream, which
# the caller captures TOGETHER with the exit code, so the function would return
# Object[] [EXEC line, exitcode] and every [int] use would fail.
#
# TIMEOUT: waits up to $TimeoutMinutes for the process, then kills the whole
# tree. Start-Process -Wait has no timeout of its own, so the wait is done
# with Wait-Process -Timeout, which throws on expiry. The throw is caught
# below and turned into a halt; it must NOT be allowed to escape, because
# $ErrorActionPreference = "Stop" would abort the driver mid-run and skip the
# JSON write, losing every completed run.
#
# KILLING THE TREE, NOT JUST THE PARENT: the venv launcher (.venv\Scripts\
# python.exe) is a shim that spawns the REAL interpreter as a child (observed:
# shim PID 16904 -> interpreter PID 31736). Killing only the shim orphans the
# interpreter, which keeps the GPU and the log file handle alive. That is why
# the earlier manual stop had to hunt for the child by command line. stop -Id
# on the parent plus taskkill /T on it covers both.
function Invoke-Run([string]$py, [string[]]$argv, [string]$log, [int]$TimeoutSeconds, [string]$TimeoutLabel) {
    $out = "$log.out"
    $err = "$log.err"
    $exe = (Resolve-Path $py).Path
    $argLine = ($argv -join ' ')
    Write-Host ("EXEC: {0} {1}" -f $exe, $argLine)
    $p = Start-Process -FilePath $exe -ArgumentList $argLine `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    # CACHE THE PROCESS HANDLE. Without this, .ExitCode reads as $null for any
    # process that has already exited by the time it is queried -- PowerShell
    # cannot get the exit code from a handle it never held, and the process
    # object reports an empty ExitCode rather than throwing. Verified directly:
    # `cmd /c exit 5` gave ExitCode=[] with no cached handle and ExitCode=[5]
    # once $p.Handle was touched. Touching .Handle is the documented way to
    # make the value available, and it costs nothing.
    $null = $p.Handle
    $timedOut = $false
    # POLLING, NOT Wait-Process. Wait-Process -Timeout raises when the PID can
    # no longer be found, which is ALSO what happens when a process exits
    # quickly and is reaped before the wait begins. A self-test with
    # `cmd /c exit 0` reproduced exactly that: the process was gone on arrival
    # and the run was wrongly reported as a timeout. Checking HasExited first
    # and then polling removes the race entirely and makes the deadline exact.
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while (-not $p.HasExited) {
        if ((Get-Date) -ge $deadline) { $timedOut = $true; break }
        # Poll often enough that a kill lands promptly, cheaply enough not to
        # matter. Sleep first so a process that exits instantly is not polled.
        Start-Sleep -Milliseconds 500
    }
    if ($timedOut) {
        Write-Host ("{0} -- killing process tree for PID {1}" -f $TimeoutLabel, $p.Id)
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        # /T kills the whole tree, which is what reclaims the GPU.
        try { & taskkill.exe /PID $p.Id /T /F 2>&1 | Out-Null } catch {}
        # Give the OS a moment to release the GPU context and file handles so
        # the folded log is not written while the child still holds $log.out.
        Start-Sleep -Seconds 3
        try { $p.WaitForExit(15000) | Out-Null } catch {}
        Add-Content -Path $log -Encoding utf8 -Value $TimeoutLabel
        return [int]124
    }
    $p.WaitForExit()
    return $p.ExitCode
}

# Anchored on step 00600 so a mid-run eval cannot be picked up. Accepts
# nan/inf so a non-finite value is reported as a NaN halt, not a parse failure.
function Get-FinalBpb([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $text = [System.IO.File]::ReadAllText($path)
    $m = [regex]::Matches($text,
          'Step 00600 \| Validation bpb:\s*([0-9]+\.[0-9]+|nan|inf|-inf|NaN|Inf)')
    if ($m.Count -eq 0) { return $null }
    if ($m.Count -gt 1) {
        Write-Output ("ABORT: {0} step-00600 BPB line appears {1} times." -f $path, $m.Count)
        return $null
    }
    $v = 0.0
    if (-not [double]::TryParse($m[0].Groups[1].Value, [ref]$v)) { return [double]::NaN }
    return $v
}

function Get-WallMinutes([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $m = [regex]::Matches([System.IO.File]::ReadAllText($path),
                          'Total training time:\s*([0-9]+\.?[0-9]*)m')
    if ($m.Count -eq 0) { return $null }
    return [double]($m[$m.Count - 1].Groups[1].Value)
}

# ---------------------------------------------------------------- PRE-FLIGHT
Write-Output "=== PRE-FLIGHT ==="
$head = (git rev-parse HEAD).Trim()
Write-Output "HEAD = $head"
git merge-base --is-ancestor 969e067 HEAD
if ($LASTEXITCODE -ne 0) { Write-Output "ABORT: leak fix 969e067 not in HEAD."; exit 1 }
Write-Output "leak fix 969e067 IS in HEAD"

$dirty = git status --porcelain
if ($dirty) {
    Write-Output "NOTE: tree DIRTY at start; runs will report TREE: DIRTY."
    Write-Output $dirty
}

Write-Output "running diagnostics\test_no_label_leak.py"
& $py diagnostics\test_no_label_leak.py 2>&1 |
    Select-String -Pattern "RESULT|FAIL|violation" | ForEach-Object { $_.ToString() }
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: test_no_label_leak.py FAILED (exit $LASTEXITCODE)."
    exit 1
}
Write-Output "PRE-FLIGHT PASSED"

# ---------------------------------------------------------------- RUN LOOP
$order = @(
    @{ arm = "A";      seed = 42; label = "A_s42_dbs2_full" },
    @{ arm = "A";      seed = 43; label = "A_s43_dbs2_full" },
    @{ arm = "Bprime"; seed = 43; label = "Bprime_s43_dbs2_full" },
    @{ arm = "C";      seed = 43; label = "C_s43_dbs2_full" },
    @{ arm = "A";      seed = 44; label = "A_s44_dbs2_full" },
    @{ arm = "Bprime"; seed = 44; label = "Bprime_s44_dbs2_full" },
    @{ arm = "C";      seed = 44; label = "C_s44_dbs2_full" }
)

$results = [ordered]@{
    label       = "ab_v3_dbs2"
    written_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    head        = $head
    metric_line = 'Step 00600 | Validation bpb: <x>   (scripts/base_train.py:740)'
    base_flags  = $BASE_FLAGS
    arms        = [ordered]@{
        baseline_ref = "from ab_v2 at commit f272938"
        Bprime_ref   = "from ab_v2 at commit f272938"
        C            = "--rda --rda-const-state"
    }
    runs        = [ordered]@{}
    halted      = $null
    timeout_seconds_per_run = $TIMEOUT_SECONDS
    timeout_label = $TIMEOUT_LABEL
}

foreach ($r in $order) {
    $built = Build-Cmd $r.arm $r.seed
    $tag   = $built.tag
    $argv  = $built.argv
    $log   = Join-Path "logs" ("ab_v3_dbs2_full_{0}_s{1}.txt" -f $r.label, $r.seed)

    # COLLISION GUARD: the script appends, so reusing a log would merge runs.
    if (Test-Path $log) {
        Write-Output ("ABORT: log exists, refusing to append: {0}" -f $log)
        $results.halted = @{ label = $r.label; exit = -1; bpb = "n/a"; nan = $false
                            reason = "log collision"; log = $log }
        break
    }

    Write-Output ""
    Write-Output ("=== START {0} (arm {1} seed {2}) {3} ===" -f `
        $r.label, $r.arm, $r.seed, (Get-Date -Format "HH:mm:ss"))
    Write-Output ("COMMAND: " + ($argv -join " "))

    $stamp = @(
        "GIT_COMMIT: $head",
        "AB_V3_LABEL: $($r.label)",
        "AB_V3_ARM:   $($r.arm)",
        "AB_V3_SEED:  $($r.seed)",
        "AB_V3_START: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')",
        ("COMMAND: " + ($argv -join " "))
    )
    Set-Content -Path $log -Value $stamp -Encoding utf8

    $t0 = Get-Date
    $code = Invoke-Run $py $argv $log $TIMEOUT_SECONDS $TIMEOUT_LABEL
    $wall = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)
    # 124 is the conventional timeout status and is distinct from any real
    # exit code, so it is checked BEFORE the generic nonzero-exit branch.
    $timedOut = ($code -eq 124)

    Add-Content -Path $log -Value "" -Encoding utf8
    Add-Content -Path $log -Value "===== STDOUT =====" -Encoding utf8
    if (Test-Path "$log.out") { Get-Content "$log.out" -Encoding utf8 | Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== STDERR =====" -Encoding utf8
    if (Test-Path "$log.err") { Get-Content "$log.err" -Encoding utf8 | Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== END (exit=$code) =====" -Encoding utf8
    Remove-Item "$log.out","$log.err" -ErrorAction SilentlyContinue

    $bpb = Get-FinalBpb $log
    $trainMin = Get-WallMinutes $log
    Write-Output ("--- {0}: exit={1} wall={2}m train={3}m bpb={4}" -f `
        $r.label, $code, $wall, $trainMin, $bpb)

    $isNan = ($bpb -ne $null) -and ([double]::IsNaN($bpb) -or [double]::IsInfinity($bpb))
    if ($timedOut) {
        Write-Output ("ABORT: {0} {1} (cap {2} s = {3} min, wall {4} min)." -f `
            $r.label, $TIMEOUT_LABEL, $TIMEOUT_SECONDS, ($TIMEOUT_SECONDS / 60), $wall)
        $results.halted = [ordered]@{
            label  = $r.label
            arm    = $r.arm
            seed   = $r.seed
            exit   = $code
            timed_out = $true
            timeout_seconds = $TIMEOUT_SECONDS
            wall_minutes = $wall
            bpb    = $bpb        # $null: no step-00600 line, run was incomplete
            nan    = $isNan
            reason = $TIMEOUT_LABEL
            log    = $log
        }
        break
    }
    if ($isNan -or $code -ne 0 -or $bpb -eq $null) {
        $reason = if ($isNan) { "non-finite BPB" } elseif ($bpb -eq $null) { "BPB line missing" } else { "nonzero exit" }
        Write-Output ("ABORT: {0} failed. exit={1} bpb={2} reason={3}" -f $r.label, $code, $bpb, $reason)
        $results.halted = @{ label = $r.label; exit = $code; nan = $isNan; reason = $reason; log = $log }
        break
    }

    $results.runs[$r.label] = [ordered]@{
        arm = $r.arm; seed = $r.seed; tag = $tag
        bpb_final = $bpb; wall_minutes = $wall
        train_minutes = $trainMin; log = $log
    }
}

$json = ConvertTo-Json -InputObject $results -Depth 6
Set-Content -LiteralPath $RESULTS_PATH -Value $json -Encoding utf8
Write-Output ""
Write-Output ("wrote {0}" -f $RESULTS_PATH)
Write-Output "=== SWEEP COMPLETE ==="
