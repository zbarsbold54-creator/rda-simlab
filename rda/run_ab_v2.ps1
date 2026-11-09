# run_ab_v2.ps1 -- VALID post-leak-fix A/B of B' vs baseline.
#
# WHY THIS EXISTS. Every previous B' A/B is invalid: the withdrawn 1.226305 came
# from b-stab-s42, which ran at commit 6ae93ce -- the commit IMMEDIATELY BEFORE
# the label-leak fix 969e067. No arm has ever run on post-fix code. This runs
# both arms at the same HEAD so the only difference is --rda.
#
# METRIC. Validation BPB at the FINAL step (600), parsed from
#   "Step 06000 | Validation bpb: <x>"      (scripts/base_train.py:740)
# NOT "Minimum validation bpb" (line 1065), which is the BEST checkpoint.
# Step 600 always evaluates: line 730 notes the loop runs num_iterations+1
# times specifically so it can eval/save at the end.
#
# INTERLEAVED order limits machine drift (thermal, clock, background load):
#   A42 B42 A43 B43 A44 B44 A42-repeat
# The repeat measures the baseline's own run-to-run noise at this step count,
# which is what the paired differences must beat before any effect is called
# real.
#
# SAFETY. Nothing here edits model/gain/drive/training code. scripts/base_train.py
# is invoked and never modified. The pre-flight runs the label-leak regression
# test and refuses to start if it fails.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$root = "C:\AI\nanochat"
Set-Location $root

# Environment: local shards + UTF-8 console. Both matter -- without the second,
# base_train.py dies on a cp1252 encode error printing a proportionality sign.
$env:NANOCHAT_DATASET = "fineweb-edu"
$env:PYTHONIOENCODING  = "utf-8"

# NOTE THE NAME. This must NOT be called $results: PowerShell variable names
# are CASE-INSENSITIVE, so $results (the ordered dictionary built at line ~169)
# would be the SAME variable and would overwrite this path. That is exactly
# what happened in attempt 2: Set-Content then received the dictionary as its
# -Path, and wrote the JSON to a file literally named
# "System.Collections.Specialized.OrderedDictionary" in the repo root instead
# of to logs/ab_v2_results.json.
$RESULTS_PATH = "logs\ab_v2_results.json"
$LOGDIR  = "logs"

# Identical in both arms. Every value traces to run_sweep.ps1 $BASE_FLAGS
# (the flags used by BOTH historical arms) or to a base_train.py default.
# d_model = depth * aspect-ratio = 8 * 96 = 768.  T = max-seq-len = 1024.
$BASE_FLAGS = @(
    "--depth=8", "--aspect-ratio=96", "--head-dim=128",
    "--max-seq-len=1024", "--window-pattern=L",
    "--device-batch-size=4", "--total-batch-size=32768",
    "--num-iterations=600",
    "--eval-every=50", "--eval-tokens=524288",
    "--core-metric-every=-1", "--sample-every=-1"
)

# The two arms, as ordered pairs. Identical except --rda.
function Build-Cmd([string]$arm, [int]$seed) {
    $tag = "ab-v2-$arm-s$seed"
    $cmd = @("-m", "scripts.base_train", "--model-tag", $tag, "--seed", "$seed") + $BASE_FLAGS
    if ($arm -eq "B") { $cmd += "--rda" }
    return @{ tag = $tag; argv = $cmd }
}

# Invoke one run via Start-Process.
#
# WHY NOT "& $py @argv 2>&1 | Out-File". base_train.py writes its INFO logging
# to STDERR. Under "2>&1" PowerShell wraps each stderr line as an ErrorRecord
# (NativeCommandError), and with $ErrorActionPreference = "Stop" the FIRST such
# line becomes a terminating exception. Attempt 1 died ~6 s into run 1 on
# "nanochat.common - INFO - Distributed world size: 1" with zero training done.
# Start-Process redirects the two streams to real FILES, so nothing is ever
# reinterpreted as an error record and ErrorActionPreference cannot fire.
#
# Argument quoting: Start-Process -ArgumentList takes an ARRAY and joins it,
# but each element must already be a single token. Every element here is either
# "-m"/"scripts.base_train" or a "--flag=value" with no space, so no element can
# be split. The exe path is passed via -FilePath, NOT -ArgumentList, so the
# space in "C:\AI\nanochat\...\python.exe" is handled by the OS, not by us.
function Invoke-Run([string]$py, [string[]]$argv, [string]$log) {
    $out = "$log.out"
    $err = "$log.err"
    $exe = (Resolve-Path $py).Path
    # Single string, NOT an array: an array is re-quoted per element and can
    # corrupt tokens. One joined string passes through verbatim.
    $argLine = ($argv -join ' ')
    # Write-Host, NOT Write-Output: Write-Output lands on the success stream,
    # which the caller captures TOGETHER with the exit code, so the function
    # would return an Object[] [EXEC line, exitcode] and every [int] use of
    # the exit code would fail. Write-Host is not capturable, so this returns
    # ONLY the integer.
    Write-Host ("EXEC: {0} {1}" -f $exe, $argLine)
    $p = Start-Process -FilePath $exe -ArgumentList $argLine `
                       -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    return $p.ExitCode
}

# Parse "Step 00600 | Validation bpb: 1.234567" -> 1.234567.
#
# FORMAT VERIFIED against logs/baseline_train.log, which contains 13 real
# lines. base_train.py:740 is f"Step {step:05d} | Validation bpb: {v:.6f}".
# :05d pads 600 to FIVE digits, so step 600 renders as "00600", NOT "06000".
# An earlier draft of this parser anchored on "Step 06000" and would have
# matched nothing at all.
#
# Anchored on the exact step 600 so a mid-run eval can never be picked up, and
# so "Step 00600" cannot be confused with "Step 06000" or "Step 600".
# Accepts nan/inf so a non-finite value is reported as a NaN halt rather than
# as a silent parse failure.
function Get-FinalBpb([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $text = [System.IO.File]::ReadAllText($path)
    $m = [regex]::Matches($text,
          'Step 00600 \| Validation bpb:\s*([0-9]+\.[0-9]+|nan|inf|-inf|NaN|Inf)')
    if ($m.Count -eq 0) { return $null }
    # More than one match means the log merged two runs, or the step-600 eval
    # printed twice. Either way the number is ambiguous, so refuse it.
    if ($m.Count -gt 1) {
        Write-Output ("ABORT: {0} step-00600 BPB line appears {1} times." -f $path, $m.Count)
        return $null
    }
    $raw = $m[0].Groups[1].Value
    $v = 0.0
    if (-not [double]::TryParse($raw, [ref]$v)) { return [double]::NaN }
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
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: leak fix 969e067 is NOT in HEAD."
    exit 1
}
Write-Output "leak fix 969e067 IS in HEAD"

$dirty = git status --porcelain
if ($dirty) {
    Write-Output "NOTE: tree is DIRTY at start; runs will report TREE: DIRTY."
    Write-Output $dirty
}

Write-Output "running diagnostics\test_no_label_leak.py"
& $py diagnostics\test_no_label_leak.py 2>&1 |
    Select-String -Pattern "RESULT|FAIL|violation" |
    ForEach-Object { $_.ToString() }
if ($LASTEXITCODE -ne 0) {
    Write-Output "ABORT: test_no_label_leak.py FAILED (exit $LASTEXITCODE)."
    exit 1
}
Write-Output "PRE-FLIGHT PASSED"

# ---------------------------------------------------------------- RUN LOOP
# Interleaved: A42 B42 A43 B43 A44 B44 A42R
$order = @(
    @{ arm = "A"; seed = 42; label = "baseline_s42" },
    @{ arm = "B"; seed = 42; label = "B_s42" },
    @{ arm = "A"; seed = 43; label = "baseline_s43" },
    @{ arm = "B"; seed = 43; label = "B_s43" },
    @{ arm = "A"; seed = 44; label = "baseline_s44" },
    @{ arm = "B"; seed = 44; label = "B_s44" },
    @{ arm = "A"; seed = 42; label = "baseline_s42_repeat" }
)

$results = [ordered]@{
    label        = "ab_v2"
    written_utc  = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    head         = $head
    metric_line  = 'Step 00600 | Validation bpb: <x>   (scripts/base_train.py:740)'
    metric_note  = "FINAL-step BPB, not 'Minimum validation bpb' (best checkpoint)"
    base_flags   = $BASE_FLAGS
    runs         = [ordered]@{}
    halted       = $null
}

foreach ($r in $order) {
    $built  = Build-Cmd $r.arm $r.seed
    $tag    = $built.tag
    $argv   = $built.argv
    $repeat = $r.label -like "*repeat*"
    if ($repeat) { $tag = "$tag-repeat" }

    $log = Join-Path $LOGDIR ("ab_v2_{0}_s{1}.txt" -f $r.label, $r.seed)

    # COLLISION GUARD. The script APPENDS, so reusing an existing log would
    # silently merge two runs and make the BPB parse ambiguous. Halt instead.
    if (Test-Path $log) {
        Write-Output ("ABORT: log already exists, refusing to append: {0}" -f $log)
        $results.halted = @{ label = $r.label; exit = -1; bpb = "n/a"
                             nan = $false; reason = "log collision: $log"; log = $log }
        break
    }

    Write-Output ""
    Write-Output ("=== START {0} (arm {1} seed {2}) {3} ===" -f `
        $r.label, $r.arm, $r.seed, (Get-Date -Format "HH:mm:ss"))

    $full = "COMMAND: " + ($argv -join " ")
    Write-Output $full
    # Commit hash goes at the TOP of every log, before any output, so each
    # log is self-identifying even if the run dies mid-write.
    $stamp = @(
        "GIT_COMMIT: $head",
        "AB_LABEL:   $($r.label)",
        "AB_ARM:     $($r.arm)",
        "AB_SEED:    $($r.seed)",
        "AB_START:   $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')",
        $full
    )
    Set-Content -Path $log -Value $stamp -Encoding utf8

    $t0 = Get-Date
    $code = Invoke-Run $py $argv $log
    $wall = [math]::Round(((Get-Date) - $t0).TotalMinutes, 2)

    # Fold both redirected streams into the single per-run log, each under a
    # clear separator, so one file holds header + stdout + stderr in order.
    Add-Content -Path $log -Value "" -Encoding utf8
    Add-Content -Path $log -Value "===== STDOUT =====" -Encoding utf8
    if (Test-Path "$log.out") { Get-Content "$log.out" -Encoding utf8 |
        Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== STDERR =====" -Encoding utf8
    if (Test-Path "$log.err") { Get-Content "$log.err" -Encoding utf8 |
        Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== END (exit=$code) =====" -Encoding utf8
    # The .out/.err files are intermediates; the .txt is the record.
    Remove-Item "$log.out","$log.err" -ErrorAction SilentlyContinue

    $bpb = Get-FinalBpb $log
    $trainMin = Get-WallMinutes $log

    Write-Output ("--- {0}: exit={1} wall={2}m train={3}m bpb={4}" -f `
        $r.label, $code, $wall, $trainMin, $bpb)

    # NaN handling. PowerShell's -ne $null is TRUE for NaN (NaN compares
    # unequal to everything, including $null), so a NaN must be tested with
    # IsNaN FIRST or it would sail through as a normal value.
    $isNan = ($bpb -ne $null) -and ([double]::IsNaN($bpb) -or [double]::IsInfinity($bpb))
    if ($isNan) { $bpb = $bpb }
    if ($isNan -or $code -ne 0 -or $bpb -eq $null) {
        Write-Output ("ABORT: {0} failed. exit={1} bpb={2} nan={3} reason={4}" -f `
            $r.label, $code, $bpb, $isNan,
            $(if ($isNan) { "non-finite BPB" } elseif ($bpb -eq $null) { "BPB line missing" } else { "nonzero exit" }))
        Write-Output "Stopping the sequence and reporting."
        $results.halted = @{
            label = $r.label; exit = $code
            bpb = $(if ($isNan) { "non-finite" } else { $bpb })
            nan = $isNan
            reason = $(if ($isNan) { "non-finite BPB" } elseif ($bpb -eq $null) { "BPB line missing" } else { "nonzero exit" })
            log = $log
        }
        break
    }

    $results.runs[$r.label] = [ordered]@{
        arm = $r.arm; seed = $r.seed; tag = $tag
        bpb_final = $bpb
        wall_minutes = $wall
        train_minutes = $trainMin
        log = $log
    }
}

# ---------------------------------------------------------------- DERIVE
if (-not $results.halted) {
    Write-Output ""
    Write-Output "=== PAIRED COMPARISON (B' minus baseline; negative = B' better) ==="
    $diffs = @()
    foreach ($s in 42, 43, 44) {
        $a = $results.runs["baseline_s$s"].bpb_final
        $b = $results.runs["B_s$s"].bpb_final
        $d = [math]::Round($b - $a, 6)
        $diffs += $d
        Write-Output ("seed {0}: baseline={1:F6}  B'={2:F6}  diff={3:+0.000000;-0.000000}" -f `
            $s, $a, $b, $d)
    }
    $a42  = $results.runs["baseline_s42"].bpb_final
    $a42r = $results.runs["baseline_s42_repeat"].bpb_final
    $repDiff = [math]::Round($a42r - $a42, 6)
    $meanAbs = [math]::Round((($diffs | ForEach-Object { [math]::Abs($_) }) |
                              Measure-Object -Average).Average, 6)
    $allSame = (($diffs | Where-Object { $_ -gt 0 }).Count -eq 0) -or `
               (($diffs | Where-Object { $_ -lt 0 }).Count -eq 0)

    Write-Output ("baseline s42 repeat diff = {0:+0.000000;-0.000000}" -f $repDiff)
    Write-Output ("mean |paired diff|         = {0:F6}" -f $meanAbs)
    Write-Output ("all three same sign       = {0}" -f $allSame)
    Write-Output ("threshold 0.0137 exceeded = {0}" -f ($meanAbs -gt 0.0137))

    $real = $allSame -and ($meanAbs -gt [math]::Abs($repDiff)) -and ($meanAbs -gt 0.0137)
    $results.derived = [ordered]@{
        per_seed_diff = $diffs
        baseline_s42_repeat_diff = $repDiff
        mean_abs_paired_diff = $meanAbs
        all_three_same_sign = $allSame
        beats_repeat = ($meanAbs -gt [math]::Abs($repDiff))
        beats_seed_spread = ($meanAbs -gt 0.0137)
        verdict_real = $real
        verdict_note = "real requires: all three same sign AND mean |diff| > |repeat diff| AND > 0.0137"
    }
    Write-Output ""
    Write-Output ("VERDICT: effect is REAL = {0}" -f $real)
}

# Convert to a string variable FIRST, then write with -LiteralPath. This keeps
# the pipeline out of the write entirely, so no object's ToString() can ever be
# bound as a destination path.
$json = ConvertTo-Json -InputObject $results -Depth 6
Set-Content -LiteralPath $RESULTS_PATH -Value $json -Encoding utf8
Write-Output ""
Write-Output ("wrote {0}" -f $RESULTS_PATH)
Write-Output "=== SWEEP COMPLETE ==="