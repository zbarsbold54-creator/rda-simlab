# dry_test_wrapper.ps1 -- NO TRAINING. Exercises the run_ab_v2.ps1
# Start-Process wrapper against a tiny python FILE, not "-c".
#
# WHY A FILE, NOT "-c": an earlier version of this test used
#   python -c "import sys; print('X'); ..."
# and it FAILED with SyntaxError, because the code string contains SPACES and
# Start-Process -ArgumentList splits the joined string on whitespace, so python
# received a bare "import". That is a real property of the wrapper, so the test
# must not hide it: an argument containing a space CANNOT survive the join.
#
# It does not matter for the sweep, because EVERY sweep argument is space-free
# ("-m", "scripts.base_train", "--depth=8", ... "--rda"). What MUST be proven is
# that those pass through intact, that the exe path (which contains no space and
# is passed via -FilePath, not -ArgumentList) resolves, and that both streams
# land in one log with no error-record text.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# Payload: one stdout line, one INFO-style stderr line (the exact shape that
# killed attempt 1), and an echo of the argv it received so the test can PROVE
# each --flag=value arrived whole and in order.
$payload = @'
import sys
print("STDOUT_LINE_OK")
sys.stderr.write("INFO nanochat.common - Distributed world size: 1\n")
sys.stderr.flush()
print("ARGV_RECEIVED:" + "|".join(sys.argv[1:]))
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
'@
New-Item -ItemType Directory -Force -Path logs\dry | Out-Null
$pyfile = "logs\dry\payload.py"
Set-Content -Path $pyfile -Value $payload -Encoding utf8

# Wrapper copied VERBATIM from run_ab_v2.ps1.
function Invoke-Run([string]$py, [string[]]$argv, [string]$log) {
    $out = "$log.out"
    $err = "$log.err"
    $exe = (Resolve-Path $py).Path
    $argLine = ($argv -join ' ')
    Write-Host ("EXEC: {0} {1}" -f $exe, $argLine)
    $p = Start-Process -FilePath $exe -ArgumentList $argLine `
                       -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    return $p.ExitCode
}

function Fold([string]$log, [int]$code) {
    Add-Content -Path $log -Value "" -Encoding utf8
    Add-Content -Path $log -Value "===== STDOUT =====" -Encoding utf8
    if (Test-Path "$log.out") { Get-Content "$log.out" -Encoding utf8 |
        Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== STDERR =====" -Encoding utf8
    if (Test-Path "$log.err") { Get-Content "$log.err" -Encoding utf8 |
        Add-Content -Path $log -Encoding utf8 }
    Add-Content -Path $log -Value "===== END (exit=$code) =====" -Encoding utf8
    Remove-Item "$log.out","$log.err" -ErrorAction SilentlyContinue
}

Get-ChildItem logs\dry -Filter *.txt -ErrorAction SilentlyContinue | Remove-Item -Force

# ---------------------------------------------------------------- TEST A
# argv deliberately mirrors the real sweep: space-free tokens, --flag=value
# pairs, a positional module name. Exit 0.
Write-Output "########## TEST A: exit 0, sweep-shaped argv ##########"
$logA = "logs\dry\dry_a.txt"
$argvA = @($pyfile, "0", "--depth=8", "--aspect-ratio=96", "--max-seq-len=1024",
          "--model-tag", "ab-v2-A-s42", "--seed", "42")
Set-Content -Path $logA -Value @("GIT_COMMIT: drytest","COMMAND: " + ($argvA -join ' ')) -Encoding utf8
$codeA = Invoke-Run $py $argvA $logA
Fold $logA $codeA
Write-Output ("TEST A exit code = {0}   (expected 0)" -f $codeA)

# ---------------------------------------------------------------- TEST B
# Same shape, exit 3 -- what base_train.py's saturation abort raises.
Write-Output ""
Write-Output "########## TEST B: exit 3, must halt ##########"
$logB = "logs\dry\dry_b.txt"
$argvB = @($pyfile, "3", "--depth=8", "--rda")
Set-Content -Path $logB -Value @("GIT_COMMIT: drytest","COMMAND: " + ($argvB -join ' ')) -Encoding utf8
$codeB = Invoke-Run $py $argvB $logB
Fold $logB $codeB
Write-Output ("TEST B exit code = {0}   (expected 3)" -f $codeB)

# ---------------------------------------------------------------- VERDICTS
Write-Output ""
Write-Output "########## VERDICTS ##########"
$a = [System.IO.File]::ReadAllText((Resolve-Path $logA))
$b = [System.IO.File]::ReadAllText((Resolve-Path $logB))
$expectedArgv = "ARGV_RECEIVED:0|--depth=8|--aspect-ratio=96|--max-seq-len=1024|--model-tag|ab-v2-A-s42|--seed|42"

$checks = @(
  @{ n = "A: exit code is 0";                    ok = ($codeA -eq 0) },
  @{ n = "A: stdout line present";               ok = $a.Contains("STDOUT_LINE_OK") },
  @{ n = "A: stderr INFO line present";          ok = $a.Contains("Distributed world size: 1") },
  @{ n = "A: stdout under STDOUT separator";     ok = ($a -match "(?s)===== STDOUT =====.*STDOUT_LINE_OK") },
  @{ n = "A: stderr under STDERR separator";     ok = ($a -match "(?s)===== STDERR =====.*Distributed world size") },
  @{ n = "A: end marker records exit=0";         ok = $a.Contains("===== END (exit=0) =====") },
  @{ n = "A: every --flag=value intact + order"; ok = $a.Contains($expectedArgv) },
  @{ n = "B: exit code is 3";                    ok = ($codeB -eq 3) },
  @{ n = "B: nonzero exit halts";                ok = ($codeB -ne 0) },
  @{ n = "B: stderr line present";               ok = $b.Contains("Distributed world size: 1") },
  @{ n = "B: end marker records exit=3";         ok = $b.Contains("===== END (exit=3) =====") },
  @{ n = "NO NativeCommandError in A";           ok = (-not $a.Contains("NativeCommandError")) },
  @{ n = "NO RemoteException in A";              ok = (-not $a.Contains("RemoteException")) },
  @{ n = "NO NativeCommandError in B";           ok = (-not $b.Contains("NativeCommandError")) },
  @{ n = "NO RemoteException in B";              ok = (-not $b.Contains("RemoteException")) }
)

$fail = 0
foreach ($c in $checks) {
    $mark = if ($c.ok) { "PASS" } else { "FAIL" }
    if (-not $c.ok) { $fail++ }
    Write-Output ("[{0}] {1}" -f $mark, $c.n)
}
Write-Output ""
Write-Output ("DRY TEST FAILURES: {0}" -f $fail)
Write-Output ""
Write-Output "########## LOG A (verbatim) ##########"
Write-Output $a
Write-Output "########## LOG B (verbatim) ##########"
Write-Output $b
if ($fail -gt 0) { Write-Output "DO NOT LAUNCH."; exit 1 }
Write-Output "DRY TESTS PASSED - safe to launch."