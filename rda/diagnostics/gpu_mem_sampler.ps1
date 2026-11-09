param([string]$OutFile, [int]$DurationSec = 1200)
$sw = [System.IO.StreamWriter]::new($OutFile, $false)
$end = (Get-Date).AddSeconds($DurationSec)
while ((Get-Date) -lt $end) {
  try {
    $sh = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples | Where-Object CookedValue -gt 0
    $de = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples | Where-Object CookedValue -gt 0
    $ts = (Get-Date).ToString('HH:mm:ss')
    foreach ($s in $sh) { $sw.WriteLine("$ts SHARED $($s.InstanceName) $($s.CookedValue)") }
    foreach ($s in $de) { $sw.WriteLine("$ts DEDICATED $($s.InstanceName) $($s.CookedValue)") }
    $sw.Flush()
  } catch { }
  Start-Sleep -Seconds 2
}
$sw.Dispose()
