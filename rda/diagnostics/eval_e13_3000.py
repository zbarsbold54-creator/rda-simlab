"""E13-fixed-3000: apply the DECLARED criteria, report PASS/FAIL first.

CRITERIA -- declared before the run, not after. Not to be reinterpreted here.

  PASS = max|E| <= 0.98 for ALL 3000 steps
         AND windowed max|E| over the LAST 1000 steps (100-step windows)
             is non-increasing, i.e. no window's max exceeds the previous
             window's max.
  FAIL = either violated. An abort (|E| > 0.98 for 10 consecutive steps)
         counts as FAIL.

Everything else printed is INFORMATIONAL and carries no pass/fail.

INFORMATIONAL
  - applied gain max|1 + _unit_rms(fs, 0.5)| every 500 steps
  - signed window-to-window change in max|E| for all 30 windows
  - per-dimension max|E| per window, and which dimension carries the max
  - PE max every 500 steps

Read-only analysis of logs/pe_3000_fixed.json. No training.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/pe_3000_fixed.json"
BOUND = 0.98
W = 100
LAST = 1000          # criterion 2 applies to the final 1000 steps only
DIM_NAMES = ["novelty", "uncertainty", "threat", "integrity",
             "agency", "learning_progress", "arousal", "homeostatic_dev"]


def main():
    d = json.load(open(PATH))
    E = [r["E_absmax"] for r in d]
    n = len(d)

    # ---- abort detection (the probe may have stopped early) ----
    over = [i for i, v in enumerate(E) if v > BOUND]
    run = 0
    longest = 0
    for v in E:
        run = run + 1 if v > BOUND else 0
        longest = max(longest, run)
    aborted = longest >= 10
    early = n < 3000

    peak = max(E)
    peak_step = E.index(peak)

    # ---- criterion 1 ----
    c1 = (peak <= BOUND) and not aborted

    # ---- criterion 2: last 1000 steps, 100-step windows ----
    start = max(0, n - LAST)
    wins = []
    for lo in range(start, n, W):
        hi = min(lo + W, n)
        if hi <= lo:
            continue
        wins.append((lo, hi, max(E[lo:hi])))
    viol = []
    for i in range(1, len(wins)):
        delta = wins[i][2] - wins[i - 1][2]
        if delta > 0:
            viol.append((wins[i][0], wins[i][1], delta))
    c2 = len(viol) == 0

    passed = c1 and c2 and not aborted

    # ================= VERDICT FIRST =================
    print("#" * 78)
    print("# E13-fixed-3000")
    print("#" * 78)
    print("  steps recorded            : %d%s"
          % (n, "   (ABORTED EARLY)" if early else ""))
    print("  peak max|E|               : %.4f at step %d  (bound %.2f, margin %.4f)"
          % (peak, peak_step, BOUND, BOUND - peak))
    print("  steps with |E| > %.2f       : %d" % (BOUND, len(over)))
    print("  longest consecutive over  : %d  (abort needs 10)" % longest)
    print("")
    print("  CRITERION 1  max|E| <= %.2f for all steps" % BOUND)
    print("               : %s" % ("PASS" if c1 else "FAIL"))
    print("  CRITERION 2  windowed max|E| non-increasing over last %d steps" % LAST)
    print("               : %s   (%d of %d windows above the previous)"
          % ("PASS" if c2 else "FAIL", len(viol), len(wins)))
    if aborted:
        print("  ABORT RULE               : FIRED -> FAIL")
    print("")
    print("  ==> E13-fixed-3000: %s" % ("PASS" if passed else "FAIL"))
    print("#" * 78)

    # ================= informational =================
    print()
    print("INFORMATIONAL -- no pass/fail attached to any of this")
    print()

    print("-" * 78)
    print("A. applied gain max|1 + _unit_rms(fs, %.2f)| and PE max, every 500 steps" % 0.5)
    print("-" * 78)
    print("%7s %14s %14s %12s %12s"
          % ("step", "applied gain", "raw max|1+fs|", "PE max", "max|E|"))
    for s in range(0, n, 500):
        if s < n:
            r = d[s]
            print("%7d %14.4f %14.4f %12.3f %12.4f"
                  % (s, r.get("applied_scale_absmax", float("nan")),
                     r["mod_scale_absmax"], r["pe_max"], r["E_absmax"]))
    if (n - 1) not in range(0, n, 500):
        r = d[-1]
        print("%7d %14.4f %14.4f %12.3f %12.4f"
              % (n - 1, r.get("applied_scale_absmax", float("nan")),
                 r["mod_scale_absmax"], r["pe_max"], r["E_absmax"]))
    ap = [r.get("applied_scale_absmax", float("nan")) for r in d]
    ap_ok = [v for v in ap if v == v]
    print("  applied gain over run: min %.4f  max %.4f  final %.4f"
          % (min(ap_ok), max(ap_ok), ap[-1]))
    chk = [r["applied_gain_rms_check"] for r in d[1:]]
    print("  applied_gain_rms_check over steps 1..%d: min %.4f max %.4f "
          "(expected %.2f)" % (n - 1, min(chk), max(chk), 0.5))
    print("  PE max over run: %.4f at step %d   final %.4f"
          % (max(r["pe_max"] for r in d),
             max(range(n), key=lambda i: d[i]["pe_max"]), d[-1]["pe_max"]))

    print()
    print("-" * 78)
    print("B. windowed max|E| -- ALL 30 windows, signed change vs previous")
    print("-" * 78)
    print("%-14s %10s %12s %10s" % ("window", "max|E|", "change", "verdict"))
    prev = None
    for lo, hi, mx in wins:
        if prev is None:
            print("%4d-%4d %10.4f %12s %10s" % (lo, hi - 1, mx, "--", "first"))
        else:
            dlt = mx - prev
            print("%4d-%4d %10.4f %+12.4f %10s"
                  % (lo, hi - 1, mx, dlt, "RISE" if dlt > 0 else "no"))
        prev = mx
    print("  rising windows: %d %s"
          % (len(viol), [v[0] for v in viol] if viol else ""))
    print("  (criterion 2 covers only windows starting at step >= %d)" % start)

    print()
    print("-" * 78)
    print("C. per-dimension max|E| per window, and which dimension carries the max")
    print("-" * 78)
    for lo in range(0, n, W):
        hi = min(lo + W, n)
        if hi <= lo:
            continue
        seg = d[lo:hi]
        # E_perdim[r] is per-dimension max over batch and positions
        best = [0.0] * 8
        for r in seg:
            pd = r.get("E_perdim") or []
            for k, v in enumerate(pd[:8]):
                if v > best[k]:
                    best[k] = v
        top = max(range(8), key=lambda k: best[k])
        star = " <<<" if lo >= start else ""
        print("  %4d-%4d  " % (lo, hi - 1)
              + " ".join("%.3f" % v for v in best)
              + "   max=%s (%.4f)%s" % (DIM_NAMES[top], best[top], star))
    print("  (rows starting at step >= %d are the criterion-2 windows)" % start)


if __name__ == "__main__":
    main()
