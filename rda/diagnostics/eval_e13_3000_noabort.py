"""E13-fixed-3000 (no-abort): resolve the truncation ambiguity.

WHY THIS RUN EXISTS. The aborted run stopped at step 2324 with max|E| = 0.9814
and 261 steps above 0.98. Because it stopped AT the crossing, it cannot
distinguish "the state was about to recover" from "the state was still
climbing". This run raises the abort bound to 0.999 so the trajectory past
the crossing is observed. Nothing about the model, the gain, or the drive
recurrence is changed: the abort is a post-hoc check on an already-computed
row.

IMPORTANT: THE TWO RUNS ARE NOT BIT-IDENTICAL, contrary to what an earlier
draft of this docstring claimed. Measured over the 2325-step overlap by
diagnostics/compare_abort_runs.py, max |A - B| is 2.08e-02 on E_absmax (mean
2.50e-03), 5.39e-02 on loss, 7.04 on pe_max, and 6.10e-01 on
mod_scale_absmax; NO field is bit-identical. The divergence is visible by
step 25, far before the abort bound could matter, so it is GPU
nondeterminism in the backward pass (compute_init seeds the RNGs, but
nothing calls torch.use_deterministic_algorithms), not the abort bound.

Consequence: this run is an INDEPENDENT trajectory at the same seed, not a
continuation of the aborted one. Any agreement between them is
statistical, not an identity. The truncation question is therefore answered
by a second sample rather than by resuming the first.

CRITERIA -- declared before the previous run, carried over unchanged.
  PASS = max|E| <= 0.98 for ALL 3000 steps
         AND windowed max|E| over the LAST 1000 steps (100-step windows)
             is non-increasing.
  FAIL = either violated.

This script evaluates criterion 2 on the TRUE last 1000 steps, which the
aborted run could not reach.

Read-only analysis. No training.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/pe_3000_fixed_noabort.json"
BOUND = 0.98
W = 100
LAST = 1000
DIM_NAMES = ["novelty", "uncertainty", "threat", "integrity",
             "agency", "learning_progress", "arousal", "homeostatic_dev"]


def main():
    d = json.load(open(PATH))
    n = len(d)
    E = [r["E_absmax"] for r in d]
    peak, pstep = max(E), E.index(max(E))
    over = [i for i, v in enumerate(E) if v > BOUND]
    first_over = over[0] if over else None

    c1 = peak <= BOUND
    start = n - LAST
    wins = []
    for lo in range(start, n, W):
        hi = min(lo + W, n)
        if hi > lo:
            wins.append((lo, hi, max(E[lo:hi])))
    viol = [(w[0], w[1], w[2] - wins[i - 1][2])
            for i, w in enumerate(wins) if i > 0 and w[2] > wins[i - 1][2]]
    c2 = len(viol) == 0

    print("#" * 78)
    print("# E13-fixed-3000 (no-abort)")
    print("#" * 78)
    print("  steps recorded        : %d%s" % (n, "" if n >= 3000 else "  (truncated)"))
    print("  peak max|E|           : %.4f at step %d   (bound %.2f, margin %+.4f)"
          % (peak, pstep, BOUND, BOUND - peak))
    print("  steps with |E| > %.2f   : %d" % (BOUND, len(over)))
    print("  first such step       : %s" % first_over)
    print("  final max|E|          : %.4f at step %d" % (E[-1], n - 1))
    print("")
    print("  CRITERION 1  max|E| <= %.2f for all steps   : %s"
          % (BOUND, "PASS" if c1 else "FAIL"))
    print("  CRITERION 2  windowed max|E| non-increasing  : %s"
          % ("PASS" if c2 else "FAIL"))
    print("               over steps %d-%d  (%d of %d windows rise)"
          % (start, n - 1, len(viol), len(wins)))
    print("")
    print("  ==> E13-fixed-3000 (no-abort): %s" % ("PASS" if (c1 and c2) else "FAIL"))
    print("#" * 78)

    # ---------- (a) threat behaviour ----------
    print()
    print("=" * 78)
    print("(a) DOES threat STAY ABOVE 0.98, SATURATE, OR RECOVER?")
    print("=" * 78)
    tE = [r["threat_E_absmax"] for r in d]
    print("  %7s %10s %12s %12s %12s %10s"
          % ("step", "threat|E|", "tanh_arg", "drive_pre", "applied gain", "PE max"))
    for s in list(range(0, n, 250)) + [n - 1]:
        if s < n:
            r = d[s]
            print("  %7d %10.4f %12.4f %12.4f %12.4f %10.3f"
                  % (s, r["threat_E_absmax"], r["threat_tanh_arg_absmax"],
                     r["threat_drive_preclip_absmax"],
                     r.get("threat_applied_gain", float("nan")), r["pe_max"]))
    if tE[-1] > BOUND:
        state = "STAYS ABOVE %.2f at step %d" % (BOUND, n - 1)
    elif max(tE[-500:]) <= BOUND:
        state = "RECOVERED, below %.2f for the last 500 steps" % BOUND
    else:
        state = "crossed back below"
    print("  threat max|E|: peak %.4f at step %d, final %.4f  -> %s"
          % (max(tE), tE.index(max(tE)), tE[-1], state))
    print("  steps with threat |E| > %.2f : %d of %d" % (BOUND, len(over), n))
    print("  last 500 steps threat |E| range: %.4f .. %.4f"
          % (min(tE[-500:]), max(tE[-500:])))

    print()
    print("  DECOMPOSITION: drive = tanh(event_term + pe_term)")
    print("  state = decay*prev + (1-decay)*drive,  decay=0.95")
    print()
    print("  %7s %12s %12s %12s %12s %12s %12s"
          % ("step", "event_abs", "pe_abs", "tanh_arg", "drive_pre",
             "pe_proj_b", "pe_proj_w"))
    for s in list(range(0, n, 250)) + [n - 1]:
        if s < n:
            r = d[s]
            print("  %7d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f"
                  % (s, r["threat_event_term_absmax"], r["threat_pe_term_absmax"],
                     r["threat_tanh_arg_absmax"], r["threat_drive_preclip_absmax"],
                     r["threat_pe_proj_b"], r["threat_pe_proj_w"]))

    # ---------- (b) applied gain ----------
    print()
    print("=" * 78)
    print("(b) DOES THE APPLIED GAIN PLATEAU?")
    print("=" * 78)
    ap = [r.get("applied_scale_absmax", float("nan")) for r in d]
    for s in list(range(0, n, 250)) + [n - 1]:
        if s < n:
            print("  step %5d  applied %.4f   raw max|1+fs| %.4f"
                  % (s, ap[s], d[s]["mod_scale_absmax"]))
    seg = ap[n // 2:]
    slope = (seg[-1] - seg[0]) / max(1, (len(seg) - 1))
    print("  applied gain: first %.4f  final %.4f  max %.4f" % (ap[0], ap[-1], max(ap)))
    print("  slope over second half: %+.5f per step  -> %s"
          % (slope, "STILL CLIMBING" if slope > 1e-4 else "flat / plateaued"))

    # ---------- Table A ----------
    print()
    print("=" * 78)
    print("TABLE A -- every 500 steps")
    print("=" * 78)
    print("%7s %14s %14s %12s %12s"
          % ("step", "applied gain", "raw max|1+fs|", "PE max", "max|E|"))
    for s in list(range(0, n, 500)) + [n - 1]:
        if s < n:
            r = d[s]
            print("%7d %14.4f %14.4f %12.3f %12.4f"
                  % (s, r.get("applied_scale_absmax", float("nan")),
                     r["mod_scale_absmax"], r["pe_max"], r["E_absmax"]))
    chk = [r["applied_gain_rms_check"] for r in d[1:]]
    print("  applied_gain_rms_check steps 1..%d: min %.4f max %.4f (expect 0.50)"
          % (n - 1, min(chk), max(chk)))
    print("  PE max over run: %.4f at step %d   final %.4f"
          % (max(r["pe_max"] for r in d),
             max(range(n), key=lambda i: d[i]["pe_max"]), d[-1]["pe_max"]))

    # ---------- Table B ----------
    print()
    print("=" * 78)
    print("TABLE B -- windowed max|E|, signed, all windows")
    print("=" * 78)
    prev = None
    for lo in range(0, n, W):
        hi = min(lo + W, n)
        if hi <= lo:
            continue
        mx = max(E[lo:hi])
        if prev is None:
            print("%4d-%4d %10.4f %12s %10s" % (lo, hi - 1, mx, "--", "first"))
        else:
            dl = mx - prev
            crit = "   <= criterion 2" if lo >= start else ""
            print("%4d-%4d %10.4f %+12.4f %10s%s"
                  % (lo, hi - 1, mx, dl, "RISE" if dl > 0 else "no", crit))
        prev = mx
    print("  criterion-2 windows (step >= %d): %d rising of %d"
          % (start, len(viol), len(wins)))

    # ---------- Table C ----------
    print()
    print("=" * 78)
    print("TABLE C -- per-dimension max|E| per 100-step window")
    print("=" * 78)
    for lo in range(0, n, W):
        hi = min(lo + W, n)
        if hi <= lo:
            continue
        best = [0.0] * 8
        for r in d[lo:hi]:
            pd = r.get("E_perdim") or []
            for k, v in enumerate(pd[:8]):
                if v > best[k]:
                    best[k] = v
        top = max(range(8), key=lambda k: best[k])
        star = " <<<" if lo >= start else ""
        print("  %4d-%4d  " % (lo, hi - 1) + " ".join("%.3f" % v for v in best)
              + "   max=%s (%.4f)%s" % (DIM_NAMES[top], best[top], star))


if __name__ == "__main__":
    main()
