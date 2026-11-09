"""
Settle the question E13-fixed-3000 left open: is the threat drive BIMODAL
(two separated modes) or UNIMODAL near -1 with a thin positive tail?

Reads logs/threat_drive_dist.json (written by probe_pe_film.py --dist-out)
and reports, per saved step:

  (a) a shape verdict from the 100-bin histogram, decided by an EXPLICIT
      valley test rather than by eye;
  (b) p1 / p50 / p99 and the two tail fractions against applied gain;
  (c) whether p50 approaches -0.98, and the step at which it would cross.

METHOD NOTE. The shape verdict is deliberately mechanical. A histogram of a
bimodal distribution has a genuine interior minimum with substantial mass on
BOTH sides of it. A unimodal distribution with a tail has a monotonically
falling shoulder off its single mode, so its only interior minimum is where
the tail runs out -- at which point there is (almost) no mass beyond.

For each interior bin i we compute:
    valley depth  = min(hist[i-1..i+1]) / hist[i]
    mass_below    = sum(hist[0..i-1])
    mass_above    = sum(hist[i+1..end])
A valley only counts as evidence of bimodality if
    valley depth < 0.5   AND   both mass_above and mass_below are >= 1%
of the total. Otherwise the best (deepest) interior valley is reported but
labelled NOT a mode separator.

Caveat carried into the output: 100 bins over [-1.2, 1.2] is 0.024 per bin.
Differences finer than that are not resolvable by this histogram, and the
verdict is conditional on binning. That is a limitation of the measurement,
not a fact about the drive.
"""

import json
import os
import sys

DIST = sys.argv[1] if len(sys.argv) > 1 else "logs/threat_drive_dist.json"
BOUND = -0.98          # the p50 target the user asked about
VALLEY_MAX = 0.5       # valley must dip below this fraction of the peak
SIDE_MIN_FRAC = 0.01   # each side of a real separator must hold >= 1% of mass


def valley_test(hist, n, total):
    """Return (is_bimodal, best_idx, best_depth, below, above).

    CORRECTED 2026-09-30. The first version of this function used
        min(h[i-1], h[i+1]) / h[i]
    as the valley depth. That metric is INVERTED: it is small at a PEAK,
    not at a valley, so it selected the mode itself and reported "BIMODAL"
    for a three-bin spike with no gap at all.

    The depth of a valley at i is the height of i RELATIVE TO ITS
    NEIGHBOURS, i.e. h[i]/min(h[i-1],h[i+1]) -- small means a real dip.
    Two further guards, both learned the hard way:
      * both neighbours must be non-empty, else the support edge (which is
        trivially adjacent to empty bins) registers as a valley;
      * the valley must be INTERIOR to the occupied support, for the
        same reason.
    """
    occ = [i for i, c in enumerate(hist) if c > 0]
    lo, hi = occ[0], occ[-1]
    best = None
    for i in range(lo + 1, hi):
        a, b = hist[i - 1], hist[i + 1]
        if a <= 0 or b <= 0:
            continue
        d = hist[i] / min(a, b)
        below = sum(hist[:i])
        above = sum(hist[i + 1:])
        bal = min(below, above)
        # deepest valley = smallest d; on ties prefer the more balanced one
        if best is None or (d, -bal) < (best[0], -best[1]):
            best = (d, bal, i, below, above)
    if best is None:
        # fewer than 3 occupied bins, or no interior bin with two
        # occupied neighbours: there is no gap anywhere in the support.
        return False, -1, float("nan"), 0, 0
    d, _, i, below, above = best
    bimodal = (d < VALLEY_MAX
               and below >= SIDE_MIN_FRAC * total
               and above >= SIDE_MIN_FRAC * total)
    return bimodal, i, d, below, above


def bucket(hist, n, lo, hi, edge):
    """Index of the bucket edge nearest `edge`."""
    w = (hi - lo) / n
    return max(0, min(n, int(round((edge - lo) / w))))


def main():
    d = json.load(open(DIST, encoding="utf-8"))
    print("file      : %s (%d bytes)" % (DIST, os.path.getsize(DIST)))
    print("label     : %s   seed %s   steps %s" % (d["label"], d["seed"], d["steps"]))
    print("dimension : %s" % d["dimension"])
    print("quantity  : %s" % d["quantity"])
    print("hist steps: %s" % d["histogram_steps"])
    print("per-step  : %d rows\n" % len(d["per_step"]))

    bystep = {h["step"]: h for h in d["histograms"]}
    rows = d["per_step"]

    # ---------- (a) + (b) at each histogram step ----------
    print("=" * 108)
    print("(a)/(b) SHAPE AND QUANTILES AT EACH SAVED STEP")
    print("=" * 108)
    hdr = ("step", "gain", "p1", "p50", "p99", "f>0.9", "f<-0.9",
           "mode_idx", "valley", "below%", "above%", "und", "ovf", "verdict")
    print("%5s %6s %7s %8s %7s %6s %6s %8s %6s %6s %6s %4s %4s %s"
          % hdr)
    for r in rows:
        s = r["step"]
        if s not in bystep:
            continue
        h = bystep[s]
        hist, n = h["hist"], h["n_bins"]
        total = sum(hist)
        bim, i, depth, below, above = valley_test(hist, n, total)
        w = (h["bin_hi"] - h["bin_lo"]) / n
        mi = hist.index(max(hist))
        print("%5d %6.3f %+7.4f %+8.4f %+7.4f %6.4f %6.4f %8.3f %6.3f %6.2f %6.2f %4d %4d %s"
              % (s, r["threat_applied_gain"],
                 r["threat_drv_q_p1"], r["threat_drv_q_p50"], r["threat_drv_q_p99"],
                 r["threat_drv_frac_gt_0p9"], r["threat_drv_frac_lt_0p9"],
                 h["bin_lo"] + (mi + 0.5) * w,
                 depth, 100.0 * below / total, 100.0 * above / total,
                 h["underflow_count"], h["overflow_count"],
                 "BIMODAL" if bim else "unimodal+tail"))
    print()
    print("mode_idx  = centre of the tallest bin (bin width %.4f)" %
          (1.2 * 2 / 100))
    print("valley    = h[i]/min(h[i-1],h[i+1]) at the deepest interior dip; "
          "<%.2f counts as a gap" % VALLEY_MAX)
    print("below/above = %% of all positions on each side of that minimum; "
          "a real separator needs >=%.0f%% on BOTH" % (SIDE_MIN_FRAC * 100))
    print("und/ovf   = positions outside [%.1f, %.1f]; tanh cannot reach "
          "them, so these should be 0" % (-1.2, 1.2))

    # ---------- (b) trend of the tail fractions with gain ----------
    print()
    print("=" * 108)
    print("(b) TAIL MOVEMENT vs APPLIED GAIN  (saved steps only)")
    print("=" * 108)
    print("%5s %8s %8s %8s %8s %9s %9s" %
          ("step", "gain", "p1", "p50", "p99", "f>0.9", "f<-0.9"))
    for r in rows:
        if r["step"] not in bystep:
            continue
        print("%5d %8.4f %+8.4f %+8.4f %+8.4f %9.5f %9.5f"
              % (r["step"], r["threat_applied_gain"],
                 r["threat_drv_q_p1"], r["threat_drv_q_p50"], r["threat_drv_q_p99"],
                 r["threat_drv_frac_gt_0p9"], r["threat_drv_frac_lt_0p9"]))

    print()
    print("ALL STEPS: min/max/range of each statistic over the whole run")
    print("-" * 70)
    for key, label in (("threat_drv_q_p1", "p1"),
                       ("threat_drv_q_p50", "p50"),
                       ("threat_drv_q_p95", "p95"),
                       ("threat_drv_q_p99", "p99"),
                       ("threat_drv_frac_gt_0p9", "f>0.9"),
                       ("threat_drv_frac_lt_0p9", "f<-0.9"),
                       ("threat_applied_gain", "gain")):
        v = [r[key] for r in rows]
        print("  %-6s min %+9.5f   max %+9.5f   delta %+9.5f   first %+9.5f   last %+9.5f"
              % (label, min(v), max(v), max(v) - min(v), v[0], v[-1]))

    # ---------- (c) does p50 approach -0.98? ----------
    print()
    print("=" * 108)
    print("(c) DOES p50 APPROACH %.2f ?" % BOUND)
    print("=" * 108)
    tail = [r for r in rows if r["step"] >= 1000]
    p50s = [r["threat_drv_q_p50"] for r in tail]
    print("p50 over steps >=1000: min %+.5f  max %+.5f  span %.5f"
          % (min(p50s), max(p50s), max(p50s) - min(p50s)))
    print("distance from p50 to %.2f : %.5f at step %d, %.5f at step %d"
          % (BOUND, tail[0]["threat_drv_q_p50"] - BOUND, tail[0]["step"],
             tail[-1]["threat_drv_q_p50"] - BOUND, tail[-1]["step"]))
    if abs(p50s[-1] - p50s[0]) < 1e-4:
        print("p50 is FLAT over that window (delta < 1e-4): no approach, "
              "no crossing to extrapolate.")
        print("VERDICT: p50 has NOT approached %.2f. It is stationary at "
              "about %+.3f." % (BOUND, p50s[-1]))
        print("         The max|E| ~ 0.98 reported elsewhere is therefore "
              "NOT the bulk of the")
        print("         distribution sitting at the bound; it comes from a "
              "minority of positions,")
        print("         which is exactly what the p99 column shows.")
    else:
        slope = (p50s[-1] - p50s[0]) / (tail[-1]["step"] - tail[0]["step"])
        need = (BOUND - p50s[-1]) / slope if slope else float("inf")
        print("least-squares-free endpoint slope: %.3e per step" % slope)
        print("steps to reach %.2f at that slope: %.0f (=> step %.0f)"
              % (BOUND, need, tail[-1]["step"] + need))
        print("NOTE: this is a LINEAR EXTRAPOLATION FROM TWO ENDPOINTS. "
              "It is an inference,")
        print("      not a measurement. The p50 over steps >=1000 is the "
              "quantity to check")
        print("      against it, and the histogram bin width (0.024) is "
              "coarser than the")
        print("      total movement quoted above.")

    # ---------- (a) detail: the actual histograms ----------
    print()
    print("=" * 108)
    print("(a) HISTOGRAM DETAIL -- coarse 20-bin re-bucketing of the 100 bins")
    print("=" * 108)
    for s in sorted(bystep):
        h = bystep[s]
        hist, n = h["hist"], h["n_bins"]
        g = 5
        coarse = [sum(hist[i:i + g]) for i in range(0, n, g)]
        peak = max(coarse) or 1
        print("step %4d  gain %.3f  (bar = %.1f%% of peak)" % (s, 0, 100.0 / peak))
        for j, c in enumerate(coarse):
            lo = -1.2 + j * (2.4 / len(coarse))
            print("  [%+6.3f,%+6.3f) %8d %s"
                  % (lo, lo + 2.4 / len(coarse), c, "#" * int(60 * c / peak)))
        print("  underflow %d  overflow %d  total %d\n"
              % (h["underflow_count"], h["overflow_count"], sum(hist)))


if __name__ == "__main__":
    main()
