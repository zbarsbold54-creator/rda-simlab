"""Apply the SAME E13 pass rules to logs/pe_1000_fixed.json.

RULES CARRIED OVER UNCHANGED from the original E13 gate (see
research/EXPERIMENT_LOG.md E13), so the two runs are judged identically:

  G1  max|E| <= 0.98 for every recorded step   (the saturation abort bound)
  G2  FiLM gain mean over 250-999 is near 1.5   (the healthy modulated regime)
  G3  FiLM gain shows NO upward trend over 250-999 (no self-excitation)
  G4  loss decreases and stays finite (no collapse, no divergence)

Reported alongside: max|E| overall, FiLM gain at 0/250/500/750/999, the
least-squares slope of gain over 250-999, PE max, and loss at the same
checkpoints. Windowed monotonicity is checked the same way as before
(100-step windows, and a 3-window smoothed trend) because raw step-to-step
noise is expected and means nothing.

Run:  python diagnostics/eval_e13_fixed.py
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/pe_1000_fixed.json"
TOL_MANE = 0.98
GAIN_NEAR = 1.5
GAIN_TOL = 0.5           # "near 1.5"
SLOPE_TOL = 1e-4        # per step; above this counts as upward trend

d = json.load(open(PATH))
print("=" * 72)
print("E13-fixed GATE  --  %s" % PATH)
print("=" * 72)
print("steps recorded: %d" % len(d))

E = [r["E_absmax"] for r in d]
gain = [r["film_s_rms"] for r in d]
loss = [r["loss"] for r in d]
pe = [r["pe_max"] for r in d]

# ---- G1: saturation bound ---- #
peak = max(E)
peak_step = E.index(peak)
g1 = peak <= TOL_MANE
print("")
print("G1  max|E| <= %.2f" % TOL_MANE)
print("      peak max|E| = %.4f at step %d" % (peak, peak_step))
print("      steps over bound: %d" % sum(1 for v in E if v > TOL_MANE))
print("      %s" % ("PASS" if g1 else "FAIL"))

# ---- G2/G3: FiLM gain level and trend over 250-999 ---- #
seg = list(range(250, min(1000, len(gain))))
sg = [gain[i] for i in seg]
mean_gain = sum(sg) / len(sg)
g2 = abs(mean_gain - GAIN_NEAR) <= GAIN_TOL
n = len(seg)
xs = seg
mx = sum(xs) / n
my = mean_gain
num = sum((xs[i] - mx) * (sg[i] - my) for i in range(n))
den = sum((xs[i] - mx) ** 2 for i in range(n))
slope = num / den if den else 0.0
g3 = slope <= SLOPE_TOL
print("")
print("G2  FiLM gain mean over 250-999 near %.1f (+/- %.1f)" % (GAIN_NEAR, GAIN_TOL))
print("      mean gain = %.4f   %s" % (mean_gain, "PASS" if g2 else "FAIL"))
print("")
print("G3  FiLM gain has no upward trend over 250-999 (slope <= %.0e)" % SLOPE_TOL)
print("      least-squares slope = %+.3e per step   %s" % (slope, "PASS" if g3 else "FAIL"))
print("      min/max gain in window: %.4f / %.4f" % (min(sg), max(sg)))

# ---- windowed monotonicity, same method as the original gate ---- #
print("")
print("    max|E| per 100-step window, 250-999")
print("    %-12s %10s %10s %8s" % ("window", "max|E|", "mean|E|", "vs prev"))
prev = None
rises = []
wmax = []
for lo in range(250, 1000, 100):
    hi = min(lo + 100, len(E))
    if lo >= len(E):
        break
    w = E[lo:hi]
    mx_ = max(w)
    wmax.append(mx_)
    up = prev is not None and mx_ > prev
    if up:
        rises.append(lo)
    print("    %4d-%4d %10.4f %10.4f %8s"
          % (lo, hi - 1, mx_, sum(w) / len(w), "UP" if up else "no"))
    prev = mx_
print("    windows exceeding previous: %s" % (rises if rises else "NONE"))

if len(wmax) >= 3:
    sm = [sum(wmax[i:i + 3]) / 3 for i in range(len(wmax) - 2)]
    ups = [i for i in range(1, len(sm)) if sm[i] > sm[i - 1]]
    print("    3-window smoothed rising segments: %s" % (ups if ups else "NONE"))

# ---- G4: loss ---- #
g4 = all(v == v and v < 1e4 for v in loss) and loss[-1] < loss[0]
print("")
print("G4  loss finite and decreasing")
print("      loss step 0 = %.4f -> step %d = %.4f   %s"
      % (loss[0], len(loss) - 1, loss[-1], "PASS" if g4 else "FAIL"))

# ---- requested checkpoint table ---- #
print("")
print("    step      loss    max|E|   film gain    PE max")
for s in (0, 250, 500, 750, 999):
    if s < len(d):
        print("    %5d %10.4f %9.4f %10.4f %9.4f"
              % (s, loss[s], E[s], gain[s], pe[s]))
print("    PE max over run: %.4f   (step %d)" % (max(pe), pe.index(max(pe))))

print("")
print("=" * 72)
allp = g1 and g2 and g3 and g4
print("E13-fixed: %s" % ("PASS" if allp else "FAIL"))
if not allp:
    for nm, ok in (("G1 max|E|", g1), ("G2 gain level", g2),
                   ("G3 gain trend", g3), ("G4 loss", g4)):
        if not ok:
            print("  failed: %s" % nm)
print("=" * 72)
sys.exit(0 if allp else 1)
