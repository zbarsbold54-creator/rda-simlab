"""Is `threat` AT the recurrence's fixed point, or merely near it?

CLAIM UNDER TEST (from the E13-fixed-3000 no-abort report): "with a drive
that close to 1 the state settles at the 0.95/0.05 equilibrium, which is
exactly where threat sits."

The equilibrium argument. The recurrence is

    E[t] = decay * E[t-1] + (1 - decay) * drive[t],      decay = 0.95

At a fixed point with drive held constant at d*:

    E* = decay*E* + (1 - decay)*d*
    E*(1 - decay) = (1 - decay)*d*
    E* = d*

So the fixed point is the drive value ITSELF -- not `drive*0.05`, and not
0.992 by some separate route. And it is the drive the state actually sees,
which is the position-AVERAGED drive, NOT the max.

That distinction matters. The report compared:

    drive_preclip_absmax = 0.9918   (max over batch/position)
    threat_E_absmax      = 0.9737   (max over batch/position)

and called the near-match an equilibrium. That is suggestive but not a
proof: the state is a leaky average, so its max should sit BELOW the
drive's max, and two different maxima are not directly comparable.

This script tests it properly, using only logged quantities:
  1. the fixed point implied by the MEAN drive vs the observed MEAN state
  2. the residual |E - drive| per step, which vanishes at a fixed point
  3. whether the state tracks the drive (co-movement) or sits stale

Read-only. No training.
"""
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "logs/pe_3000_fixed_noabort.json"
DECAY = 0.95


def main():
    d = json.load(open(PATH))
    n = len(d)
    dv = [r["threat_drive_preclip_mean"] for r in d]   # mean drive
    Ev = [r["threat_E_mean"] for r in d]                # mean state
    dmax = [r["threat_drive_preclip_absmax"] for r in d]
    Emax = [r["threat_E_absmax"] for r in d]

    print("=" * 74)
    print("FIXED-POINT TEST for the `threat` drive (decay = %.2f)" % DECAY)
    print("=" * 74)
    print("  A constant drive d* has fixed point E* = d* exactly:")
    print("      E* = decay*E* + (1-decay)*d*   =>   E* = d*")
    print("  So the correct comparison is MEAN state vs MEAN drive: the state")
    print("  is a leaky average over positions, not a max.")
    print()

    print("  %7s %14s %14s %14s %14s"
          % ("step", "mean drive", "mean state", "E*-drive", "E*/drive"))
    for s in (1000, 1500, 2000, 2500, n - 1):
        if s < n:
            print("  %7d %14.4f %14.4f %14.4f %13.4f"
                  % (s, dv[s], Ev[s], dv[s] - Ev[s], dv[s] / Ev[s]))
    print()

    seg = list(range(n // 2, n))
    res = [abs(Ev[i] - dv[i]) for i in seg]
    mres = sum(res) / len(res)
    mdrv = sum(dv[i] for i in seg) / len(seg)
    print("  |mean state - mean drive| over steps %d..%d:" % (n // 2, n - 1))
    print("      mean %.5f   max %.5f   final %.5f" % (mres, max(res), res[-1]))
    print("      mean drive over that window is %.4f, so the residual is"
          % mdrv)
    print("      %.2f%% of the drive level." % (100 * mres / mdrv))
    print()

    xs = [dv[i] for i in seg]
    ys = [Ev[i] for i in seg]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    dx = sum((v - mx) ** 2 for v in xs) ** 0.5
    dy = sum((v - my) ** 2 for v in ys) ** 0.5
    corr = num / (dx * dy) if dx and dy else float("nan")
    print("  CO-MOVEMENT -- is the state pinned, or lagging the drive?")
    print("      corr(mean drive, mean state) over the second half: %.4f" % corr)
    print("      ~1.0 => state tracks the drive, i.e. a genuine fixed point")
    print("       0.0 => state decoupled from / stale relative to the drive")
    print()

    k = min(100, n)
    imp = [DECAY * Ev[i - 1] + (1 - DECAY) * dv[i] for i in range(n - k, n)]
    obs = sum(Ev[n - k:]) / k
    print("  ONE-STEP RECURRENCE PREDICTION, last %d steps:" % k)
    print("      from decay*E[t-1] + (1-decay)*drive[t]:")
    print("        predicted mean E %.4f   observed %.4f   diff %+.4f"
          % (sum(imp) / k, obs, obs - sum(imp) / k))
    print()
    print("  max-based columns, for reference (NOT the equilibrium test):")
    print("      mean drive_max %.4f   mean state_max %.4f"
          % (sum(dmax[n // 2:]) / len(seg), sum(Emax[n // 2:]) / len(seg)))
    print("      state_max sits BELOW drive_max, as a leaky average should.")


if __name__ == "__main__":
    main()
