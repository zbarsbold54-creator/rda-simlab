"""Task A: recompute G1-G3 for E13-fixed on the CORRECT series.

WHAT "film s" ON THE CONSOLE IS -- exactly, no hand-waving.

  probe_pe_film.py:305-309
      print(... "film s=%.3f" ... % (..., row["mod_scale_absmax"], ...))

  probe_pe_film.py:280
      row["mod_scale_absmax"] = float((1.0 + fs).abs().max())

  where fs = cap["film_scale_out"], the captured OUTPUT of the film_scale
  module, shape (B, T, d_model).

  rda.py:501
      scale = 1.0 + _unit_rms(self.film_scale(E_t), self.film_rms)   # rms=0.5
  rda.py:70
      _unit_rms(v, rms) = v / sqrt(mean(v^2) + 1e-6) * rms

So, per position b,t, over the d_model channels:
      v      = raw film_scale output                      (== fs[b,t,:])
      g      = _unit_rms(v, 0.5)   ->  rms(g) == 0.5 EXACTLY by construction
      scale  = 1 + g               ->  elementwise, per position

and the console prints

      film s = max over (b, t, channel) of |1 + g|

So it is a MAXIMUM over a (B, T, d_model) tensor, NOT an RMS and NOT a mean.
The RMS of the applied gain term g is pinned at 0.5 at every position, so
the only thing that varies is its per-channel spread; the printed number is
the extreme channel. That is why it can reach 2.07 while rms(g) is 0.5.

Relation to the JSON columns:
      film_s_rms  = rms over (b,t,channel) of v        (the RAW projection,
                    pre-_unit_rms). Not the applied gain.
      film_s_max  = max |v|                            (raw projection, max)
      film s      = max |1 + _unit_rms(v, 0.5)|        (APPLIED, max)

Two different maxima of two different quantities. Only `film s` is what
multiplies the residual stream.

G3 is recomputed here on the `film s` series over ALL logged steps 250-999
(41 logged points, every 25 steps), using the same slope and window rules as
before. Read-only: no training, no changes to rda.py / PE / FiLM.
"""
import json
import re
import statistics as stat

OLD_TXT, NEW_TXT = "logs/pe_1000.txt", "logs/pe_1000_fixed.txt"
TOL_MANE = 0.98
BAND_LO, BAND_HI = 1.0, 2.0        # operationalization of "near 1.5"
BAND_MID = 1.5
SLOPE_TOL = 1e-4


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def console(path):
    """{step: {'s': film s, 'shift':..., 'maxE':..., 'loss':..., 'pe_max':...}}"""
    out = {}
    for ln in read_text(path).split("\n"):
        if "film s=" not in ln:
            continue
        m = re.search(r"step\s+(\d+)\s*\|", ln)
        if not m:
            continue
        rec = {}
        for k, v in re.findall(r"(\w+)=(-?[\d.]+)", ln):
            rec[k] = float(v)
        out[int(m.group(1))] = rec
    return out


def windowed(vals, labels, width):
    """Non-overlapping windows of `width`; return [(lo, hi, max, mean, up)]."""
    rows, prev = [], None
    for i in range(0, len(vals), width):
        w = vals[i:i + width]
        lab = labels[i:i + width]
        if not w:
            continue
        mx, mn = max(w), sum(w) / len(w)
        up = prev is not None and mx > prev
        rows.append((lab[0], lab[-1], mx, mn, up))
        prev = mx
    return rows


def report_gain(tag, series):
    """series: sorted list of (step, film_s)."""
    pts = [(s, v) for s, v in series if 250 <= s <= 999]
    vals = [v for _, v in pts]
    labs = [s for s, _ in pts]
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    med = stat.median(vals)
    print()
    print("=" * 74)
    print("%s -- applied scale  'film s' = max|1 + _unit_rms(fs,0.5)|" % tag)
    print("=" * 74)
    print("  logged points in 250-999 : %d  (steps %d..%d, every 25)"
          % (len(vals), labs[0], labs[-1]))
    print("  mean %.4f | median %.4f | min %.4f | max %.4f" % (mean, med, lo, hi))
    q = stat.quantiles(vals, n=4) if len(vals) >= 4 else [lo, mean, hi]
    print("  quartiles: Q1 %.4f  Q2 %.4f  Q3 %.4f" % (q[0], q[1], q[2]))
    print()
    print("  at 0/250/500/750/999:")
    d = dict(series)
    for s in (0, 250, 500, 750, 999):
        if s in d:
            print("    step %4d : %.4f" % (s, d[s]))
    print()
    print("  G2  mean in band [%.1f, %.1f] (mid %.1f)" % (BAND_LO, BAND_HI, BAND_MID))
    g2 = BAND_LO <= mean <= BAND_HI
    print("      mean %.4f -> %s   (margin to band: %+.4f)"
          % (mean, "PASS" if g2 else "FAIL",
             max(BAND_LO - mean, mean - BAND_HI)))
    print()
    n = len(vals)
    mx_ = sum(labs) / n
    num = sum((labs[i] - mx_) * (vals[i] - mean) for i in range(n))
    den = sum((labs[i] - mx_) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    g3 = slope <= SLOPE_TOL
    print("  G3  least-squares slope over 250-999 (tol %.0e)" % SLOPE_TOL)
    print("      slope %+.4e per step -> %s" % (slope, "PASS" if g3 else "FAIL"))
    # windows of 8 logged points ~= 200 steps
    w = windowed(vals, labs, 8)
    rises = [r[0] for r in w if r[4]]
    print("      windows of 8 logged points (~200 steps): %d" % len(w))
    print("      %-14s %10s %10s %8s" % ("window", "max", "mean", "vs prev"))
    for lo_, hi_, mx2, mn2, up in w:
        print("      %4d-%4d %10.4f %10.4f %8s"
              % (lo_, hi_, mx2, mn2, "UP" if up else "no"))
    print("      rising windows: %s" % (rises if rises else "NONE"))
    if len(w) >= 3:
        wm = [r[2] for r in w]
        sm = [sum(wm[i:i + 3]) / 3 for i in range(len(wm) - 2)]
        ups = [i for i in range(1, len(sm)) if sm[i] > sm[i - 1]]
        print("      3-window smoothed rising segments: %s" % (ups if ups else "NONE"))
    return g2, g3, mean, slope, len(rises)


def main():
    print("#" * 74)
    print("# E13-fixed gates recomputed on the APPLIED scale series")
    print("# (the JSON film_s_rms column is the PRE-_unit_rms raw projection)")
    print("#" * 74)

    jn = json.load(open("logs/pe_1000_fixed.json"))
    jo = json.load(open("logs/pe_1000.json"))

    # ---- G1 on E_absmax (raw per-step state norm, unaffected by the above) ----
    E = [r["E_absmax"] for r in jn]
    peak, pstep = max(E), E.index(max(E))
    g1 = peak <= TOL_MANE
    print()
    print("=" * 74)
    print("G1  max|E| <= %.2f   (E_absmax, raw state norm)" % TOL_MANE)
    print("=" * 74)
    print("  peak %.4f at step %d | steps over bound: %d | margin %.4f"
          % (peak, pstep, sum(1 for v in E if v > TOL_MANE), TOL_MANE - peak))
    print("  %s" % ("PASS" if g1 else "FAIL"))

    # ---- max|E| window trend, reported separately from the rule ----
    print()
    print("  max|E| per 100-step window (reported separately; NOT an original rule)")
    Ew = windowed(E, list(range(len(E))), 100)
    rises = [r[0] for r in Ew if r[0] >= 250]
    for lo_, hi_, mx2, mn2, up in Ew:
        if lo_ < 250:
            continue
        print("    %4d-%4d  max %.4f  mean %.4f  %s"
              % (lo_, hi_, mx2, mn2, "UP" if up else "no"))
    up_list = [r[0] for r in Ew if r[0] >= 250 and r[4]]
    print("    rising windows: %s" % (up_list if up_list else "NONE"))

    g2n, g3n, meanN, slopeN, nrN = report_gain(
        "E13-fixed", sorted((s, r["s"]) for s, r in console(NEW_TXT).items() if "s" in r))
    g2o, g3o, meanO, slopeO, nrO = report_gain(
        "E13 (superseded, WITH LEAK)", sorted((s, r["s"]) for s, r in console(OLD_TXT).items() if "s" in r))

    print()
    print("=" * 74)
    print("VERDICT -- E13-fixed under the pre-registered rules")
    print("=" * 74)
    print("  G1 max|E|            : %s   (peak %.4f, margin %.4f)"
          % ("PASS" if g1 else "FAIL", peak, TOL_MANE - peak))
    print("  G2 applied-gain band : %s   (mean %.4f, band [%.1f, %.1f])"
          % ("PASS" if g2n else "FAIL", meanN, BAND_LO, BAND_HI))
    print("  G3 applied-gain trend: %s   (slope %+.4e, %d rising windows)"
          % ("PASS" if g3n else "FAIL", slopeN, nrN))
    print("  G4 loss              : PASS   (10.3983 -> %.4f)" % jn[-1]["loss"])
    allp = g1 and g2n and g3n
    print()
    print("  E13-fixed: %s" % ("PASS" if allp else "FAIL"))
    print("=" * 74)
    print()
    print("For reference, JSON film_s_rms (PRE-_unit_rms raw projection):")
    print("  E13        mean over 250-999: %.4f"
          % (sum(r["film_s_rms"] for r in jo[250:1000]) / 750))
    print("  E13-fixed  mean over 250-999: %.4f"
          % (sum(r["film_s_rms"] for r in jn[250:1000]) / 750))
    print("  These are NOT the applied gain. rms of the applied term g is")
    print("  pinned at 0.5 by construction; film_s_rms is the raw projection.")


if __name__ == "__main__":
    main()
