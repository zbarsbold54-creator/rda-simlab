"""
All-8-dimension diagnosis of the RDA drive collapse.

Reads logs/affect_preact_all.json and answers items 5 and 6-9 of the
all-dimensions prompt, for every dimension, SIGNED.

SIGN DISCIPLINE. Every extreme is reported with its sign, and every absmax
is printed next to the signed min and max it came from. This is not
pedantry: the two retractions in this investigation both came from reading
an unsigned aggregate as though it had a sign ("max|E| ~ 0.98 comes from a
POSITIVE TAIL" when the measured fraction above +0.9 was exactly 0.0).

HEALTH DEFINITION (item 9, as specified by the user):
    healthy  <=>  R_d = p5 - p95 range of the DRIVE >= 0.3
                 AND no bias offset above 50% of the arg
"bias offset above 50% of the arg" is evaluated as
    |bias| / |arg_p50|  when arg_p50 is not ~0
and is reported per dimension so the ratio can be inspected rather than
taken on trust. When arg_p50 is near zero the ratio is meaningless (a small
arg makes any bias look dominant) so it is reported as undefined rather
than silently large.

The thresholds 0.3 and 50% are the USER'S, not calibrated here. They are
applied as given and flagged as uncalibrated at the point of use.
"""

import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "logs/affect_preact_all.json"
RAIL = 0.9          # |drive| beyond this is "against a rail"
HEALTHY_RANGE = 0.3
HEALTHY_BIAS_SHARE = 0.5
SMALL_ARG = 0.25    # below this, a bias/arg ratio is not meaningful


def s(v, w=9, p=4):
    if v is None:
        return " " * (w - 1) + "-"
    return ("%+" + str(w) + "." + str(p) + "f") % v


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    R = d["per_step"]
    D = d["dims"]
    V = {r["step"]: r for r in d["variance_steps"]}
    print("file   : %s (%d bytes)" % (SRC, os.path.getsize(SRC)))
    print("label  : %s  seed %s  steps %s" % (d["label"], d["seed"], d["steps"]))
    print("rows   : %d per-step, variance at %s"
          % (len(R), sorted(V)))
    print()
    print("IDENTITY CHECK (arg = ev + wpe + bias), all 8 dims, all steps")
    w = d["identity_worst"]
    print("  worst residual %.3e at step %s dim %s ; tolerance %.0e -> %s"
          % (w["residual"], w["step"], w["dim"], d["identity_tolerance"],
             "PASS" if d["identity_ok"] else "*** FAIL ***"))
    if not d["identity_ok"]:
        print("  STOP: the decomposition does not reconcile; every number "
              "below is void.")
        return
    print()
    print("CONSUMERS OF THE DRIVE / E STATE (read off nanochat/rda.py)")
    for k in ("drive_direct", "E_state"):
        for line in d["consumers"][k]:
            print("  [%s] %s" % (k, line))
    print("  note: %s" % d["consumers"]["feedback_note"])
    print()

    last = R[-1]

    # ---------- (a) arg p50 crossings, SIGNED ----------
    print("=" * 104)
    print("(a) STEP AT WHICH arg p50 FALLS BELOW -2 OR RISES ABOVE +2  (signed)")
    print("=" * 104)
    print("%18s %12s %12s %12s" % ("dim", "first<-2", "first>+2", "arg_p50@end"))
    for n in D:
        lo = next((r["step"] for r in R if r[n + "_arg_p50"] < -2.0), None)
        hi = next((r["step"] for r in R if r[n + "_arg_p50"] > 2.0), None)
        print("%18s %12s %12s %12s"
              % (n, lo if lo else "never", hi if hi else "never",
                 s(last[n + "_arg_p50"])))
    print()

    # ---------- (b) bias trajectory + gradient sign ----------
    print("=" * 104)
    print("(b) pe_proj.bias[d] TRAJECTORY AND SIGNED GRADIENT SIGN OVER TIME")
    print("=" * 104)
    print("%18s %10s %10s %10s %8s %8s %8s %9s"
          % ("dim", "start", "end", "delta", "g<0", "g>0", "g=0", "g_end"))
    for n in D:
        gs = [r[n + "_g_bias_signed"] for r in R if n + "_g_bias_signed" in r]
        neg = sum(1 for g in gs if g < 0)
        pos = sum(1 for g in gs if g > 0)
        zer = len(gs) - neg - pos
        print("%18s %10s %10s %10s %8d %8d %8d %9s"
              % (n, s(R[0][n + "_bias"], 10), s(last[n + "_bias"], 10),
                 s(last[n + "_bias"] - R[0][n + "_bias"], 10),
                 neg, pos, zer, s(gs[-1], 9, 6)))
    print()
    print("gradient dominance (which sign the bias is mostly being pushed):")
    for n in D:
        gs = [r[n + "_g_bias_signed"] for r in R if n + "_g_bias_signed" in r]
        neg = sum(1 for g in gs if g < 0)
        pos = sum(1 for g in gs if g > 0)
        tot = neg + pos
        frac = (max(neg, pos) / tot) if tot else 0.0
        dom = "DOWN(-)" if neg > pos else "UP(+)"
        mixed = "MIXED" if frac < 0.6 else dom
        print("  %-18s %-6s  dominant fraction %.3f  %s"
              % (n, dom, frac,
                 "(one-signed ratchet)" if frac >= 0.9 else "(two-sided)"))
    print()

    # ---------- (c) tanh derivative below 0.05 ----------
    print("=" * 104)
    print("(c) STEP AT WHICH mean tanh DERIVATIVE FALLS BELOW 0.05")
    print("=" * 104)
    print("%18s %12s %12s %10s %10s"
          % ("dim", "first<0.05", "first<0.10", "min", "min@step"))
    for n in D:
        td = [(r["step"], r[n + "_tanh_deriv_mean"]) for r in R]
        b05 = next((st for st, v in td if v < 0.05), None)
        b10 = next((st for st, v in td if v < 0.10), None)
        mn = min(td, key=lambda t: t[1])
        print("%18s %12s %12s %10.4f %10d"
              % (n, b05 if b05 else "NEVER", b10 if b10 else "never",
                 mn[1], mn[0]))
    print()

    # ---------- (d) which bound the drive p50 sits against ----------
    print("=" * 104)
    print("(d) WHICH BOUND THE DRIVE p50 SITS AGAINST  (signed; steps "
          "0/500/1000/1500/2000/2999)")
    print("=" * 104)
    show = (0, 500, 1000, 1500, 2000, 2999)
    for n in D:
        print("  %s" % n)
        print("    %6s %10s %10s %10s %9s %9s  %s"
              % ("step", "drv_p5", "drv_p50", "drv_p95", "min", "max",
                 "verdict"))
        for st in show:
            r = next((x for x in R if x["step"] == st), None)
            if r is None:
                continue
            p50 = r[n + "_drv_p50"]
            p5, p95 = r[n + "_drv_p5"], r[n + "_drv_p95"]
            if p50 <= -RAIL:
                v = "LOWER RAIL (-)"
            elif p50 >= RAIL:
                v = "UPPER RAIL (+)"
            elif abs(p50) < 0.25:
                v = "centre (no rail)"
            else:
                v = "off-rail"
            print("    %6d %10s %10s %10s %9s %9s  %s"
                  % (st, s(p5), s(p50), s(p95),
                     s(r[n + "_drv_min"]), s(r[n + "_drv_max"]), v))
    print()

    # ---------- (6) pe weight value + zero crossing ----------
    print("=" * 104)
    print("(6) pe_proj.weight[d,0] PER DIMENSION, AND ITS ZERO-CROSSING STEP")
    print("=" * 104)
    for n in D:
        vs = [(r["step"], r[n + "_pe_weight"]) for r in R]
        cross = None
        for i in range(1, len(vs)):
            a, b = vs[i - 1][1], vs[i][1]
            if (a > 0) != (b > 0) and a != 0 and b != 0:
                cross = vs[i][0]
                break
        nz = next((st for st, v in vs if v == 0.0), None)
        print("  %-18s start %s  end %s  zero-crossing %s%s"
              % (n, s(vs[0][1]), s(vs[-1][1]),
                 cross if cross else "never",
                 "  (exact 0 at step %d)" % nz if nz else ""))
    print()

    # ---------- (7) variance shares ----------
    print("=" * 104)
    print("(7) ACROSS-POSITION VARIANCE OF EACH PRE-TANH TERM, AND COVARIANCE")
    print("=" * 104)
    for st in sorted(V):
        r = V[st]
        print("  step %d" % st)
        print("    %18s %11s %11s %12s %9s %9s"
              % ("dim", "ev_var", "wpe_var", "cov", "ev_share", "wpe_share"))
        for n in D:
            es = r[n + "_ev_var_share"]
            ws = r[n + "_wpe_var_share"]
            print("    %18s %11.3e %11.3e %12.3e %9s %9s"
                  % (n, r[n + "_ev_var"], r[n + "_wpe_var"], r[n + "_cov"],
                     ("%.3f" % es) if es is not None else "  undef",
                     ("%.3f" % ws) if ws is not None else "  undef"))
    print()
    print("variance share summary at the final sampled step (%d):"
          % max(V))
    r = V[max(V)]
    print("  %18s %10s %10s %s" % ("dim", "ev_share", "wpe_share", "dominant"))
    for n in D:
        es, ws = r[n + "_ev_var_share"], r[n + "_wpe_var_share"]
        if es is None or ws is None:
            print("  %18s %10s %10s %s"
                  % (n, "undef", "undef", "both terms constant"))
            continue
        dom = "event" if es > ws else "PE-weight"
        print("  %18s %10.3f %10.3f %s" % (n, es, ws, dom))
    print()

    # ---------- (8) dL/d(drive) ----------
    print("=" * 104)
    print("(8) dL/d(drive_d) OVER POSITIONS -- mean (signed) and std")
    print("=" * 104)
    print("%18s %13s %13s %13s %13s"
          % ("dim", "mean@0", "mean@end", "std@end", "|mean|@end/max"))
    for n in D:
        m0 = R[0].get(n + "_dLddrive_mean")
        me = last.get(n + "_dLddrive_mean")
        se = last.get(n + "_dLddrive_std")
        mx = last.get(n + "_dLddrive_absmax")
        ratio = (abs(me) / mx) if (me is not None and mx) else None
        print("%18s %13s %13s %13s %13s"
              % (n, s(m0, 13, 6), s(me, 13, 6), s(se, 13, 6),
                 ("%.4f" % ratio) if ratio is not None else "-"))
    print()
    print("a |mean|/max ratio near 0 would mean the gradient CANCELS across")
    print("positions (the dimension is pushed both ways and nets out);")
    print("a ratio near 1 means the gradient is coherent across positions.")
    print()

    # ---------- (9) per-dimension pattern + health ----------
    print("=" * 104)
    print("(9) PATTERN SUMMARY AND HEALTH")
    print("=" * 104)
    print("health (as specified): R_d = p95-p5 of the DRIVE >= %.1f AND no"
          % HEALTHY_RANGE)
    print("bias offset above %.0f%% of the arg.  NOT CALIBRATED."
          % (100 * HEALTHY_BIAS_SHARE))
    print()
    print("%18s %8s %10s %10s %7s %7s %7s %7s %6s"
          % ("dim", "R_d", "bias", "arg_p50", "bias%", "(i)", "(ii)",
             "(iii)", "(iv)"))
    print("%18s %8s %10s %10s %7s %7s %7s %7s %6s"
          % ("", "range", "end", "end", "of arg", "bigB",
             "wCross", "vShare", "td<.10"))
    summary = []
    for n in D:
        rng = last[n + "_drv_p95"] - last[n + "_drv_p5"]
        bias = abs(last[n + "_bias"])
        arg = abs(last[n + "_arg_p50"])
        share = (bias / arg) if arg >= SMALL_ARG else None
        # (i) bias converged to a large-magnitude offset
        bigB = abs(last[n + "_bias"]) >= 0.5
        # (ii) weight sign crossing
        vs = [r[n + "_pe_weight"] for r in R]
        wcross = any((vs[i - 1] > 0) != (vs[i] > 0)
                     for i in range(1, len(vs)) if vs[i] != 0 and vs[i - 1] != 0)
        # (iii) variance share dominated by one term
        vr = V[max(V)]
        es, ws = vr[n + "_ev_var_share"], vr[n + "_wpe_var_share"]
        vshare = (es is not None and (es > 0.9 or ws > 0.9))
        # (iv) mean tanh derivative below 0.10
        td = [r[n + "_tanh_deriv_mean"] for r in R]
        tdl = min(td) < 0.10
        healthy = (rng >= HEALTHY_RANGE
                   and (share is None or share <= HEALTHY_BIAS_SHARE))
        summary.append((n, rng, healthy, share, bigB, wcross, vshare, tdl))
        print("%18s %8.4f %10s %10s %7s %7s %7s %7s %6s"
              % (n, rng, s(last[n + "_bias"], 10), s(last[n + "_arg_p50"], 10),
                 ("%.2f" % share) if share is not None else "undef",
                 "Y" if bigB else ".", "Y" if wcross else ".",
                 "Y" if vshare else ".", "Y" if tdl else "."))
    print()
    print("HEALTHY (R_d >= %.1f and bias share <= %.0f%%):" %
          (HEALTHY_RANGE, 100 * HEALTHY_BIAS_SHARE))
    h = [n for n, rng, ok, *_ in summary if ok]
    u = [n for n, rng, ok, *_ in summary if not ok]
    print("  healthy     : %s" % (", ".join(h) if h else "NONE"))
    print("  not healthy : %s" % (", ".join(u) if u else "none"))
    print()
    print("Pattern tally:")
    print("  (i)   bias converged to |bias| >= 0.5      : %s"
          % ", ".join(n for n, _, _, _, b, _, _, _ in summary if b))
    print("  (ii)  pe weight crossed zero              : %s"
          % ", ".join(n for n, _, _, _, _, w, _, _ in summary if w))
    print("  (iii) one term owns >90% of the variance : %s"
          % ", ".join(n for n, _, _, _, _, _, v, _ in summary if v))
    print("  (iv)  mean tanh deriv ever below 0.10     : %s"
          % ", ".join(n for n, _, _, _, _, _, _, t in summary if t))
    print("  NONE of the above                          : %s"
          % ", ".join(n for n, _, _, _, b, w, v, t in summary
                      if not (b or w or v or t)))


if __name__ == "__main__":
    main()
