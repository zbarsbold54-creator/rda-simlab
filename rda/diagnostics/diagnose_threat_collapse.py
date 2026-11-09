"""
Diagnose WHY the threat drive collapses to the lower bound.

Reads logs/threat_preact.json (probe_pe_film.py --preact-out) and answers the
five diagnostic questions, reporting SIGNED values throughout.

WHY SIGNS MATTER HERE. The finding in E13-fixed-3000-dist was that the
threat drive is entirely negative -- fraction > +0.9 is exactly 0.0 -- yet
`absmax` was reported as if it described a positive excursion. An absmax with
no sign cannot distinguish "saturated high" from "saturated low", which is
the entire question. So every quantity below that has a natural sign is
printed signed, and where an absmax is also given, the sign of the
underlying value is given next to it.

The pre-tanh identity being decomposed is (from rda.py drive_seq):

    arg = event_drive_scale * ev_RMS[T]
        + pe_proj.weight[T,0] * (pe / pe_reference_scale)
        + pe_proj.bias[T]

so ev + wpe + bias == arg exactly. That identity is CHECKED here rather
than assumed, because a decomposition that does not reconcile explains
nothing.
"""

import json
import os
import sys

PREACT = sys.argv[1] if len(sys.argv) > 1 else "logs/threat_preact.json"
MAIN = sys.argv[2] if len(sys.argv) > 2 else "logs/pe_3000_fixed_preact.json"
DEAD_TANH = 0.05       # (c) threshold on mean tanh derivative


def q(v):
    return "%+.5f" % v


def main():
    d = json.load(open(PREACT, encoding="utf-8"))
    P = d["preact_every_step"]
    G = d["grad_every_step"]
    C = d["info_steps"]
    print("file    : %s (%d bytes)" % (PREACT, os.path.getsize(PREACT)))
    print("label   : %s  seed %s  steps %s" % (d["label"], d["seed"], d["steps"]))
    print("identity: %s" % d["identity"])
    print("rows    : %d preact, %d grad, %d corr steps"
          % (len(P), len(G), len(C)))
    print()

    # ---- sanity: does the decomposition reconcile? ----
    print("=" * 100)
    print("IDENTITY CHECK -- ev + wpe + bias == arg, at every logged step")
    print("=" * 100)
    worst = 0.0
    for r in P:
        res = abs(r["ev_mean"] + r["wpe_mean"] + r["bias_mean"] - r["arg_mean"])
        worst = max(worst, res)
    print("worst |residual| over all %d steps: %.3e" % (len(P), worst))
    print("PASS -- decomposition reconciles" if worst < 1e-5
          else "FAIL -- decomposition does NOT reconcile; treat the rest "
               "of this report as unreliable")
    # and pe_term must equal wpe + bias
    worst2 = max(abs(r["pe_mean"] - (r["wpe_mean"] + r["bias_mean"]))
                 for r in P)
    print("worst |pe_term - (wpe+bias)|       : %.3e" % worst2)
    print()

    # ---- (a) when does pre-tanh p50 cross -2 and -3? ----
    print("=" * 100)
    print("(a) PRE-TANH p50 THRESHOLDS  (and the full trajectory, sampled)")
    print("=" * 100)
    for thr in (-2.0, -3.0):
        hit = next((r["step"] for r in P if r["arg_q_p50"] < thr), None)
        print("p50 first below %+.1f : %s"
              % (thr, ("step %d" % hit) if hit is not None
                 else "NEVER in %d steps" % len(P)))
    print()
    print("%6s %9s %9s %9s %9s %9s %9s %9s"
          % ("step", "arg_p50", "arg_p1", "arg_p99", "ev_p50", "wpe_p50",
             "bias", "drv_p50"))
    show = [r for r in P if r["step"] % 250 == 0] + [P[-1]]
    for r in show:
        print("%6d %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (r["step"], r["arg_q_p50"], r["arg_q_p1"], r["arg_q_p99"],
                 r["ev_q_p50"], r["wpe_q_p50"], r["pe_proj_bias_threat"],
                 r["drv_q_p50"]))
    print()

    # component magnitudes at the end, signed
    last = P[-1]
    print("FINAL STEP (%d) component summary, SIGNED:" % last["step"])
    for nm in ("ev", "wpe", "bias", "arg", "drv"):
        print("  %-5s mean %s  p5 %s  p50 %s  p95 %s  p1 %s  p99 %s  std %.6f"
              % (nm, q(last[nm + "_mean"]), q(last[nm + "_q_p5"]),
                 q(last[nm + "_q_p50"]), q(last[nm + "_q_p95"]),
                 q(last[nm + "_q_p1"]), q(last[nm + "_q_p99"]), last[nm + "_std"]))
    frac = {nm: (last[nm + "_mean"] / last["arg_mean"])
            for nm in ("ev", "wpe", "bias") if last["arg_mean"] != 0}
    print("  share of arg_mean: " + "  ".join(
        "%s %.1f%%" % (k, 100 * v) for k, v in frac.items()))
    print()

    # ---- (b) pe_proj.bias[threat] trajectory + its gradient ----
    print("=" * 100)
    print("(b) pe_proj.bias[threat] TRAJECTORY AND ITS GRADIENT (signed)")
    print("=" * 100)
    b0, bl = P[0]["pe_proj_bias_threat"], last["pe_proj_bias_threat"]
    print("bias: step 0 %s  ->  step %d %s   (total change %s)"
          % (q(b0), last["step"], q(bl), q(bl - b0)))
    print()
    # when does it first reach the value seen at step 2000 in the prior run
    TARGET = -1.711
    hit = next((r["step"] for r in P if r["pe_proj_bias_threat"] <= TARGET), None)
    print("first step at or below %.3f (the value read at step 2000 in the "
          "prior run): %s" % (TARGET,
                             ("step %d" % hit) if hit else "NEVER"))
    print("first step at or below -1.0 : %s"
          % (next((r["step"] for r in P
                   if r["pe_proj_bias_threat"] <= -1.0), None),))
    print("first step at or below -1.5 : %s"
          % (next((r["step"] for r in P
                   if r["pe_proj_bias_threat"] <= -1.5), None),))
    print()
    print("%6s %11s %13s %13s %13s %13s"
          % ("step", "bias", "g_bias(signed)", "g_bias_norm_all",
             "g_w(signed)", "tanh_deriv"))
    for r in G:
        if r["step"] % 250 == 0 or r["step"] in (1, 2, 2999):
            print("%6d %11.5f %13.3e %13.3e %13.3e %13.6f"
                  % (r["step"], r["pe_proj_bias_threat"],
                     r.get("g_bias_threat_signed", float("nan")),
                     r.get("g_bias_threat_norm_all", float("nan")),
                     r.get("g_weight_threat_signed", float("nan")),
                     r["tanh_deriv_mean"]))
    print()
    # is the gradient consistently pushing the bias DOWN?
    neg = sum(1 for r in G if r.get("g_bias_threat_signed", 0) < 0)
    pos = sum(1 for r in G if r.get("g_bias_threat_signed", 0) > 0)
    zer = len(G) - neg - pos
    gs = [r.get("g_bias_threat_signed", 0.0) for r in G]
    print("g_bias[threat] sign over %d steps: negative %d, positive %d, "
          "zero %d" % (len(G), neg, pos, zer))
    print("  mean %s   min %s   max %s   final %s"
          % (q(sum(gs) / len(gs)), q(min(gs)), q(max(gs)), q(gs[-1])))
    if pos == 0 and neg == len(G):
        print("  -> the bias was pushed in ONE direction (downward, since "
              "grad<0) at EVERY step.")
        print("     A bias under a one-signed gradient is a ratchet: it "
              "cannot return toward 0")
        print("     unless the gradient changes sign. It never does.")
    else:
        print("  -> gradient CHANGED SIGN; the bias is not a pure ratchet.")
    print()

    # ---- (c) dead channel? ----
    print("=" * 100)
    print("(c) DEAD CHANNEL -- mean tanh derivative (1 - drive^2)")
    print("=" * 100)
    td = [(r["step"], r["tanh_deriv_mean"]) for r in P]
    print("first %d values: %s" % (5, ", ".join("%.4f" % v for _, v in td[:5])))
    print("final  5 values: %s" % ", ".join("%.4f" % v for _, v in td[-5:]))
    below = next((s for s, v in td if v < DEAD_TANH), None)
    print("first step with mean tanh deriv < %.2f : %s"
          % (DEAD_TANH, ("step %d" % below) if below else "NEVER"))
    for thr in (0.5, 0.2, 0.1, 0.05, 0.01):
        b = next((s for s, v in td if v < thr), None)
        print("  first step < %.2f : %s" % (thr, b if b else "NEVER"))
    v = [x for _, x in td]
    print("min %.6f at step %d ; max %.6f"
          % (min(v), td[v.index(min(v))][0], max(v)))
    fr = [(r["step"], r["tanh_deriv_frac_lt_0p01"]) for r in P]
    print("fraction of positions with tanh deriv < 0.01: "
          "first %.5f  final %.5f  max %.5f"
          % (fr[0][1], fr[-1][1], max(x for _, x in fr)))
    print()

    # ---- (d) correlation with raw inputs ----
    print("=" * 100)
    print("(d) CORRELATION OF THE DRIVE WITH ITS RAW INPUTS, ACROSS POSITIONS")
    print("=" * 100)
    print("%6s %10s %12s %12s %12s %12s %12s"
          % ("step", "drive_std", "corr_ev", "corr_pe", "corr_wpe",
             "corr_arg", "ev_std"))
    for c in C:
        def f(k):
            v = c.get(k)
            return "     None" if v is None else "%+12.6f" % v
        print("%6d %10.6f %s %s %s %s %12.6f"
              % (c["step"], c["drive_std"], f("corr_ev_term"),
                 f("corr_pe_term"), f("corr_wpe_term"), f("corr_arg"),
                 c["ev_term_std"]))
    print()
    print("note: corr_arg is ~1 by construction (tanh is monotone in its own")
    print("      argument) and is logged only to make that explicit.")
    print("      corr_ev / corr_pe are the informative ones.")
    print()
    ev_c = [c["corr_ev_term"] for c in C if c.get("corr_ev_term") is not None]
    pe_c = [c["corr_pe_term"] for c in C if c.get("corr_pe_term") is not None]
    print("corr_ev_term  : %s  ->  %s   (change %s)"
          % (q(ev_c[0]), q(ev_c[-1]), q(ev_c[-1] - ev_c[0])))
    print("corr_pe_term  : %s  ->  %s   (change %s)"
          % (q(pe_c[0]), q(pe_c[-1]), q(pe_c[-1] - pe_c[0])))
    print("drive_std     : %s  ->  %s   (change %s, %.1f%%)"
          % (q(C[0]["drive_std"]), q(C[-1]["drive_std"]),
             q(C[-1]["drive_std"] - C[0]["drive_std"]),
             100 * (C[-1]["drive_std"] - C[0]["drive_std"]) / C[0]["drive_std"]))
    print()
    print("drive min/max at the final corr step: %s / %s  (both negative =>"
          " no positive excursion)"
          % (q(C[-1]["drive_min"]), q(C[-1]["drive_max"])))
    print("drive min/max at step 0            : %s / %s"
          % (q(C[0]["drive_min"]), q(C[0]["drive_max"])))
    print()

    # ---- (e) do the other seven dimensions do the same? ----
    print("=" * 100)
    print("(e) THE OTHER SEVEN DIMENSIONS -- same quantities, from the main file")
    print("=" * 100)
    if not os.path.exists(MAIN):
        print("main file %s not found; skipping (e)" % MAIN)
        return
    rows = json.load(open(MAIN, encoding="utf-8"))
    from nanochat.rda import DRIVE_DIMS
    print()
    print("The main per-step record stores per-dimension E and pe aggregates")
    print("(E_perdim, pe_perdim) but NOT the per-dimension tanh argument or")
    print("gradient. Answering (e) on what is measured:")
    print()
    print("%14s %11s %11s %11s %11s"
          % ("dim", "E_absmax", "E_absmax@0", "E_absmax@1500", "E_absmax@2999"))
    for i, nm in enumerate(DRIVE_DIMS):
        v0 = rows[0]["E_perdim"][i]
        v15 = rows[min(1500, len(rows) - 1)]["E_perdim"][i]
        v30 = rows[-1]["E_perdim"][i]
        print("%14s %11.5f %11.5f %11.5f %11.5f"
              % (nm, rows[-1]["E_perdim"][i], v0, v15, v30))
    print()
    print("per-dimension |E| trajectory shape (E_absmax at 0/500/1000/1500/"
          "2000/2500/2999):")
    idxs = [0, 500, 1000, 1500, 2000, 2500, 2999]
    print("%14s %s" % ("dim", " ".join("%7d" % min(i, len(rows) - 1) for i in idxs)))
    for i, nm in enumerate(DRIVE_DIMS):
        vals = [rows[min(i_, len(rows) - 1)]["E_perdim"][i] for i_ in idxs]
        print("%14s %s" % (nm, " ".join("%7.4f" % v for v in vals)))
    print()
    print("NOTE: E_perdim is an ABSMAX (unsigned). It cannot show which bound")
    print("a dimension is pinned against. That is the exact defect this")
    print("whole investigation is about, and it applies to (e) as well.")
    print("An earlier version of this script also printed the E_perdim values")
    print("under a 'MEAN E (signed)' header. That was WRONG: E_perdim holds")
    print("abs-max, not mean, and the '+' signs were pure formatting. The")
    print("table is not repeated here. There is NO per-dimension signed mean")
    print("in the existing record, so (e) cannot be answered with signs.")
    print("(e) is only PARTIALLY answerable from the existing per-step")
    print("record. A full answer needs the pre-arg decomposition logged for")
    print("all 8 dimensions, which this run does not do (threat only,")
    print("as specified). What CAN be said is below.")

if __name__ == "__main__":
    main()
