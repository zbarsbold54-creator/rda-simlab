"""
Is the threat channel a CONSTANT OFFSET into FiLM?

Reads logs/film_weights.json and answers (a), (b), (c).

METHOD. FiLM is linear in E:
    film_scale(E)[:, c] = sum_d W_scale[c, d] * E[:, d]
so the contribution of dimension d to output channel c is exactly
    W[:, d] * E[:, d]
-- a constant vector times a per-position scalar. Its across-position mean
is therefore the DC (offset) part and its across-position std the varying
part, with no modelling assumption. The ratio mean/std is a direct
offset-to-signal measure for that dimension's share of the FiLM output.

SIGNED THROUGHOUT. A negative offset is still an offset, so the signed mean
and the signed ratio are both reported, with |ratio| alongside. Collapsing
these to an absmax would lose exactly the distinction the question is about.

TWO COMPARISONS ARE KEPT SEPARATE because they answer different questions:
  * per-dimension ratio  -- is THIS dimension an offset into FiLM?
  * cross-dimension column norm -- is threat's column LARGER than the rest?
A dimension can have a small column norm and still be mostly offset, and a
large column norm while varying. The two are reported side by side rather
than combined into one score.
"""

import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "logs/film_weights.json"


def s(v, w=10, p=4):
    if v is None:
        return " " * (w - 1) + "-"
    return ("%+" + str(w) + "." + str(p) + "f") % v


def ratio_str(v):
    return "  undef" if v is None else "%+.3f" % v


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    R = {r["step"]: r for r in d["rows"]}
    D = d["dims"]
    steps = sorted(R)
    print("file   : %s (%d bytes)" % (SRC, os.path.getsize(SRC)))
    print("label  : %s  seed %s  steps %s" % (d["label"], d["seed"], d["steps"]))
    print("E      : %s" % d["e_tensor"])
    print("column : %s" % d["column_convention"])
    print("rows   : steps %s" % steps)
    print()
    last = R[steps[-1]]

    # ---------- (1)+(2) column norms and E stats ----------
    print("=" * 112)
    print("(1)+(2) FILM COLUMN NORMS AND PER-CHANNEL E STATISTICS, SIGNED")
    print("=" * 112)
    for st in steps:
        r = R[st]
        print("  step %d   (film_rms=%.2f)" % (st, r["film_rms"]))
        print("    %18s %10s %10s %10s %10s %10s %10s"
              % ("dim", "sc_col_nrm", "sh_col_nrm", "E_mean", "E_std",
                 "E_min", "E_max"))
        for n in D:
            print("    %18s %10.5f %10.5f %10s %10.5f %10s %10s"
                  % (n, r[n + "_scale_col_norm"], r[n + "_shift_col_norm"],
                     s(r[n + "_E_mean"]), r[n + "_E_std"],
                     s(r[n + "_E_min"]), s(r[n + "_E_max"])))
        print()

    # ---------- (a) does threat's column grow relative to the others? ----------
    print("=" * 112)
    print("(a) DOES THE THREAT COLUMN NORM GROW RELATIVE TO THE OTHER SEVEN?")
    print("=" * 112)
    for tag in ("scale", "shift"):
        print("  %s column norm" % tag)
        print("    %18s %s" % ("dim", " ".join("%9d" % st for st in steps)))
        for n in D:
            vals = [R[st][n + "_%s_col_norm" % tag] for st in steps]
            print("    %18s %s" % (n, " ".join("%9.4f" % v for v in vals)))
        # threat's share of the total, and its rank.
        # At step 0 every column norm is exactly 0 (FiLM is zero-init), so
        # the share and the threat/other7 ratio are 0/0. They are printed as
        # "n/a" rather than being allowed to raise or to report a fake 0.
        def _share(st):
            tot = sum(R[st]["%s_%s_col_norm" % (n, tag)] for n in D)
            if tot <= 1e-12:
                return "     n/a"
            return "%7.1f%%" % (100.0 * R[st]["threat_%s_col_norm" % tag] / tot)

        def _rel(st):
            oth = sum(R[st]["%s_%s_col_norm" % (n, tag)]
                      for n in D if n != "threat") / 7.0
            if oth <= 1e-12:
                return "     n/a"
            return "%9.4f" % (R[st]["threat_%s_col_norm" % tag] / oth)

        print("    %18s %s" % ("threat share", " ".join(_share(st) for st in steps)))
        print("    %18s %s"
              % ("threat rank",
                 " ".join("%9s" % (
                     "n/a" if sum(R[st]["%s_%s_col_norm" % (n, tag)]
                                 for n in D) <= 1e-12
                     else (1 + sum(1 for n in D
                                   if R[st][n + "_%s_col_norm" % tag]
                                   > R[st]["threat_%s_col_norm" % tag])))
                     for st in steps)))
        print("    %18s %s"
              % ("mean of other 7",
                 " ".join("%9.4f" % (
                     sum(R[st]["%s_%s_col_norm" % (n, tag)]
                         for n in D if n != "threat") / 7.0)
                     for st in steps)))
        print("    %18s %s" % ("threat / other7", " ".join(_rel(st) for st in steps)))
        print()

    # ---------- (3)+(b) offset vs varying ----------
    print("=" * 112)
    print("(3)+(b) OFFSET vs VARYING CONTRIBUTION TO THE FiLM OUTPUT, SIGNED")
    print("=" * 112)
    for st in steps:
        r = R[st]
        print("  step %d" % st)
        print("    %18s %13s %13s %11s %13s %13s %11s"
              % ("dim", "scale_mean", "scale_std", "scale o/v",
                 "shift_mean", "shift_std", "shift o/v"))
        for n in D:
            print("    %18s %13s %13.3e %11s %13s %13.3e %11s"
                  % (n, s(r[n + "_scale_contrib_mean"], 13, 6),
                     r[n + "_scale_contrib_std"],
                     ratio_str(r[n + "_scale_off_over_var"]),
                     s(r[n + "_shift_contrib_mean"], 13, 6),
                     r[n + "_shift_contrib_std"],
                     ratio_str(r[n + "_shift_off_over_var"])))
        print("    combined (scale+shift), which is what modulate() adds:")
        print("    %18s %13s %13s %11s %13s"
              % ("dim", "contrib_mean", "contrib_std", "o/v", "|o/v|"))
        for n in D:
            v = r[n + "_film_off_over_var"]
            print("    %18s %13s %13.3e %11s %13s"
                  % (n, s(r[n + "_film_contrib_mean"], 13, 6),
                     r[n + "_film_contrib_std"], ratio_str(v),
                     ("%.3f" % abs(v)) if v is not None else "  undef"))
        print()

    # ranked by |offset/varying| at the final step
    print("  RANKED BY |offset/varying| AT STEP %d (combined)" % steps[-1])
    rank = sorted(D, key=lambda n: -abs(last[n + "_film_off_over_var"] or 0))
    print("    %3s %18s %10s %14s %14s %10s"
          % ("#", "dim", "|o/v|", "E_mean", "E_std", "col_nrm"))
    for i, n in enumerate(rank, 1):
        v = last[n + "_film_off_over_var"]
        print("    %3d %18s %10s %14s %14.3e %10.4f"
              % (i, n, ("%.3f" % abs(v)) if v is not None else "undef",
                 s(last[n + "_E_mean"]), last[n + "_E_std"],
                 last[n + "_scale_col_norm"]))
    print()

    # ---------- (c) does col norm grow as E std shrinks? ----------
    print("=" * 112)
    print("(c) DOES THE COLUMN NORM GROW AS E_std SHRINKS?")
    print("=" * 112)
    for n in D:
        e0, e1 = R[steps[0]][n + "_E_std"], last[n + "_E_std"]
        c0, c1 = R[steps[0]][n + "_scale_col_norm"], last[n + "_scale_col_norm"]
        m0, m1 = R[steps[0]][n + "_E_mean"], last[n + "_E_mean"]
        # the product |col| * E_std is the magnitude of the VARYING part,
        # so it separates "column grew" from "variation collapsed"
        v0, v1 = c0 * e0, c1 * e1
        print("  %-18s E_std %9.5f -> %9.5f (%+6.1f%%) | col %8.4f -> %8.4f "
              "(|col| x E_std %8.4f -> %8.4f, %+6.1f%%)"
              % (n, e0, e1, 100 * (e1 - e0) / e0, c0, c1, v0, v1,
                 100 * (v1 - v0) / v0 if v0 else float("nan")))
    print()
    print("the |col| x E_std column is the size of the VARYING contribution;")
    print("the E_mean column is the size of the OFFSET contribution.")
    print()
    print("  OFFSET magnitude |col| * |E_mean| per dimension:")
    print("    %18s %s" % ("dim", " ".join("%9d" % st for st in steps)))
    for n in D:
        vals = [R[st][n + "_scale_col_norm"] * abs(R[st][n + "_E_mean"])
                for st in steps]
        print("    %18s %s" % (n, " ".join("%9.4f" % v for v in vals)))
    print()
    print("  VARYING magnitude |col| * E_std per dimension:")
    print("    %18s %s" % ("dim", " ".join("%9d" % st for st in steps)))
    for n in D:
        vals = [R[st][n + "_scale_col_norm"] * R[st][n + "_E_std"]
                for st in steps]
        print("    %18s %s" % (n, " ".join("%9.4f" % v for v in vals)))


if __name__ == "__main__":
    main()
