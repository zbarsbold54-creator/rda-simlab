"""MEASURE the persistence semantics of RDAStateModule.decay.

Read-only, CPU-only, tiny tensors. No training, no checkpoints. Run:

    python diagnostics/verify_decay_semantics.py

WHY THIS EXISTS. `decay` is commonly read as "how far the state persists", and
the name invites a reading the shipped value does not support. At decay=0.95 the
1/e horizon is ~19.5 POSITIONS, so:

  * within a 1024-token row a carried-state perturbation is below half its
    initial size by t=13 and numerically zero by t=500;
  * across a document boundary inside a packed row (~256 tokens) the carried
    state arrives attenuated by 0.95**256 = 2.0e-6, i.e. it does NOT persist
    across documents in any measurable sense.

So the decay implements WITHIN-SEQUENCE INERTIA. That is a coherent design and
it is not a bug -- but the distinction has to be measured rather than asserted,
because "persistent state" is the module's central claim. See
docs/decay_semantics.md and research/CODE_REVIEW.md F3.

THIS DOES NOT CHANGE ANY PARAMETER. It is a measurement of the current
configuration, so that a run record can state which persistence is active.
"""
import importlib.util
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanochat.rda import RDAStateModule  # noqa: E402

B, T = 2, 1024
N_DRIVES = 8
DOC_SPACING = 256          # docs are packed ~256 tokens apart in a 1024 row


def build(decay):
    torch.manual_seed(0)
    return RDAStateModule(8, N_DRIVES, decay=decay)


def main():
    sm = build(0.95)
    drive = torch.tanh(torch.randn(B, T, N_DRIVES) * 0.3)
    E0 = sm.init_state(B, "cpu")
    ref = sm.update_seq(E0, drive)
    alt = sm.update_seq(E0 + 0.5, drive)          # perturb ONLY the carried state
    d = (ref - alt).abs().amax(dim=-1)[0]          # (T,) influence per position

    print("=" * 72)
    print("RDA decay semantics (measured on production update_seq, CPU)")
    print("=" * 72)
    print("  decay_horizon attribute = %.2f positions" % sm.decay_horizon)
    print("  1/ln(1/decay)          = %.2f positions" % (1 / math.log(1 / 0.95)))
    print("")
    print("  %-8s %-14s %-14s" % ("t", "|dE| from carry", "0.95**t"))
    for t in (0, 5, 10, 20, 50, 100, 200, 500, 1023):
        print("  %-8d %-14.4e %-14.4e" % (t, d[t].item(), 0.95 ** t))
    half = (d > 0.5 * d[0]).nonzero()
    print("")
    print("  carried influence above 50%% of initial out to t = %d"
          % (int(half[-1]) if len(half) else 0))

    print("")
    print("  Cross-document (a packed row crosses doc boundaries mid-row):")
    for ds in range(DOC_SPACING, T, DOC_SPACING):
        print("    doc start t=%-5d |dE| = %.4e   (0.95**%d = %.3e)"
              % (ds, d[ds].item(), ds, 0.95 ** ds))
    print("  => carried state across a %d-token document is %.1e of its"
          % (DOC_SPACING, 0.95 ** DOC_SPACING))
    print("     original size: NOT persisted across documents at this decay.")

    # --- what the value would have to be for each role (numbers, not a change)
    print("")
    print("  decay needed per role (informational):")
    for label, decay in [("within-seq ~20 tok (current)", 0.95),
                         ("within-seq ~100 tok", 0.99),
                         ("across a 256-tok document", 0.999),
                         ("across a 1024-tok micro-step", 0.9999)]:
        print("    %-30s decay=%.5f  1/e horizon=%7.0f pos"
              % (label, decay, 1 / math.log(1 / decay)))

    # --- the bound and the closed form are UNAFFECTED by decay. Assert it, so
    # --- a future change to decay cannot silently reintroduce saturation.
    print("")
    print("  Bound / closed-form safety vs decay (both must stay valid):")
    ok = True
    for decay in (0.95, 0.99, 0.999, 0.9999):
        s = build(decay)
        dd = torch.tanh(torch.randn(B, T, N_DRIVES) * 0.3)
        E = s.update_seq(s.init_state(B, "cpu"), dd)
        lim = RDAStateModule._max_safe_T(decay, torch.float64)
        bound_ok = E.abs().max().item() <= 1.0 + 1e-6
        safe_ok = T <= lim
        ok = ok and bound_ok and safe_ok
        print("    decay=%-8s max|E|=%.6f (bound ok=%s)  float64 limit T=%d (T<limit=%s)"
              % (decay, E.abs().max().item(), bound_ok, lim, safe_ok))
    print("    -> the |E|<=1 bound is a property of the CONVEX COMBINATION, not of")
    print("       the decay value, and _max_safe_T GROWS as decay -> 1. Raising")
    print("       decay is numerically safe; it is a scientific decision, not a")
    print("       numerical one.")

    print("")
    print("=" * 72)
    print("VERDICT: decay=0.95 implements within-sequence inertia (~20 positions).")
    print("Cross-document and cross-micro-step persistence are NOT active.")
    print("This is a measurement, not a defect. No parameter is changed here.")
    print("=" * 72)
    if not ok:
        print("FAIL: bound or closed-form limit violated for some decay", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())