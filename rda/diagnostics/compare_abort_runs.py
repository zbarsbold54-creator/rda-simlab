"""Compare the aborted and no-abort runs over their overlapping steps.

I wrote in eval_e13_3000_noabort.py's docstring that "the trajectory up to
step 2324 is identical to the aborted run", on the reasoning that the abort
is a post-hoc check on an already-computed row. The console output at step 25
disagrees: loss 7.9223 here vs 7.9165 there, pe_max 12.651 vs 12.642.

If that holds, the two runs are NOT bit-identical, and the docstring claim is
wrong. The likely cause is GPU nondeterminism (nondeterministic reductions in
the backward pass), not the abort bound: compute_init fixes seeds, but
nothing here calls torch.use_deterministic_algorithms.

This matters for interpretation. If the runs diverge, then differences
between them cannot be attributed to the abort bound alone, and any
statement of the form "the no-abort run confirms the aborted run up to
step X" needs to be a statistical claim, not an identity claim.

Read-only.
"""
import json
import sys

A = json.load(open("logs/pe_3000_fixed.json"))          # aborted at 2324
B = json.load(open("logs/pe_3000_fixed_noabort.json"))  # full 3000

n = min(len(A), len(B))
print("aborted run steps    : %d" % len(A))
print("no-abort run steps   : %d" % len(B))
print("overlap              : %d steps" % n)
print()

KEYS = ["loss", "E_absmax", "pe_max", "mod_scale_absmax", "applied_scale_absmax",
        "threat_E_absmax", "threat_tanh_arg_absmax", "threat_applied_gain"]
have = [k for k in KEYS if k in A[0] and k in B[0]]
missing = [k for k in KEYS if k not in A[0] or k not in B[0]]
if missing:
    print("NOT in the aborted run (added later): %s" % missing)
    print()

print("%-24s %14s %14s %12s" % ("field", "max |A-B|", "rel", "identical?"))
for k in have:
    mx = 0.0
    for i in range(n):
        mx = max(mx, abs(A[i][k] - B[i][k]))
    denom = max(1e-12, max(abs(A[i][k]) for i in range(n)))
    print("%-24s %14.3e %14.3e %12s"
          % (k, mx, mx / denom, "YES" if mx == 0.0 else "no"))

print()
exact = [k for k in have
         if all(A[i][k] == B[i][k] for i in range(n))]
print("fields bit-identical over the whole overlap: %s"
      % (", ".join(exact) if exact else "NONE"))
print()

# Does E_absmax track closely enough for the abort to be the only difference?
e = [abs(A[i]["E_absmax"] - B[i]["E_absmax"]) for i in range(n)]
print("E_absmax: max abs divergence %.3e, mean %.3e over %d steps"
      % (max(e), sum(e) / len(e), n))
print("E_absmax at step 2320: aborted %.6f  no-abort %.6f"
      % (A[2320]["E_absmax"] if len(A) > 2320 else float('nan'),
         B[2320]["E_absmax"]))
