# STATUS OF THE LABEL-LEAK ALLEGATION

Date checked: 2026-10-03
Checker: code review (`research/CODE_REVIEW.md`)
Subject: the claim that `nanochat/rda.py` currently leaks labels

```
STATUS: NOT REPRODUCED
```

## What this means precisely

The historical leak was **real**. `research/Label Leakage.md` and
`research/EXPERIMENT_LOG.md` record that `rda.py` once built `pe_seq` from
`targets[:, 1:]`, which made `film_state[p]` a function of `y[p]` — the exact
label `logits[p]` is graded against — and that the `B'-s42 1.226305` canary was
withdrawn as a result. That history stands and is **not** retracted here.

What could **not** be reproduced is a leak in the **current** tree. The pre-fix
code was replaced by commit `969e067`, which changed the slice to
`targets[:, :-1]`. This check re-derived the index arithmetic from the current
source and tested it dynamically. The current tree does not leak.

This note preserves both facts, so the retraction history stays intact while the
current state is unambiguous.

## The index arithmetic, re-derived from current source

Preconditions, both verified rather than assumed:

- `dataloader.py` produces `y[p] == x[p+1]` (measured fraction `1.000000`)
- `gpt.py:532` scores `cross_entropy(logits[p], targets[p])` with **no shift**,
  so `y[p]` is the label attached to position `p`'s logits.

Required invariant:

```
film_state[p] may depend on y[j] only for j < p.
```

Because `y[j] == x[j+1]`, that is equivalent to `film_state[p]` depending on
`x[0..p]` and nothing beyond — exactly a causal transformer's entitlement at
position `p`.

How the current code achieves it:

| location | fact |
|---|---|
| `rda.py:963-965` | `pe_seq = prediction_error_from_logits(_safe_logits(...), targets[:, :-1])`, so `pe_seq[j]` scores `y[j] == x[j+1]` |
| `rda.py` recurrence | `E_next_seq[j] = decay*E_next_seq[j-1] + (1-decay)*drive[j]`, so `E_next_seq[j]` depends on `y[0..j]` |
| `rda.py:1057` | `film_state = cat([E_start.unsqueeze(1), E_next_seq])`, hence `film_state[0] = E_start` and `film_state[p] = E_next_seq[p-1]`, which depends on `y[0..p-1]` — i.e. `y[j]` only for `j < p`. **Satisfied.** |

The comment block at `rda.py:940-956` correctly explains why `targets[:, :-1]`
is required. That block is itself worth reading: the *original* comment made the
same indexing check and then mislabelled what lives at the index, which is how
the bug survived a careful review.

## Dynamic verification

**1. `diagnostics/test_no_label_leak.py` — PASS at every probed position.**

Importantly, this test **randomises the FiLM weights and then asserts they are
non-zero before running any check**, because at the zero-init FiLM
`modulate()` is the exact identity and *every* such check passes vacuously.
That guard is why its PASS is meaningful.

```
k=0    logits[:, :1]  bit-identical: YES  | pos>k moved: 3.604e-06
k=1    logits[:, :2]  bit-identical: YES  | pos>k moved: 2.658e-06
k=17   logits[:, :18] bit-identical: YES  | pos>k moved: 1.370e-06
k=127  logits[:, :128] bit-identical: YES  | pos>k moved: 0.000e+00
p=127  all labels >= 127 randomized; logits[:, :128] identical: YES
```

**2. Independent perturbation test**, written separately from the repo's own,
with the same vacuity guard.

> Note the first attempt was **itself vacuous**: at zero-init FiLM it reported
> "no leak" for any implementation. After breaking the identity, and asserting
> non-vacuity first:

```
sanity: loss changes when E_t_external changes = True
k=0: positions <= k changed? False   positions > k changed? True
k=5: positions <= k changed? False   positions > k changed: True
RESULT: any position p whose hidden moved when y[p] changed? False
```

Positions **after** `k` move; positions at or before `k` never do. That is the
signature of a causal, leak-free model.

## Scope and limits of this check

- Covers `film_state` / FiLM, which `rda.py:1079` states is the **only** route
  from the drive state to the logits.
- Does **not** affect `EXPERIMENT_LOG.md:2151`, which records that `1.226305`
  ran at commit `6ae93ce`, *before* the fix `969e067`. That run remains invalid
  regardless of the current tree's status — a number produced by pre-fix code is
  not rehabilitated by the code being fixed afterwards.
- `diagnostics/audit_*.py` were written to analyse the **pre-fix** code and will
  no longer reproduce their original findings against HEAD. That is expected;
  they are retained as the regression rationale.

## Conclusion

- **Historical leak (pre-`969e067`):** real, documented, affected numbers stay
  withdrawn.
- **Current tree:** no leak reproduced; the invariant is verified statically and
  dynamically, with the vacuity trap explicitly guarded against.
- Any future change to the PE indexing, the `E_in`/`film_state` shifts, or the
  FiLM zero-initialisation must re-run `test_no_label_leak.py` **before** its
  numbers are reported.