# E15 — end-of-run per-dimension ablation of the FiLM input

> ### READ THE CORRECTION FIRST
>
> **The interpretation below is SUPERSEDED.** The measured table is correct and
> stands. Two conclusions drawn from it are WITHDRAWN:
>
> 1. **"offset is used" is WRONG for threat and novelty.** Negative delta means
>    the loss DECREASED when the channel was removed. Threat zero-ablate
>    (-0.0017) and novelty (-0.0008) mean those channels were a net COST to the
>    loss on this checkpoint. Only `uncertainty` (+0.0177) and `agency` (+0.0016)
>    have a clearly useful offset.
> 2. **"420x" is an arithmetic error.** zero_ablate_ALL is +0.0317, which is
>    **1.79x the largest single delta** and **1.96x the sum of the singles**
>    (+0.0162). The relationship is SUPER-ADDITIVE, not cancelling. The
>    "anti-correlated columns" explanation is withdrawn; the interaction is
>    NOT measured.
>
> Also: the noise floor 0.0 is a DETERMINISTIC eval, not sampling error, and the
> `within_noise` field in the JSON is meaningless. See the E15 correction block
> in `research/EXPERIMENT_LOG.md`.

Date: 2026-10-01. Seed 42, 3000 steps, `--abort-steps 10 --abort-bound 0.999`,
8 held-out val batches, batch size 4, T=1024, weights frozen, `torch.no_grad()`,
no optimizer step. **FAIL against the pre-registered criterion** (see below).

## TWO BUGS FOUND BEFORE THE RUN WAS TRUSTWORTHY

**1. The ablation was computed and then discarded.** `--ablate`/`--ablate-out`
already existed in `1ef0861`, but the `modulate` wrapper built the ablated
tensor `_E_use` and then returned `_orig_modulate(x, E_t, ...)` — the ORIGINAL
`E_t`. Every delta would have been exactly 0.0 and every dimension would have
falsely read as inert. Fixed to `return _orig_modulate(x, _E_use, ...)`. When
disarmed `_E_use is E_t`, so the training forward is bit-for-bit unchanged.
Verified by smoke test: zero-ablate deltas became nonzero.

**2. `per_batch_state` was recorded but never serialized.** `_eval_condition`
returns only `ABL["state_used"]`, so the per-batch list was dropped at write
time. The single-batch `state_used` below IS from the first held-out batch only.
**STILL UNFIXED** — the per-batch means for all 8 batches are not in the output.

## A NOTE ON THE NOISE FLOOR

    baseline      4.24794864654541
    repeat        4.24794864654541
    noise floor   0.0   EXACTLY

The repeat is bit-identical to the baseline. The non-determinism documented in
E13-fixed-3000 is in the BACKWARD pass; there is no backward pass here, so
evaluation is deterministic and the noise floor is legitimately zero. This
makes the `within_noise` field in the JSON **useless** — with floor 0.0 every
nonzero delta is flagged "not within noise". Do not read it. The deltas below
are compared against each other and against zero, not against the floor.

## MEASURED — per-dimension loss deltas (signed, vs baseline)

| dim | (a) mean-ablate | (b) zero-ablate |
|---|---|---|
| novelty | **+0.0000043** | **-0.0007911** |
| uncertainty | -0.0002193 | **+0.0176988** |
| threat | **-0.0000010** | **-0.0017176** |
| integrity | +0.0000000 | -0.0000505 |
| agency | +0.0000176 | +0.0016174 |
| learning_progress | +0.0000391 | -0.0000639 |
| arousal | -0.0000229 | -0.0004416 |
| homeostatic_dev | -0.0000277 | -0.0000830 |
| **ALL 8 mean** | **+0.0004377** | **+0.0317354** |

baseline loss 4.247949. Signs are as measured; negative = ablated channel
HELPED, i.e. removing it lowered the loss.

MOST IMPORTANT LINE: **zero-ablate_ALL is +0.0317, i.e. 420x the largest single
dimension and it is POSITIVE**, while every per-dimension zero-ablate is at
most +0.0177. The effects are NOT additive — they nearly cancel.

## Per-sequence means used (first held-out batch, E shape 4x1024x8)

threat (idx 2):   -0.97160  -0.96949  -0.96825  -0.97067   (absmax 0.9764-0.9775)
novelty (idx 0):  +0.74699  +0.74357  +0.75375  +0.74197   (absmax 0.7515-0.7618)

Both match E13-fixed-3000-film-fix (threat |E_mean|/E_std = 436) and
E13-fixed-3000-all8 (novelty p50 +0.7165, off-rail) to within run-to-run spread.
Threat varies across sequences by only 0.0034 across the four sequences;
novelty by 0.0118. Threat is the flatter of the two, as expected.

## INTERPRETATION — measured first

For **threat**:
- (a) mean-ablate delta = **-0.0000010**. Within any plausible reading of zero.
- (b) zero-ablate delta = **-0.0017176**, ~1700x larger in magnitude than (a).
- So: **position variation is NOT used; the offset IS used.**

For **novelty**:
- (a) = **+0.0000043**, effectively zero.
- (b) = **-0.0007911**, ~180x larger than (a).
- Same pattern: **position variation is NOT used; the offset IS used.**

This is the signature the ablation was designed to detect, and it is exactly
what the |E_mean|/E_std measurement predicted for both dimensions (436 and 140,
the two highest of all eight).

**LABELLED INFERENCE (not measured):** that the FiLM pathway uses these
dimensions mainly as a near-constant per-sequence offset, and that the
across-position variation is functionally dead weight at step 2999. The
inference is that the mechanism that USED to amplify that variation (the FiLM
self-excitation loop, E10/E13) has been damped by the `ab9b560` normalization
fix, leaving the offset intact. That last step is a story, not a measurement.

**NOT a significance claim.** With a noise floor of exactly 0.0 every nonzero
delta is formally resolvable, but (a) values of 1e-6 to 4e-5 on a loss of 4.25
are relative effects of 1e-7 to 1e-5. They are best described as "indistinguishable
from zero at the precision this measurement supports", not "exactly zero".

**ONE SEED, ONE CHECKPOINT.** Seed 42, one run, one end state, 8 batches. No
seed replication. GPU backward-pass nondeterminism does not apply to this eval,
but the CHECKPOINT it was measured on carries that nondeterminism from training.

## LIMITATIONS

- The pre-registered criterion was `max|E| <= 0.98` and it is NOT re-checked
  here; this entry is a diagnostic, not a gate.
- Single seed, single checkpoint, toy scale (depth 8, d_model 768).
- 8 batches of 4x1024 = 32768 tokens. Small. Per-batch loss spread across the
  8 batches is not reported and would give a better sense of batch-to-batch
  variation than the (zero) repeat does.
- `per_batch_state` not serialized (bug 2 above), so the reported means are one
  batch. They are consistent with prior runs, but this is weaker than intended.
- The near-cancellation between per-dimension and ALL-8 deltas is unexplained.
  It may mean the columns are strongly anti-correlated. NOT measured here.
---

## Related Concepts

- [[Ablation Study]] — this run is the per-dimension instance of that method
- [[FiLM Modulation]] — what is being ablated, dimension by dimension
- [[RDA and Persistent State]] — the 8 dimensions whose contributions are measured
- [[Prediction Error]] — the input that dominates the drive
- [[State Saturation]] — the rail condition the per-dimension deltas probe
- [[BPB]] — the metric and its signed-delta convention

The two pre-run bugs recorded above are instances of the project's two most
common error classes: a wrapper returning the original input (so every delta
would be 0.0), and reading an unsigned `absmax` as if it were signed.
