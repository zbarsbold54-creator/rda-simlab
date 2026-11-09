# What `decay` actually buys: persistence semantics in `RDAStateModule`

Status: **investigation record, not a design change.** `decay` is unchanged at
`0.95`. This document exists because the parameter's *name* invites a reading its
current value does not support.

Reference: `research/CODE_REVIEW.md` finding F3.

## The one number that matters

The recurrence is

```
E[t] = decay * E[t-1] + (1 - decay) * drive[t]
```

so the influence of the incoming state at position `t` is `decay**t`, and the
**1/e horizon** is

```
horizon = 1 / ln(1 / decay)          # positions
```

`RDAStateModule.decay_horizon` exposes exactly this and `base_train.py` prints it
at startup, so a run record can never be read as claiming a persistence the
configuration does not provide.

At the shipped default `decay = 0.95`:

| quantity | value |
|---|---|
| 1/e horizon | **19.5 positions** |
| 50% echo out to | t = 13 |
| influence at t = 100 | 5.9e-3 |
| influence at t = 500 | 7.3e-12 |
| influence at t = 1023 | 1.6e-23 |
| across a 256-token document boundary | 2.0e-6 |

All measured with the production `update_seq`, not derived on paper.

## Three roles `decay` could plausibly be serving

They are different designs and they need different values. This is why the
parameter is worth being explicit about.

**A. Within-sequence inertia.** The state resists rapid change inside a row; the
drive is smoothed over ~20 positions rather than reacting to one token. At
`decay = 0.95` this is *exactly what happens*, and it is a coherent design: it is
the same role `h_norm` and the RMS normalisers play, applied in time instead of
across features. Measured behaviour matches the mechanism — a state perturbation
falls below half its initial size by t=13 and is numerically zero by t=500.

**B. Persistence across documents.** `dataloader.py` packs several documents per
row (`row_capacity = T+1`), so a row crosses document boundaries in its middle.
"Decay-only persistence" is usually read as "the state survives a document
boundary". At `decay = 0.95` it does not: a boundary 256 tokens in attenuates the
carried state by `0.95**256 = 2.0e-6`. Every document after the first begins at
effectively a fresh state. **This role is not implemented at the current value.**

**C. Persistence across micro-steps.** `base_train.py` carries `E_next` from one
optimizer step to the next. The carried value is consumed at position 0 of the
next row at full weight and is numerically irrelevant by position ~130. So the
cross-step carry is real but confined to the first ~few positions of a row.

## Why `decay` was not changed

Changing it would be a hyperparameter sweep disguised as a bug fix, and it would
invalidate every existing RDA number. Specific reasons:

1. **Role A is already served, and it is the role the value was chosen for.**
   `rda.py`'s `update()` docstring documents the measurement that motivated the
   *gated/convex* form (`decay` as a smoothing weight alongside `(1-decay)*drive`),
   not as a long-memory device. `0.95` sits in the same family as the `betas` used
   throughout the optimizer (`0.9`, `0.95`).

2. **`decay` is entangled with saturation control, and that was measured.**
   `research/EXPERIMENT_LOG.md` control **C5 (`decay 0.5`) was WORSE** for the
   event-term growth (`event -> 1.3471`, x5.37 vs x5.03). A sweep in the other
   direction (`0.99`, `0.999`) has never been run, so the trade-off is unmapped —
   which is an argument for *measuring* it, not for quietly picking a value.

3. **The numerics do not constrain it.** The `|E| <= 1` bound is a property of the
   convex combination `decay*E_prev + (1-decay)*drive`, not of the value: verified
   `max|E| = 0.152` at `0.95` and `0.012` at `0.999`. And `_max_safe_T` *grows* as
   `decay -> 1` (13837 at `0.95`, 709427 at `0.999`), so the float64 closed form
   gets safer, not less safe. There is no numerical excuse for the current value.

## If long-horizon persistence is wanted

This is a new experiment, not a fix. Requirements:

- Pick the role first (B or C above); they imply different targets.
- Expose `--rda-decay` (currently hardcoded at `rda.py:89` and `:657`) so the
  sweep is recorded in the command line and in the checkpoint's `user_config`.
- Re-run the A/B from scratch. The effect of `decay` on `val/bpb` is unknown and
  must not be inferred from existing arms.
- Re-check the saturation gate: `base_train.py` aborts when a drive dimension
  stays `|E| > 0.98` for 10 consecutive steps. A longer horizon changes the
  `arg` trajectory, so the existing abort statistics do not carry over.
- Keep `_max_safe_T` in float64. At `decay = 0.999` the float32 limit would be
  88678 — still fine at T=1024, but the float64 accumulation costs nothing here
  (the tensors are `(B, T, 8)`, no matmul) and should not be traded away.

## Reproducing the measurements

Every number above comes from `diagnostics/verify_decay_semantics.py`, which uses
the production `update_seq` on tiny CPU tensors. No training required.