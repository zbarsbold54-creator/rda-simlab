# Control Arm

## What it is

A **control arm** is a variant of the experiment that is identical to the real
one in every respect *except* the single thing you want to isolate.

Without one, a result has multiple explanations. With one, it has one.

## Why it matters here

Arm B' beats baseline by 0.018024 [[BPB]] on average across three seeds. That
number alone is consistent with **three** different explanations:

1. **The state information helps.** The 8 values carry something useful.
2. **The extra parameters help.** +27,760 parameters is a capacity increase,
   and more capacity can improve a model regardless of what it computes.
3. **The extra compute helps.** B' costs more per step, and a
   deeper-effective network is a better network.

B' alone cannot distinguish these. That is the entire reason a control exists.

## How arm C was designed

Arm C keeps every RDA parameter and every RDA computation, but replaces the
[[FiLM Modulation]] input with a learned constant:

```python
if self.const_state_vec is not None:
    film_state = self.const_state_vec.to(film_state.dtype).expand_as(film_state)
```

`expand_as` returns a **view** — zero allocation. `.to(dtype)` is a no-op when
dtypes match.

So arm C has:

| | B' | C |
|---|---|---|
| Parameters | +27,760 | +27,768 |
| Drive computed | yes | **yes** |
| Recurrence run | yes | **yes** |
| Predictor run | yes | **yes** |
| `dpe` computed | yes | **yes** |
| FiLM input | per-position `E` | constant vector |
| Backward through drive path | yes | **no** (pruned) |

The only *intended* difference is information. Three real differences remain,
and any report must state them:

1. **Information reaching FiLM** — the intended variable.
2. **RDA modules untrained at init** — in C, `event_proj`, `pe_proj` and
   `h_norm` get no gradient and are only [[Weight Decay|decayed]].
3. **No backward through the drive path** — autograd prunes it because nothing
   in the loss depends on `E`.

Points 2 and 3 are *consequences* of point 1, but they mean C is not purely
"B' with information removed". If C underperforms B', that is consistent with
either the information mattering *or* the drive modules needing training to be
useful.

## The three preconditions

A control is only as good as its preconditions. Each was verified by
measurement *before* the A/B launched.

### 1. No saddle

With zero-init FiLM weights, `∂L/∂c ∝ W_scale = 0` at step 0. If it stayed
zero, C would be a reparameterisation of the identity and the comparison
vacuous.

| backward | `||∂L/∂c||` | `||∂L/∂W_scale||` |
|---|---|---|
| 1 | 0.0000000000e+00 *(zero by design)* | 2.9452124238e-01 |
| 2 | 2.3130022455e-03 | 1.2724294662e+00 |
| 3 | 1.6563907266e-02 | 2.4881629944e+00 |
| 4 | 1.1805012822e-02 | 1.3823006153e+00 |

Nonzero from backward 2 onward.

> A first probe reported a *saddle* here — a false positive. It used
> `lr=0.0`, so the FiLM weights never left zero-init and the gradient was
> structurally zero at every step. The probe had to actually step the optimizer
> before it measured anything.

### 2. Forward-call parity

If C skipped work B' does, it is not a control. Measured per step, identical:

| function | calls/step (C) | calls/step (B') |
|---|---|---|
| `update_seq` | 1 | 1 |
| `predict_seq` | 1 | 1 |
| `event_proj` | 1 | 1 |
| `drive_seq` | 1 | 1 |
| `modulate` | 1 | 1 |

### 3. RNG isolation

C's extra parameter must not shift the data order or any shared init.
`const_state_vec` is drawn from a **private CPU generator** seeded
`seed + 1000003`. Verified: 78 shared parameters bit-identical, 0 mismatches,
and global draws after B' vs C construction identical.

## "Untrained and decayed", not "at init"

In arm C the three drive projections receive **no gradient at all** (autograd
pruned; `p.grad is None`, not a zero tensor). They are added to the grad-free
list only so `MuonAdamW._compute_adamw` does not crash calling `grad.float()`
on `None` (`optim.py:73`).

They are **not frozen**. `grad_free` is read *only* by the zero-grad priming in
`base_train.py`; `optim.py` never sees the key, and `adamw_step_fused` applies
decoupled decay unconditionally (`optim.py:75`, `p32.mul_(1 - lr*wd)`) before
examining the gradient.

Measured over 3 steps at lr 3e-3, wd 0.01: all six tensors shrank by exactly
`(1 - 3e-5)³`; observed ratio 0.999910178 vs predicted 0.999910003, matching to
`1.8e-7`.

Getting this wrong would have made the drift check unfalsifiable — an earlier
"bit-identical to init" reading was an artifact of a probe that excluded those
params from its own optimizer at `wd=0`.

## Status: unresolved

| verdict | criterion |
|---|---|
| C matches B' | `|C - B'|` < 0.0137 on **all three** seeds |
| states contribute | C worse than B' by > 0.0137 on **all three** seeds |
| otherwise | inconclusive |

> **The 0.0137 threshold is a proposal, not a calibrated value.** It is roughly
> 4× the measured repeat noise (0.000067) and comparable to the per-seed spread
> of the paired B'-baseline diffs (0.0097). It has not been derived from a power
> calculation. With three seeds, a result near the threshold should not be
> trusted in either direction.

A 200-step, single-seed preview at step 150 gave C − B' = +0.003634 —
suggestive of a capacity explanation, but far below the threshold and
inconclusive on its own.

## Related concepts

- [[RDA and Persistent State]] — what C controls for
- [[FiLM Modulation]] — the path whose input C replaces
- [[BPB]] — the metric the verdict is judged on
- [[Ablation Study]] — the related method of removing one component
- [[Prediction Error]] — one of the modules left untrained in C
- [[Weight Decay]] — what still acts on C's untrained modules

## Evidence / references

- [[E15_ABLATION]] — the related per-dimension ablation
- [[RESEARCH_QUESTIONS]] — what would count as answering the primary question
- [[HYPOTHESES]] — H1, H5
- [[EXPERIMENT_LOG]] — E17, E18: the A/B and the diagnostics
