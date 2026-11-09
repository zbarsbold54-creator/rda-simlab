# Optimizer State

## What it is

**Optimizer state** is the extra memory an optimizer keeps about each parameter
in order to update it.

For AdamW, that is two numbers per parameter:

- **first moment** (`exp_avg`) — a running average of recent gradients. Its
  rough meaning: which direction the parameter has been pushed lately.
- **second moment** (`exp_avg_sq`) — a running average of recent *squared*
  gradients. Its rough meaning: how large those pushes have been.

The update divides the first by the square root of the second, which makes the
step size roughly scale-free — a parameter receiving large gradients gets a
proportionally smaller step.

## Why it matters here

Two project-specific facts depend on this machinery, and both produced real
errors.

### 1. A `None` gradient crashes the step

`MuonAdamW._compute_adamw` calls `grad.float()` **unconditionally**. A
parameter with `grad is None` therefore crashes the optimizer step.

Several RDA parameters are legitimately grad-free by design — `e_init`,
`predictor.*`, and in the [[Control Arm]] the three drive projections. They must
be given an explicit zero gradient rather than left as `None`.

The priming happens in `base_train.py` and **must be repeated on every
backward**, because `model.zero_grad(set_to_none=True)` runs after each
optimizer step, so the gradients are `None` again by the next micro-step.
Priming once would survive step 0 and then crash on step 1.

### 2. `grad_free` does not mean frozen

This is the project's most-repeated misconception, and it caused a retraction.

```python
_g["grad_free"] = True
```

This flag is read **only** by the zero-grad priming in `base_train.py`.
`optim.py` never sees the key. And `adamw_step_fused` applies decoupled decay
unconditionally at `optim.py:75`:

```python
p32.mul_(1 - lr_t * wd_t)      # happens BEFORE the gradient is examined
```

So a "grad-free" parameter in a group with `weight_decay = 0.01` **still shrinks
every step**.

Measured over 3 steps at lr 3e-3, wd 0.01: all six tensors shrank by exactly
`(1 - 3e-5)³`. Observed ratio 0.999910178 versus predicted 0.999910003 —
matching to `1.8e-7`.

> The correct description is **"untrained and decayed"**, not "at init" and not
> "frozen".

An earlier probe reported these modules as "bit-identical to init" — but that
probe excluded them from its own optimizer at `weight_decay=0`, which is not the
training configuration. The result was an artifact of the probe.

## Muon and AdamW together

`MuonAdamW` splits parameters by kind:

- **matrix parameters** (2-D weights) → Muon, which uses an orthogonalisation
  step rather than per-coordinate moments
- **everything else** → fused AdamW

Parameters are enrolled in groups with different learning rates: matrix
parameters, unembedding, embedding, scalars, plus RDA-specific groups with
`weight_decay = args.rda_drive_wd` on `event_proj`/`pe_proj` and zero on
`e_init`.

The training loop contains a coverage assertion: if any RDA parameter is in no
optimizer group, the run raises rather than silently never updating it.

## A related environmental note

`torch.compile` is disabled on this machine (`base_train.py:332`) because Triton
is unavailable on Windows. All runs are therefore eager, and all timings are
unoptimised. This is not a bug, but it means measured seconds-per-step includes
kernel-launch overhead that a compiled run would not pay.

## Related concepts

- [[Weight Decay]] — the term that acts even on grad-free parameters
- [[Control Arm]] — where three modules become grad-free
- [[State Prediction]] — four tensors that are grad-free in every arm
- [[Prediction Error]] — whose projections are grad-free in arm C

## Evidence / references

- [[CURRENT_STATE]] — optimiser groups and RDA enrolment
- [[DECISIONS]] — Decision 6, no sparse controllers before B' is tested
- [[RDA_ARCHITECTURE_GUIDE]] — Section 16, tooling debts
- [[EXPERIMENT_LOG]] — E6, the TREE provenance check
