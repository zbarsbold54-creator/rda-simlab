# Weight Decay

## What it is

**Weight decay** is a regulariser that shrinks parameters toward zero on every
optimizer step, independent of any gradient:

```
p  ←  p − lr × wd × p
```

Equivalently: `p ← p × (1 − lr × wd)` each step.

Its purpose is to keep parameters small, which limits overfitting and reduces
the effective complexity of the model.

## Why it matters here

Weight decay became central to this project through a **retraction**. It is the
reason a "grad-free" parameter is not the same as a frozen one, and correcting
that misunderstanding changed what the [[Control Arm]] result can mean.

## `grad_free` ≠ frozen

The RDA code marks several parameters grad-free:

```python
_g["grad_free"] = True
```

The natural reading is "these are skipped". They are not.

That flag is read **only** by the zero-grad priming in `base_train.py`.
`optim.py` never receives the key. And `adamw_step_fused` applies decoupled
decay **unconditionally**, before it ever looks at the gradient:

```python
p32.mul_(1 - lr_t * wd_t)      # optim.py:75
```

So a parameter with a zero gradient, enrolled in a group with
`weight_decay = 0.01`, **still shrinks every single step**.

### The measurement

Over 3 steps at lr 3e-3, wd 0.01, all six drive-module tensors shrank by exactly
`(1 − 3e-5)³`:

| parameter | init absmax | final absmax | rel. change |
|---|---|---|---|
| `event_proj.weight` | 3.608253e-02 | 3.607929e-02 | 8.982e-05 |
| `pe_proj.weight` | 9.488938e-01 | 9.488085e-01 | 8.989e-05 |
| `h_norm.weight` | 1.000000e+00 | 9.999101e-01 | 8.994e-05 |
| `h_norm.bias` | 0.000000e+00 | 0.000000e+00 | 0.000e+00 |

Predicted scale after 3 steps: **0.999910003**
Observed `event_proj.weight` ratio: **0.999910178**
Match to **1.8e-7**.

`h_norm.bias` did not move only because it is `0.0` to begin with.

## The consequence for arm C's drift check

In arm C, `event_proj`, `pe_proj` and `h_norm` receive no gradient — autograd
pruned the path because nothing in the loss depends on `E`. An earlier probe
reported them "bit-identical to init", which would have made a drift check
trivially pass.

That reading was an **artifact**: the probe excluded those params from its own
optimizer with `weight_decay = 0.0`, which is not the training configuration.

The correct description is **"untrained and decayed"**. Getting this wrong would
have made the check unfalsifiable — it would have passed for reasons that had
nothing to do with the mechanism under test.

## Why decay is the wrong tool for bounding the event projection

An early instinct was to control the event drive's magnitude with
[[Weight Decay]]. It does not work, for a structural reason:

> Decay bounds `|W|` but not `|W·h|` when `|h|` is large.

The event projection reads the [[Residual Stream]], whose magnitude is set by the
base model and is unbounded by anything in RDA. So a small weight times a large
activation is still large.

The project fixed the **output** instead — see [[Event Signal]] — which is the
quantity that actually crossed the threshold. This is recorded as
[[DECISIONS]] Decision 2.

## Project usage

| group | weight decay |
|---|---|
| `event_proj`, `pe_proj` | `args.rda_drive_wd` (default 0.01) |
| `e_init` | 0 |
| `predictor.*` | 0 |
| matrix / scalar base params | cosine-scheduled, scaled by depth and batch size |

Base weight decay is itself rescaled:

```python
weight_decay_scaled = args.weight_decay * sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
```

and follows a cosine schedule, `0.5 × (1 + cos(π · it / num_iterations))`.

## Related concepts

- [[Optimizer State]] — where the decay is applied
- [[Control Arm]] — the modules left untrained-and-decayed
- [[Event Signal]] — why decay could not bound the drive
- [[Residual Stream]] — the unbounded input it failed to control

## Evidence / references

- [[RDA_ARCHITECTURE_GUIDE]] — Section 15, retraction 7 and 8
- [[CURRENT_STATE]] — optimiser coverage line with per-group decay
- [[DECISIONS]] — Decision 2: normalise the projection output
- [[EXPERIMENT_LOG]] — E18: the decay measurement
