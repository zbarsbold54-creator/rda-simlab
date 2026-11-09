# Prediction vs Actual State

## What it is

**Drive prediction error** (`dpe`) is the difference between what the
[[State Prediction]] module *expected* the next state to be and what it
actually was:

```python
dpe = drive_prediction_error(E_next_seq, E_hat_seq)
```

It is a **self-consistency signal**: how surprising was the state's own
evolution?

## Why it matters here

`dpe` is the one signal in the system that measures the *state itself* rather
than the input or the outcome. That makes it the natural candidate for a
"something has changed internally" signal — the kind of thing a homeostatic
system would need.

It is also, per [[HYPOTHESES]] H4, an untested hypothesis that `dpe` carries
information about state regularity.

**But it is not consumed.** Nothing about memory, learning rate, or attention is
modulated by it. It is computed, returned, logged, and that is all. See
[[DECISIONS]] Decision 4.

## Do not confuse it with Prediction Error

This is the single easiest confusion in this codebase, and the naming invites
it:

| | source | what it measures |
|---|---|---|
| **prediction error** | `prediction_error_from_logits` | how surprised the model was by the **token that arrived** |
| **drive prediction error** | `drive_prediction_error` | how far the **state** moved from what was predicted |

The first is an *input* to the state. The second is an *output* of the
predictor.

Both derive from the word "prediction", they are computed in the same
`forward()`, and they appear in the same log lines with similar field names.
They have nothing to do with each other.

## Why it was measured

The `threat` saturation investigation needed to know which pathway was
responsible. Decomposing the pre-tanh argument:

```
arg = 0.1·ev + pe_proj.weight·(pe/10) + pe_proj.bias
```

gave, at step 2999:

| component | value | share |
|---|---|---|
| `pe_proj.bias` | −1.71124 | **80%** |
| `pe_proj.weight` | −0.355 | 16.6% |
| `ev` | −0.074 | 3.5% |

Correlations confirmed the pathway swap: `corr(drive, event)` collapsed
+0.9996 → +0.0353 while `corr(drive, pe)` rose +0.0099 → +0.9628.

That analysis is what identified `pe_proj.weight[threat]` **crossing zero at step
569/570** (+0.00078 → −0.00186) as the switch point — before which the PE
pathway *opposed* the event pathway, after which it *reinforced*.

`dpe` did not drive that conclusion, but it was logged throughout as context.

## The reporting trap

`dpe` and the other per-dimension fields are logged as **unsigned magnitudes**.
`absmax` cannot show which bound a value is pinned against — a drive collapsed
to the lower bound and one collapsed to the upper bound have the **same**
absmax.

Two retractions in this project came from reading an unsigned field as signed.

> Always log signed min/max. Never label an absmax table "mean".

## Related concepts

- [[State Prediction]] — the module that produces `E_hat`
- [[Prediction Error]] — the *other* prediction error, an input not an output
- [[State Saturation]] — what the decomposition was investigating
- [[RDA and Persistent State]] — the state whose regularity is measured

## Evidence / references

- [[DECISIONS]] — Decision 4: do not consume `dpe` yet
- [[HYPOTHESES]] — H4
- [[CURRENT_STATE]] — the dependency graph that tracked `dpe`
- [[EXPERIMENT_LOG]] — E9, E10: PE/FiLM dependency graph and ablations
