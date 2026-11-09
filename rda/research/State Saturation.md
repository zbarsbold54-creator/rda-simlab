# State Saturation

## What it is

**Saturation** is when a drive value is pushed so close to the edge of its
range that the gradient through it dies and the state stops responding.

For `tanh`, the derivative is `1 - tanh²`, so:

| argument | derivative |
|---|---|
| 1 | 0.42 |
| 2 | 0.07 |
| 3 | 0.0099 |

The gradient signal collapses as the argument grows. A saturated drive cannot
be corrected.

## Why it matters here

Saturation was the dominant failure mode during development. It determined the
architecture's single most important design decision: the shape of the
[[RDA and Persistent State]] update rule.

The key insight of this project is that **bounded is not the same as healthy**.
The current update rule makes `|E| ≤ 1` structurally impossible to violate —
and `threat` still ended up railed inside that bound. See "The open problem"
below.

## The first failure — summing then squashing

The original rule was:

```
E_next = tanh(0.95·E_t + tanh(event) + tanh(pe))
```

The pre-squash argument can reach `0.95 + 1 + 1 = 2.95`, which `tanh` maps to
`0.9945`. Anything above `|1.472|` already yields `|E| > 0.9` — the state is
pinned near its asymptote across most of its dynamic range.

Measured over 500 real steps: **5 of 8 dimensions spent more than 94% of steps
with `|E| > 0.9`**. An untrained module saturated ~50% of the time, confirming
the problem was structural rather than a strong learned signal.

## The fix — convex combination

```python
E_next = self.decay * E_prev + (1.0 - self.decay) * drive
```

The retained state and the fresh drive are **convex-combined** with weights
summing to one. Both lie in `[-1, 1]`, therefore so does the output —
**provably**, with no final `tanh` needed.

Compare the argument reach: `0.95·1 + 0.05·1 = 1.0`, exactly at the boundary,
versus `2.95` before.

The code asserts the bound rather than observing it:

```python
assert E_next.abs().max() <= 1.0 + 1e-6
```

If a future change reintroduces saturation, training *stops* rather than quietly
producing thousands of meaningless steps.

### The cost

The new event's full influence is only `1 - 0.95 = 0.05` per step. Moving a
drive meaningfully takes tens of steps (effective half-life ≈ 13.5 steps). That
is the deliberate trade: slow-moving state, but state that cannot rail.

## The `threat` investigation

`threat` was the **only** dimension of 8 to saturate. The pre-tanh argument
decomposes as:

```
arg = 0.1·ev + pe_proj.weight·(pe/10) + pe_proj.bias
```

Identity verified to worst residual `4.6e-7` over 3000 steps.

At step 2999:

| component | value | share of arg_p50 |
|---|---|---|
| `pe_proj.bias` | −1.71124 | **80%** |
| `pe_proj.weight` (PE pathway) | −0.355 | 16.6% |
| `ev` (event pathway) | −0.074 | 3.5% |

- `arg_p50` crossed −2.0 at step **1131** and **never** crossed −3.0; it
  plateaued near −2.11.
- Mean tanh derivative fell 0.943 → 0.059, minimum 0.0507 — **marginal, not
  dead**. The late slope was *positive* (`+2.1e-7`/step): it was relaxing back.
- `pe_proj.weight[threat]` **crossed zero at step 569/570** (+0.00078 →
  −0.00186), ending at −0.575. Before that the PE pathway *opposed* the event
  pathway; after it, *reinforced*.
- Correlations confirm the swap: `corr(drive, event)` collapsed +0.9996 →
  +0.0353, while `corr(drive, pe)` rose +0.0099 → +0.9628.

**The honest verdict: a converged bias offset**, not a runaway and not a dead
channel. It reached equilibrium around steps 1300–1800.

### It was not a ratchet

A tempting story is "the bias accumulated one way until it hit the rail". The
gradient data refuses it: **1623 positive-signed bias gradients versus 1376
negative**. The bias overshot to −1.756 at step 1750 and is now relaxing back
upward. It converged; it did not run away.

## The open problem — bounded is not healthy

The `threat` channel is inside `[-1, 1]` and still informationally dead.

FiLM's contribution is `W[:,d] · E[:,d]`. If `E[:,d]` is constant, that
contributes only a **DC offset** and carries no information.

The offset-to-varying ratio `|E_mean| / E_std` over positions, ranked at step
2999:

| dimension | o/v ratio |
|---|---|
| **threat** | **436.1** |
| novelty | 139.5 |
| uncertainty | 11.6 |
| arousal | 5.8 |
| learning_progress | 3.4 |
| homeostatic_dev | 1.2 |
| integrity | 0.36 |
| agency | 0.21 |

`threat` is **1st of 8** in offset dominance. A constant with o/v = 436 is a
constant.

> Distinguishing "bounded" from "informative" is the open design question. The
> convex combination guarantees the first; nothing guarantees the second.

## A reporting trap worth remembering

`E_perdim`, `pe_perdim` and every `*_absmax` field in the logging are
**unsigned magnitudes**. They cannot show which bound a value is pinned against.

> A drive collapsed to the lower bound and a drive collapsed to the upper bound
> have the **same** absmax.

Two separate retractions in this project came from reading an unsigned field as
if it were signed. Always log signed min/max.

## Related concepts

- [[RDA and Persistent State]] — the update rule this justified
- [[Prediction Error]] — the bias that dominated `threat`
- [[Event Signal]] — the pathway that was displaced
- [[FiLM Modulation]] — why a constant drive carries no information
- [[Ablation Study]] — how the saturation was localised
- [[Prediction vs Actual State]] — a separate error signal

## Evidence / references

- [[DECISIONS]] — Decision 3: normalise FiLM magnitude rather than hard-clamping
- [[EXPERIMENT_LOG]] — E7, E11, E13-fixed: saturation and stabilisation
- [[FAILED_EXPERIMENTS]] — F7, a summary that contradicted its own measurement
- [[CURRENT_STATE]] — event instability, PE/FiLM instability
- [[RDA_ARCHITECTURE_GUIDE]] — Sections 3 and 5
