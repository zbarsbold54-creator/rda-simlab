# Event Signal

## What it is

The **event signal** is the [[RDA and Persistent State]] drive derived from the
model's own hidden state — as opposed to the [[Prediction Error]], which is
derived from the outcome.

```
hidden state (unmodulated, pre-outcome)
    ↓  LayerNorm (h_norm)
    ↓  event_proj: Linear(768 → 8)
    ↓  unit-RMS normalise over the 8 drives
    ↓  × event_drive_scale (0.1)
    ↓
event term
```

Conceptually: "something happened in the input, and here is how it registers
*before* we know what the answer was."

## Why it matters here

The event pathway is the RDA equivalent of "something changed". Together with
[[Prediction Error]] ("and here's how surprising the result was"), it forms the
complete drive.

Three layers of bounding protect it, and **each one closes a hole found by
measurement** — the process is documented below because it is the best
diagnostic story in the codebase.

## How it works

### Layer 1 — LayerNorm on the input

```python
ev = self.event_proj(self.h_norm(hidden_state))
```

`h_norm` makes the drive depend on the **direction** of the residual stream
rather than its magnitude, so base-model activation growth cannot inflate it.

This worked exactly as intended: LayerNorm output rms stayed at **exactly
1.000** and `h_norm.weight` at **exactly 1.0** over 30 steps.

### Layer 2 — output RMS normalisation (the layer that mattered)

```python
ev = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
```

LayerNorm was never the failure. The problem: `||h_norm|| = sqrt(d) = 27.71` is
a **ceiling**, not a value. So `event_proj(h)` can still reach
`||w_row|| × 27.71`, and `||w_row||` is **learned and unbounded**.

| quantity | 30 steps, seed 42 | ratio |
|---|---|---|
| `event_proj` \|\|W\|\| | 1.63 → 3.51 | ×2.15 |
| event \|\|max\|\| | 0.25 → 1.26 | ×5.03 |
| LayerNorm out rms | 1.00 → 1.00 | ×1.00 ← **working** |

The drive grew because the **projection** grew, not because the input did.

The decisive control: freeze the **entire base model**, so `h` is byte-identical
every step. `||W||` still went 1.63 → 3.51, and the event term still reached
1.26.

That single experiment isolates the cause to the gradient on `event_proj`
itself, and rules out **both** hidden-state drift **and** any FiLM feedback loop
as the driver.

> **Normalising the input gives invariance to the input's magnitude. It does
> nothing about a learned weight that grows. Apply the normalisation where the
> magnitude actually lives.**

This is scale-invariance, not a clamp: it does not saturate, does not flatten
gradients, and preserves the learned relative scale *between* drives exactly —
the sign and pattern of the eight components is the signal.

### Layer 3 — a fixed 0.1 multiplier

```python
event_drive_scale = 0.1   # fixed, NOT a Parameter
```

Makes the event term a small perturbation of the tanh sum rather than something
that can saturate it on its own.

This is deliberately **not** `pe_reference_scale` (which normalises the PE input
and stays untouched), and **not** [[Weight Decay]] (which bounds `|W|` but not
`|W·h|` when `|h|` is large). It bounds the **output**, which is the quantity
that actually crossed the threshold.

## Project usage

The event pathway reads the **unmodulated** residual stream. That is structural,
not incidental: `drive_seq(pre_outcome_hidden, pe_seq)` is called *before*
`modulate()`, so the state can never see its own downstream effect through the
event term.

This ordering is what makes the [[RDA and Persistent State]] recurrence
loop-free in its inputs — and therefore what makes the closed-form
vectorisation possible.

Zeroing the event term in the [[Ablation Study]] gave 1.596395 versus 1.137101
for B' — a large effect, but smaller than zeroing [[Prediction Error]]
(4.371596).

## A related measurement trap

On a `(768, B, T)` tensor, calling `.std()` or `.mean()` **without an explicit
`dim`** reduces over units × batch × positions, so a term proportional to the
*column-direction spread of the weights* contaminates the result. That
produced a confident, wrong ranking of which drive was most DC-dominated.

> Always pass `dim` explicitly, reduce over positions only, then RMS over units.

## Related concepts

- [[Prediction Error]] — the other drive input
- [[RDA and Persistent State]] — what the event signal drives
- [[Residual Stream]] — what it reads
- [[State Saturation]] — the rail it originally drove the state onto
- [[Weight Decay]] — why it is not the right tool for bounding `|W·h|`
- [[Ablation Study]] — the event ablations

## Evidence / references

- [[DECISIONS]] — Decision 2: normalise the projection OUTPUT, not just its input
- [[CURRENT_STATE]] — event pathway as verified in code
- [[EXPERIMENT_LOG]] — E7, E8: saturation investigation and the fix
- [[FAILED_EXPERIMENTS]] — F5, the freeze-base probe that was structurally incapable
- [[RDA_ARCHITECTURE_GUIDE]] — Section 6(b)
