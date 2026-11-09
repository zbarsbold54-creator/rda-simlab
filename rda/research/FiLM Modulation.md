# FiLM Modulation

## What it is

**FiLM** (Feature-wise Linear Modulation) takes a conditioning vector and
produces a per-feature gain and offset for another vector:

```
output = input × (1 + scale) + shift
```

In RDA the conditioning vector is the [[RDA and Persistent State]] `E` (8
numbers), and `scale`/`shift` are 768-dimensional — one gain and one offset per
residual-stream unit.

```
E (8)  ──film_scale: Linear(8→768)──►  scale (768)
       ──film_shift:  Linear(8→768)──►  shift (768)

hidden (768) × (1 + scale) + shift  ──►  modulated hidden (768)
```

## Why it matters here

> **FiLM is the only route by which the state influences the loss.**

Everything else in the RDA system — the recurrence, the predictor, the
[[Prediction Error]] pathway — either feeds the state or observes it. If FiLM
were removed, the state would be a pure observer and the architecture would be
mathematically identical to the base model.

This is why the FiLM ablations produce the largest BPB swings in
[[E15_ABLATION]] (zeroing FiLM: 2.990616 versus 1.137101 for B' on the same
validation batches).

## How it works

### Zero-initialisation

```python
for layer in (self.film_scale, self.film_shift):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
```

**All** of weight and bias. If only the weights were zeroed, `scale = 1 + bias ≠ 1`
and `modulate()` would not be an exact identity at initialisation.

Zero-init buys a specific and heavily-relied-upon guarantee:

> Since `∂L/∂E ∝ W_scale = 0` at init, the state receives **exactly zero**
> gradient at step 0, in every dimension.

Measured with a standalone `retain_grad` test: gradient is not `None`,
absmax exactly `0.0`, nonzero-frac `0.0000`. That is correct, not broken — and
the [[Control Arm]] is built on top of it.

### Per-position, not per-sequence

`modulate()` accepts `E` of shape `(B, T, 8)` — one state per position.

An earlier version accepted `(B, 8)` and broadcast it across positions, which
meant **position 0 was modulated by a state computed from the last token** — a
1022-token lookahead at T=1024. It also made the architecture impossible to
reproduce at inference, where an autoregressive step has no "final state".

### The RMS budget

```python
def _unit_rms(v, rms):
    return v / v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * rms
```

The FiLM gain ran away in early runs: `|1 + film_scale(E)|` grew `1.00 → 2.23`
with the base trainable, and `1.00 → 9.57` with the base **frozen**.

Two ablations pinned the cause on the gain term and not on the PE projection:

- **FiLM disabled entirely:** the state never crosses `|E| = 0.98`, even though
  PE still grows to 19.6. So PE growth is a *consequence* of FiLM, not its cause.
- **Freezing `pe_proj`:** saturation still occurs, at the same step.
- **Detaching PE's gradient:** unchanged. So this is not a differentiable-loop
  pathology.

With a frozen base there is **no varying external input at all**, yet the loop
still ran away. That is the definition of self-excitation: the state modulates
the residual stream, which changes the state's own future inputs, which
modulates harder.

The RMS budget removes the one degree of freedom that lets the loop amplify
itself — the overall *magnitude* of the gain — while keeping the part that
carries information: its **direction** across the `d_model` axis.

`film_rms = 0.5` is fixed and non-learnable, so it can never be traded against
the loss.

## Project usage

`modulate()` is a **method**, so `register_forward_hook` does not catch it — a
probe must wrap the bound method on the instance and return the original output
unchanged.

FiLM receives `film_state` (shape `(B,T,8)`), **not** `E_next_seq`
(`(B,T-1,8)`). They differ by a one-position shift:

```python
film_state = torch.cat([E_start.unsqueeze(1), E_next_seq], dim=1)
```

Position `p` is modulated by the state *before* the update that consumed
position `p-1`. This shift is load-bearing for inference, not just causality.

In the [[Control Arm]] the FiLM input is replaced by a learned constant, so the
modulation still happens but carries no state information.

## Related concepts

- [[RDA and Persistent State]] — what FiLM modulates
- [[Residual Stream]] — the tensor being modulated
- [[Label Leakage]] — how state reached logits by mistake
- [[Ablation Study]] — the FiLM ablations and what they proved
- [[Control Arm]] — arm C keeps FiLM but removes its input
- [[State Saturation]] — the rail FiLM drove the state onto

## Evidence / references

- [[DECISIONS]] — Decision 1 (per-position causal FiLM), Decision 3 (RMS budget)
- [[CURRENT_STATE]] — conditioning pathway, FiLM state alignment
- [[E15_ABLATION]] — per-dimension FiLM ablations
- [[EXPERIMENT_LOG]] — E10, the C1–C6 ablation suite
- [[RDA_ARCHITECTURE_GUIDE]] — Section 4
