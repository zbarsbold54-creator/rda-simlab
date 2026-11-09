# Prediction Error

## What it is

**Prediction error (PE)** is how *surprised* a model was by the token that
actually arrived. Formally it is the **negative log-likelihood** the model
assigned to the observed token:

```
PE = -log( probability the model gave to the token that actually appeared )
```

If the model was confident and right, PE is small. If the model was surprised,
PE is large.

Three properties that matter enormously for this project:

1. **PE is always positive.** The log of a probability is never positive, so
   `-log(p)` is never negative. PE cannot express "negative surprise".
2. **PE is O(10) at initialisation.** A random token out of a large vocabulary
   has probability around `1/50000`, and `-log(1/50000) ≈ 10.8`.
3. **PE is computed from labels.** This is the mechanism behind
   [[Label Leakage]].

## Why it matters here

PE is **one of the two inputs** that drive the [[RDA and Persistent State]]. It
is the project's primary internal signal and the reason the state is not
arbitrary.

But PE's three properties above caused two separate crises, both documented in
[[FAILED_EXPERIMENTS]] and [[EXPERIMENT_LOG]].

### Crisis 1 — the sign could never change

Fed directly into `pe_proj`, a token-level NLL is ~10 and always positive. That
produced a PE term roughly **100× larger** than the [[Event Signal]], with a
**sign fixed forever**. The drive therefore had a near-constant sign and the
state was pinned at a rail — it could go one way and never come back.

The fix was to normalise by `pe_reference_scale = 10.0` before projection:

```python
pe_scaled = pe / self.pe_reference_scale
```

This makes the input O(1) and zero-centred, so it can actually **oppose** the
event drive instead of overwhelming it. The reference scale is a property of the
NLL definition (≈ ln(vocab_size)), not a tuned parameter.

### Crisis 2 — PE reached the loss through FiLM

PE feeds the state, the state modulates the residual stream via
[[FiLM Modulation]], and the modulated stream produces the logits that are
scored. That makes PE a **path from the label to the loss**.

The index was wrong for a long time. The fix is one character —
`targets[:, :-1]` instead of `targets[:, 1:]` — and it is what made a valid A/B
possible at all. See [[Label Leakage]].

## How it works in the project

```python
pe_seq = self.state.prediction_error_from_logits(
    _safe_logits(self.base, hidden, pre_outcome_hidden), targets[:, :-1]
)
```

The `targets[:, :-1]` slice is the **entire leak surface**. Because
`y[p] == x[p+1]` exactly and `gpt.py` scores with no shift, `targets[p]` is the
label attached to position `p`'s logits. Slicing with `:-1` means `pe_seq[j]`
scores `y[j]`, which is what the causal argument requires.

## The projector is untrained

The [[State Prediction]] path predicts the next state from the current state.
The resulting error (`dpe`) is computed, returned and logged — and **never
consumed**. Per [[DECISIONS]] Decision 4, this is deliberate: `dpe` is not yet
allowed to modulate memory, learning rate, or attention.

Because nothing in the loss depends on the predictor, its four tensors receive
**no gradient at all** in every arm. They are listed as grad-free purely so
`MuonAdamW._compute_adamw` does not crash calling `grad.float()` on `None`
(`optim.py:73`). They are still trained by nothing.

## Related concepts

- [[Event Signal]] — the other drive input, which PE overwhelmed
- [[RDA and Persistent State]] — what PE drives
- [[State Prediction]] — predicting the state from the state
- [[Prediction vs Actual State]] — the error between them
- [[Label Leakage]] — how PE reached the loss by mistake
- [[FiLM Modulation]] — the path PE took to the loss

## Evidence / references

- [[CURRENT_STATE]] — PE pathway as verified in code
- [[DECISIONS]] — Decision 4: do not consume `dpe` yet
- [[FAILED_EXPERIMENTS]] — F2, the two temporal leaks
- [[EXPERIMENT_LOG]] — E3, E14: the leak fix and audit
- [[RDA_ARCHITECTURE_GUIDE]] — Section 8, causality worked through
