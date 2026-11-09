# State Prediction

## What it is

The [[RDA and Persistent State]] module contains a small **predictor** that
tries to predict the next state from the current state:

```
E_hat_next = predictor(E_t, hidden_state)
```

It is a self-model: the state attempts to forecast itself.

The resulting discrepancy — `E_hat_next` versus what actually happened — is the
**drive prediction error**, `dpe`.

## Why it matters here

`dpe` is **computed, returned, and logged — and never consumed.**

> Nothing about memory, learning rate, or attention is modulated by it. The
> state influences the model **only** through [[FiLM Modulation]].

This is deliberate. Per [[DECISIONS]] Decision 4, consuming `dpe` is
postponed until B' is tested — so that any effect can be attributed to the
state pathway rather than to a second mechanism.

The predictor exists in this release as **instrumentation**: it produces the
`dpe` signal that the saturation analysis used to localise where the
`threat` rail came from.

## How it works

The predictor is applied *after* the state sequence exists, shifted by one:

```python
E_next_seq = self.state.update_seq(E_start, drive)      # (B, T-1, 8)
E_in = torch.cat([E_start.unsqueeze(1), E_next_seq[:, :-1, :]], dim=1)
E_hat_seq = self.predictor.predict_seq(E_in, pre_outcome_hidden)
dpe = drive_prediction_error(E_next_seq, E_hat_seq)
```

The prediction for position `t` is made from the state *before* the update that
consumed position `t` — which is exactly what the sequential reference loop did.

## It is untrained — in every arm

Because nothing in the loss depends on the predictor, its **four tensors
receive no gradient at all**, in arm B' as well as arm [[Control Arm|C]].

They are listed as grad-free purely so `MuonAdamW._compute_adamw` does not crash
calling `grad.float()` on `None` (`optim.py:73`). They remain **enrolled** in
the optimizer, so coverage holds and they would update if that ever changed.

This is worth stating plainly, because "the predictor is a 4-tensor module in
the model" invites the assumption that it trains. It does not. Its outputs are
diagnostic data, not learned structure.

## The measurement trap

`dpe` is logged as a mean over positions. The project has been bitten twice by
reading such summaries wrongly:

- [[FAILED_EXPERIMENTS]] F7 — a diagnostic summary that contradicted its own
  measurement.
- [[State Saturation]] — an unsigned `absmax` field read as if it were signed,
  twice.

`dpe.norm()` on a `(B, n_drives)` tensor is the **Frobenius** norm over all
`B × n` elements, so it scales as `sqrt(B)` above the per-vector ceiling
`sqrt(n_drives)`. At `B=2, n=8` the per-row ceiling is 2.828 while `.norm()`
legitimately reaches ~3.5. This was misread as a tanh bound violation. It was
not.

> Use `.norm(dim=-1)` for per-vector state norms. Plain `.norm()` is valid only
> for a single unbatched `(n_drives,)` tensor.

## Related concepts

- [[Prediction vs Actual State]] — the error this produces
- [[RDA and Persistent State]] — the thing being predicted
- [[Prediction Error]] — a different error signal, do not confuse the two
- [[FiLM Modulation]] — the *only* path from state to loss
- [[Control Arm]] — where the predictor is also untrained

## Evidence / references

- [[DECISIONS]] — Decision 4: do not consume `dpe` yet
- [[CURRENT_STATE]] — "the gradient-dormancy detail"
- [[HYPOTHESES]] — H4: `dpe` carries information about state regularity
- [[E15_ABLATION]] — `dpe` appears in the per-dimension measurements
