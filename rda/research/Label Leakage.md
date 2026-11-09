# Label Leakage

## What it is

**Label leakage** is when information that should not be available at
prediction time reaches the model's output anyway — because the target itself
snuck into the computation that produced the prediction.

A language model scoring position `p` must never see the token it is being
graded on. If it does, it can cheat, and every measurement after that is
meaningless.

## Why it matters here

This project found and fixed a real label leak, and it **invalidated more than
twelve experiments**. Roughly a third of all retractions in
[[FAILED_EXPERIMENTS]] trace back to it.

The failure was subtle because the code was *right* and the *reasoning* was
wrong.

## How the leak worked

### The setup

The dataloader packs multiple documents per row and slides two views:

```python
y = x[1:]      # so y[p] == x[p+1] EXACTLY, within a row
```

`gpt.py:532` scores with **no shift**:

```python
cross_entropy(logits[p], targets[p])
```

Therefore `y[p]` is the label attached to position `p`'s logits. Verified:
fraction exactly `1.000000`.

### The requirement

For position `p`'s logits not to depend on its own label:

```
film_state[p] may depend on y[j] only for j < p
```

Since `y[j] = x[j+1]`, this is equivalent to saying `film_state[p]` may depend
on `x[0..p]` and nothing beyond — exactly what a causal position is entitled to.

### The bug

The prediction error was built from `targets[:, 1:]`:

```python
pe_seq = ... prediction_error_from_logits(logits, targets[:, 1:])
```

So `pe_seq[j]` scored `y[j+1] = x[j+2]`. Then:

```
E_next_seq[j] depends on x[j+2]
film_state[p] = E_next_seq[p-1] depends on x[p+1]
logits[p] is graded against targets[p] = x[p+1]   ← the leak
```

**The logits at position `p` were built from the label they are scored on.**

### Why the original reasoning was wrong

The comment justifying `targets[:, 1:]` read, in essence:

> "`E_next_seq[p-1]` depends on `pe_seq[:, p-1]`, which depends on token `p` —
> strictly causal."

Both halves were correct about the **mechanism** and wrong about the
**arithmetic**. `pe_seq[:, p-1]` did depend on `targets[p]` — but
`targets[p]` **is** `x[p+1]`, not `x[p]`.

The reasoning verified the index and then mislabelled what lives at that index.
This is the single most transferable lesson in the project's failure record.

### The fix

One character:

```python
pe_seq = ... prediction_error_from_logits(logits, targets[:, :-1])
```

Now `pe_seq[j]` scores `y[j]`, so `film_state[p]` depends on
`y[0..p-1] = x[1..p] ⊆ x[0..p]`. Satisfied.

## Dynamic proof

Perturbing `y[:,k]` alone:

- hidden and logits at `p < k`: change **exactly 0**
- hidden and logits at `p ≥ k`: all move (cumulative recurrence)

`max|dE|` at `p < k` ≈ 4.5e-2

## Two counterfactuals that were also invalid

**"Shuffle `y` and re-score."** Invalid: the *baseline* also moves (+1.748),
because scoring wrong answers is simply harder. It confounds the cheat with
difficulty rather than isolating the cheat.

**"PE off via `targets=None`"** (4.29 BPB). Invalid counterfactual: it trains
FiLM in an unseen state distribution, so the number measures distribution shift
rather than the leak.

## The permanent gate

`diagnostics/test_no_label_leak.py` fails on the buggy version and passes on
the fixed one. `run_ab_v3.ps1` runs it in PRE-FLIGHT before any sweep starts.

> A leak found after twelve runs costs far more than one second of preflight.

## What was withdrawn

| claim | status |
|---|---|
| `B'-s42 val BPB 1.226305 vs baseline 1.301217` | **retracted** |
| the 0.0749 BPB gain from E13 | **withdrawn** |
| E13-SUPERSEDED 1000-step confirmation | **measured with a leak** |

## Corrected measurements, after the fix

24 identical validation batches:

| configuration | BPB |
|---|---|
| baseline (seed 42) | 1.242340 |
| B' (post-fix) | 1.137101 |
| FiLM zeroed | 2.990616 |
| PE zeroed | 4.371596 |
| event zeroed | 1.596395 |

## Related concepts

- [[Prediction Error]] — the vector the leak travelled along
- [[RDA and Persistent State]] — what the leak contaminated
- [[FiLM Modulation]] — the path from state to logits
- [[Control Arm]] — the experiment that finally became possible
- [[Ablation Study]] — how the leak was localised

## Evidence / references

- [[FAILED_EXPERIMENTS]] — F1, F2, F5, F6
- [[EXPERIMENT_LOG]] — E2, E3, E14
- [[AB_SETUP_READONLY]] — verification that both arms contain the fix
- [[RDA_ARCHITECTURE_GUIDE]] — Section 8, the full derivation
