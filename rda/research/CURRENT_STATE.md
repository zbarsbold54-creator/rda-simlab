# CURRENT STATE

Factual snapshot of the RDA project. Every number here was measured on the
production code in `nanochat/rda.py`; nothing is aspirational.

Last verified: 2026-09-28, at commit `ab9b560`.

---

## Project

RDA (Regulatory Drive Architecture) is a research fork of Karpathy's
[nanochat](https://github.com/karpathy/nanochat) investigating whether a
**persistent low-dimensional internal state** can improve a small language
model.

The internal state `E` consists of **8 dimensions**:

| # | Dimension | Intended role |
|---|---|---|
| 0 | `novelty` | how much the input departs from what was expected |
| 1 | `uncertainty` | ambiguity of the current prediction |
| 2 | `threat` | predicted cost / negative valence |
| 3 | `integrity` | violation of a constraint or expectation |
| 4 | `agency` | degree of control the model has over the outcome |
| 5 | `learning_progress` | evidence the model is getting better at this |
| 6 | `arousal` | overall activation intensity |
| 7 | `homeostatic_dev` | drift from the model's settled baseline |

> **These are primitive state dimensions, NOT supervised emotion labels.**
> Nothing in the training objective references emotion, sentiment, or any
> affective dataset. The names describe the *function each coordinate is
> meant to acquire*, so that its learned structure can later be inspected.
> No claim is made — and none has been tested — that these coordinates
> correspond to human emotion categories. They are slots with motivated
> names, and naming them is a hypothesis, not a result.

---

## Architecture

Three pathways, all verified against `nanochat/rda.py`.

### Event pathway

```
hidden state (unmodulated, pre-outcome)
    ↓
LayerNorm                      (h_norm, d_model)
    ↓
event projection               (event_proj: Linear(d_model -> 8))
    ↓
event output normalization     (unit RMS over the 8 drive dims)
    ↓
× event_drive_scale (0.1)
    ↓
state update
```

The event pathway reads the **unmodulated** residual stream. This is
structural: `drive_seq(pre_outcome_hidden, pe_seq)` is called *before*
`modulate()`, so the state can never see its own downstream effect through
the event term.

### PE pathway

```
prediction error
    (NLL of the observed token, from the unmodulated logits)
    ↓
÷ pe_reference_scale (10.0)
    ↓
PE projection                  (pe_proj: Linear(1 -> 8))
    ↓
state update
```

### Conditioning pathway

```
state E  (per position)
    ↓
FiLM   film_scale: Linear(8 -> d_model)
       film_shift: Linear(8 -> d_model)
    ↓  each normalized to a fixed RMS budget (film_rms = 0.5)
transformer residual stream
    x * (1 + film_scale(E)) + film_shift(E)
```

### Recurrent state equation, as implemented

```python
ev   = event_proj(h_norm(hidden))
ev   = ev / rms(ev)                       # unit RMS over 8 drives
drive = tanh( 0.1 * ev  +  pe_proj(pe / 10.0) )
E_next = 0.95 * E_prev + 0.05 * drive
```

Defaults: `decay = 0.95`, `event_drive_scale = 0.1`,
`pe_reference_scale = 10.0`, `film_rms = 0.5`.

Because the retained state and the fresh drive are convex-combined with
weights 0.95 and 0.05, and the drive is a `tanh`, `|E| <= 1` holds **by
construction** rather than by clamping.

### FiLM state alignment

```python
film_state = cat([E_start.unsqueeze(1), E_next_seq], dim=1)
```

Position 0 is modulated by the state entering the step. Position `p` is
modulated by `E_next_seq[p-1]`, i.e. the state *before* the update that
consumed position `p-1`. The tempting alternative, `E_next_seq[p]`, is a
one-token lookahead and is deliberately not used.

---

## Causal fixes

Two temporal leaks were identified and fixed, in commit `bc3deb8`
("Fix two temporal leaks in RDA FiLM, and make the model decodable"):

1. **FiLM leaked the future into position 0.** `modulate()` took a single
   `(B, n_drives)` state and broadcast it across all positions, so position
   0 was modulated by a state computed from the *last* token — a 1022-token
   lookahead at T=1024. It also made the architecture impossible to
   reproduce autoregressively, since a decode step has no "final state".
   Fixed by making the state per-position.

2. **PE introduced a one-token lookahead.** `E_next_seq[p]` is built from
   `pe_seq[p]`, which is the model's error for token `p+1`. Modulating
   position `p` with it let position `p`'s hidden state — and therefore the
   logits that predict token `p+1` — depend on token `p+1`. Fixed by
   shifting the FiLM state by one position, as shown above.

---

## Baseline

The established 3-seed baseline (A), from the unmodified nanochat path:

| Seed | val_bpb |
|---|---|
| 42 | 1.301217 |
| 43 | 1.287544 |
| 44 | 1.295109 |

```
mean   = 1.294623
spread = 0.0137      (max - min)
std    ≈ 0.0069
```

An earlier RDA run reported **1.286248** and was described as an
improvement. **That claim was withdrawn** (commit `8d09ecf`). Two reasons:

1. It was a **single seed**, and 1.286248 falls **inside** the baseline's own
   seed range (1.287544–1.301217), so it is not distinguishable from a
   baseline draw.
2. The run that produced it was **leaky**, predating the fixes in `bc3deb8`.

> **This is not a result. It is a retracted claim, recorded so that it is
> not rediscovered and re-reported.**

---

## Event instability

**Mechanism, as demonstrated.** LayerNorm normalized the *input* to
`event_proj`, but it did not constrain the learned *output* magnitude:
`||h_norm|| = sqrt(d) = 27.71` is a **ceiling, not a value**, and
`event_proj(h)` can still reach `||w_row|| * 27.71` where `||w_row||` is
learned and unbounded.

Measured over 30 real steps (seed 42):

| quantity | start | end | factor |
|---|---|---|---|
| LayerNorm output rms | 1.0000 | 1.0000 | ×1.00 |
| LayerNorm weight norm | 27.7128 | 27.7128 | ×1.00 |
| `event_proj` ‖W‖ | 1.6317 | 3.5092 | ×2.15 |
| event \|max\| | 0.2508 | 1.2612 | ×5.03 |

LayerNorm was working perfectly. The **projection** grew.

**Hidden-state drift was ruled out as the primary cause.** Freezing the
entire base model — making `h` byte-identical every step, with
`grad||base||` exactly 0 — changed nothing: `||W||` still went
1.6317 → 3.5092 and `event` still reached 1.2612, matching the unfrozen run
to 4 decimal places. The cause is the gradient on `event_proj` itself.

**Stabilization:** RMS-normalize the event projection **output** over the
drive axis, so the drive depends on the direction of `W·h_norm` rather than
its magnitude. Verified over 250 steps: `||W||` now *shrinks* (1.63 → 0.358)
because a global scale on `W` became a no-op and only its direction is
learned.

---

## PE/FiLM instability

A **separate** instability was found. With the event term controlled, the
state still saturated.

### Verified dependency graph

```
loss
  ↓
state / PE pathway
  ↓
FiLM
  ↓
modulated hidden representation
  ↓
loss
```

This cycle is **closed**, and `diagnostics/map_pe_film_graph.py` verifies
each edge against the real graph.

### The gradient-dormancy detail

The loop is closed but **gradient-dormant at step 0**, because FiLM is
zero-initialized and

```
dL/dE  =  film_scale.weightᵀ · (dL/dhidden_mod · hidden) + film_shift.weightᵀ · dL/dhidden_mod
```

is proportional to film weights, which are exactly 0 at initialization.
Measured: `grad(pe_proj) = grad(event_proj) = grad(h_norm) = 0.000e+00` at
step 0, and nonzero from step 1. The runaway is therefore an **onset-time**
phenomenon, not a steady state — which is why it could not be diagnosed from
any single step.

### Ablations (250 steps, seed 42, production code)

| Control | first \|E\|>0.98 | max\|E\| | `pe_proj.bias` final | FiLM gain max |
|---|---|---|---|---|
| C1 reference | 217 | 0.9853 | 1.3900 (*shrinks*) | 2.227 |
| C2 no-FiLM | **never** | 0.9390 | 1.5149 | 1.000 |
| C3 freeze base | 82 | 0.9989 | 3.7130 (*grows*) | 9.568 |
| C4 freeze `pe_proj` | 217 | 0.9922 | 1.5149 | 2.283 |
| C5 detach PE grad | 217 | 0.9853 | 1.3896 | 2.238 |

Conclusions actually supported:

- **PE projection drift is NOT the cause.** `pe_proj.bias` *shrinks* in C1,
  and C4 (frozen) still saturates at the same step.
- **It is not a differentiable-loop pathology.** C5, which detaches PE's
  gradient while keeping its numerical value, gives max\|E\| 0.9853 and the
  same crossing step — indistinguishable.
- **FiLM amplification IS the cause.** C2 never saturates. PE still grows
  to 19.6 without FiLM, so PE growth is a *consequence*, not a cause.
- C3 is the decisive control: with the base frozen there is **no varying
  input at all** (PE flat at 10.50 for 250 steps), yet `pe_proj.bias` grows
  1.53 → 3.71 and the FiLM gain grows 1.00 → 9.57. A loop that runs with
  constant input is self-exciting.

### Causality

The PE signal **remained causal** throughout. Verified: `PE == NLL(logits[:T-1],
targets[1:])` to 9.5e-07, and perturbing `targets[:, k]` leaves every state
position `< k` bit-identical.

### Stabilization chosen

Per-position **RMS normalization of the FiLM modulation vector** to a fixed
budget (`film_rms = 0.5`):

```python
scale = 1.0 + unit_rms(film_scale(E), 0.5)
shift =      unit_rms(film_shift(E), 0.5)
```

- **Direction preservation verified: cosine = 1.000000000000** per position.
  Only the magnitude — the one degree of freedom that let the loop amplify
  itself — is constrained.
- **Initialization remains an exact identity**, because zero FiLM weights
  remain zero after normalization. The "B′ starts equal to A" property is
  preserved.
- It is a reparameterization, **not** a clamp: no dead zone, no gradient
  flattening.

---

## Current stabilization evidence

Production-code 250-step result, seed 42:

| | Before | After |
|---|---|---|
| first \|E\| > 0.98 | step **217** | **never** |
| max \|E\| | 0.9853 | **0.9402** |
| FiLM gain max | 2.227 | **1.522** |
| loss | 5.805 | **5.808** |

The loss is **unchanged** (5.808 vs 5.805), which indicates the fix did not
close off a loss shortcut — the state pathway is still doing work.

### Limitations of this evidence

- **One seed** (42).
- **250 steps.** Saturation previously began at step 217, so 250 steps is
  only ~15% past onset. This is the single biggest weakness in the claim.
- **Toy-scale configuration** (depth 8, d_model 768), not the real B′ config.
- **1000-step verification remains outstanding.**
- **PE still grows** to ~36 (from 10.5) and remains a topic for
  interpretation, even though it no longer drives state saturation in the
  current stabilized configuration.

> ### 3-seed B′ A/B comparison has NOT yet been performed.
>
> The gate is to demonstrate through causal ablations and production-code
> diagnostics that the PE/FiLM saturation mechanism has been **controlled
> rather than merely delayed**. The ablations support "controlled"; the
> horizon does not yet rule out "delayed". The 1000-step run must come
> first.

---

## Related Concepts

The architecture described above is decomposed into concept notes, one per
mechanism:

- [[RDA and Persistent State]] — the system this file documents; the 8
  dimensions and all three pathways
- [[Event Signal]] — the hidden-state drive pathway
- [[Prediction Error]] — the outcome drive pathway, and the leak surface
- [[FiLM Modulation]] — the conditioning pathway, and the only state-to-loss route
- [[Residual Stream]] — the tensor the event pathway reads and FiLM writes
- [[State Saturation]] — why the recurrence is shaped as a convex combination
- [[State Prediction]] — the predictor, and why its output is unused
- [[Prediction vs Actual State]] — the `dpe` signal, distinct from prediction error
- [[Online Learning]] — decay-only persistence across documents
- [[Optimizer State]] — param groups, RDA enrolment, and the grad-free subtlety
- [[Weight Decay]] — which acts even on grad-free parameters

Measurement and experiment:

- [[BPB]] — the metric every number here is read in
- [[Ablation Study]] — the per-dimension and pathway ablations
- [[Control Arm]] — the arm C comparison
- [[Label Leakage]] — the bug that required the causal fixes above

Learning material:

- [[RDA_ARCHITECTURE_GUIDE]] — full walkthrough with diagrams
- [[Python for ML]] — the Python needed to read this code
- [[DECISIONS]] — the decisions behind each design choice
- [[FAILED_EXPERIMENTS]] — what went wrong and why
