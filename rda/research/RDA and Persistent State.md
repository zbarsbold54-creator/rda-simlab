# RDA and Persistent State

## What it is

**RDA (Regulatory Drive Architecture)** is this project's research fork of
Karpathy's nanochat. It wraps a standard GPT with a small **recurrent state**
vector — 8 numbers — that persists across positions and across training steps,
and modulates the model's computation through that state.

The base model is untouched. RDA is a wrapper: `GPTWithRDA` holds a base `GPT`
and adds a state module, a predictor, and a FiLM modulation path.

The idea in one sentence: **give the model a small persistent internal state
that carries information across positions, and let that state influence
computation.**

## Why it matters here

A vanilla transformer is **stateless between tokens**. Every position is
computed from scratch using only what came before it. RDA adds memory: the
state vector summarises "what kind of situation is this" and modulates the
residual stream accordingly.

The plausible mechanism is **contextual adaptation**. Rather than recomputing
an appropriate transformation at every position from scratch, the model
amortizes it through a recurrent state — much as a biological system amortizes
adaptation across a lifetime rather than relearning it moment to moment.

The honest caveat: the drive inputs are almost entirely functions of the
[[Prediction Error]], which is itself a function of the labels. So the
"situational" content is largely "how surprised has the model recently been".
That is a *legitimate* signal for a language model, but narrower than the
affective framing suggests — and it is a direct path from label to state, which
is exactly what caused the [[Label Leakage]] documented in [[FAILED_EXPERIMENTS]].

## How it works

Three pathways, all verified against `nanochat/rda.py`.

### The 8 state dimensions

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

> These are primitive state dimensions, **NOT supervised emotion labels**.
> Nothing in the training objective references emotion, sentiment, or any
> affective dataset. The names describe *the function each coordinate is meant
> to acquire*, so its learned structure can later be inspected. No claim is
> made — and none has been tested — that these coordinates correspond to human
> emotion categories. **They are slots with motivated names, and naming them is
> a hypothesis, not a result.**

### The three pathways

```
EVENT PATHWAY                          PE PATHWAY
hidden (unmodulated)                   prediction error
    ↓                                      ↓
LayerNorm (h_norm)                     ÷ pe_reference_scale (10.0)
    ↓                                      ↓
event_proj: 768 → 8                    pe_proj: 1 → 8
    ↓                                      ↓
unit-RMS normalize                     ─────┐
    ↓                                   │
× event_drive_scale (0.1)              │
    ↓                                   ↓
    └──────────► drive = tanh(ev + pe) ◄──┘
                        ↓
              E_next = 0.95·E_prev + 0.05·drive
                        ↓
                  FiLM modulation
                        ↓
              x·(1 + scale) + shift
                        ↓
                  LM head → loss
```

The **conditioning pathway** reads the per-position state and produces a gain
and an offset for every one of the 768 residual-stream units.

### The recurrence, as implemented

```python
ev    = event_proj(h_norm(hidden))
ev    = ev / rms(ev)                       # unit RMS over the 8 drives
drive = tanh(0.1 * ev + pe_proj(pe / 10.0))
E_next = 0.95 * E_prev + 0.05 * drive
```

Because the retained state and the fresh drive are **convex-combined** with
weights 0.95 and 0.05, and the drive is a `tanh`, the bound `|E| ≤ 1` holds **by
construction** rather than by clamping. This is the single most important
architectural decision in the project — see [[State Saturation]].

The event pathway reads the **unmodulated** residual stream. That is
structural, not incidental: `drive_seq(...)` is called *before* `modulate()`, so
the state can never see its own downstream effect through the event term.

## Project usage

`GPTWithRDA` wraps a base `GPT`. State flows across micro-steps within an
episode under **decay-only persistence** — the state is never hard-reset at a
document boundary, and the `decay` term is the only mechanism limiting how much
prior state survives into a new document.

Parameter cost: +27,760 on a 207,618,378-parameter base, i.e. **+0.0134%**.

The state influences the model **only** through [[FiLM Modulation]]. The
[[State Prediction]] path and its error signal are computed and logged but
never consumed.

## Open questions

Does the state information actually help? That is what the [[Control Arm]]
exists to answer, and it is unresolved. See [[RESEARCH_QUESTIONS]] for the
primary question and [[HYPOTHESES]] for H1.

## Related concepts

- [[Residual Stream]] — what FiLM modulates
- [[FiLM Modulation]] — the only route from state to loss
- [[Event Signal]] — one of the two drive inputs
- [[Prediction Error]] — the other drive input, and the leak surface
- [[State Saturation]] — why the update rule is shaped as it is
- [[Control Arm]] — the experiment that tests whether the state matters
- [[BPB]] — the metric everything is measured in

## Evidence / references

- [[RDA_ARCHITECTURE_GUIDE]] — full architecture walkthrough
- [[CURRENT_STATE]] — verified description of all three pathways
- [[DECISIONS]] — six architectural decisions with evidence
- [[RESEARCH_QUESTIONS]] — the primary question
