# Ablation Study

## What it is

An **ablation** removes or disables one component of a system and measures the
result. It answers: "what was that part actually doing?"

It differs from a [[Control Arm]] in scope. An ablation asks about *one
component*; a control arm is designed to isolate *one variable* while holding
everything else fixed.

## Why it matters here

RDA has several components whose contribution is not obvious from the code
reading. Ablations turned those assumptions into measurements — and twice
inverted the project's understanding.

## The ablations run

### Pathway ablations (C1–C6)

Post-leak-fix, 24 identical validation batches:

| configuration | BPB | reading |
|---|---|---|
| baseline | 1.242340 | reference |
| B' | 1.137101 | full system |
| [[FiLM Modulation]] zeroed | 2.990616 | **worse than baseline** |
| [[Prediction Error]] zeroed | 4.371596 | **far worse than baseline** |
| [[Event Signal]] zeroed | 1.596395 | worse, but less so |

Two results here are decisive and were not expected:

**Zeroing FiLM is worse than having no RDA at all.** The conditioning pathway
is not a refinement — it is load-bearing.

**Zeroing the PE pathway is worse still.** The [[Prediction Error]] is the
single most important input to the system.

### FiLM-disabled isolation

When FiLM was disabled entirely, the state **never crossed |E| = 0.98**, even
though PE still grew to 19.6.

This established a causal direction that was previously assumed backwards:

> PE growth is a *consequence* of FiLM, not its cause.

It also showed the saturation mechanism: with the FiLM gain free, the loop is
self-exciting — the state modulates the residual stream, which changes the
state's own future inputs, which modulates harder.

Two further ablations confirmed it was the gain and not the PE pathway:
freezing `pe_proj` left saturation occurring at the same step, and detaching
PE's gradient changed nothing.

### Per-dimension ablation (E15)

The most detailed ablation: zero each of the 8 drives individually and measure
the BPB delta, giving a signed per-dimension importance.

Two bugs were found before the run could be trusted:

- the wrapper returned the *original* `E_t` rather than the modified one, so
  every delta would have been exactly 0.0;
- a sign-convention error would have reported "positive tail" for values that
  were negative.

The lesson recorded there is general: **`*_absmax` fields are unsigned.** An
absmax cannot show which bound a value is pinned against. A drive collapsed to
the lower bound and one collapsed to the upper bound have the same absmax.

## What ablations cannot tell you

An ablation measures a component's contribution **given everything else stays**.
It cannot answer whether the component is necessary in a different
configuration.

This is exactly the confusion behind two retracted counterfactuals:

- **"PE off via `targets=None`"** (4.29 BPB) — trains FiLM in a state
  distribution it never sees, so the number measures distribution shift rather
  than the PE contribution.
- **"Shuffle `y` and re-score"** — the baseline also moves (+1.748), because
  scoring wrong answers is simply harder. It confounds the cheat with
  difficulty.

> An ablation's baseline arm must also be affected by the manipulation, or the
> comparison is confounded.

## Related concepts

- [[Control Arm]] — the scope difference
- [[BPB]] — the metric every ablation reports
- [[FiLM Modulation]] — the component whose ablation was most surprising
- [[Prediction Error]] — the component whose ablation was largest
- [[Label Leakage]] — invalidated an entire generation of ablations
- [[State Saturation]] — what the FiLM-disabled ablation revealed

## Evidence / references

- [[E15_ABLATION]] — the per-dimension run, with its two pre-run bugs
- [[EXPERIMENT_LOG]] — E10: the C1–C6 suite
- [[DECISIONS]] — the decisions each ablation motivated
- [[FAILED_EXPERIMENTS]] — the counterfactuals that were not valid
