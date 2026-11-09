# HYPOTHESES

Hypotheses are separated from facts. A hypothesis is a claim that could be
wrong; a fact is something measured and recorded in `EXPERIMENT_LOG.md`.

Status vocabulary:
- `UNTESTED` — no experiment run
- `PARTIALLY SUPPORTED` — evidence exists but is incomplete or from a
  limited configuration
- `SUPPORTED` — evidence exists, within the stated limitations
- `REFUTED` — evidence contradicts it
- `NOT THE MECHANISM` — tested as a candidate explanation and ruled out

---

## H1 — Persistent state provides useful information beyond context

```
Hypothesis:
  A persistent, low-dimensional, slowly-decaying state carries information
  that a causal transformer does not otherwise have available, and
  conditioning the model on it improves val_bpb relative to the
  unmodified nanochat baseline.

Motivation:
  The transformer's KV cache is bounded by the context window and is
  overwritten token by token. A state that decays with a fixed coefficient
  (0.95) retains information across arbitrarily long distances at a cost of
  8 numbers. If that summary is useful, it should be reachable at low cost.

Prediction:
  If true, a 3-seed B' run will show a mean val_bpb below the baseline
  mean of 1.294623 by MORE than the observed seed spread of 0.0137.

Experiment:
  3-seed A/B on the stabilized architecture, identical configuration.

Current evidence:
  NONE SUPPORTING. The single earlier run that appeared favourable
  (1.286248) was withdrawn: single seed, inside the baseline range, and
  produced by a leaky pre-bc3deb8 model.

Status:
  UNTESTED. This is the primary question and it remains open.
```

---

## H2 — The 8 dimensions acquire stable, interpretable structure

```
Hypothesis:
  The 8 coordinates differentiate into distinguishable roles rather than
  collapsing into redundant directions, and those roles correspond to the
  functions their names describe.

Motivation:
  The dimensions are given motivated names. Those names are a hypothesis,
  not a result — nothing supervises them.

Prediction:
  If true, the per-dimension state trajectories will be (a) not perfectly
  correlated with each other, and (b) predictable from measurable
  quantities such as next-token entropy.

Experiment:
  Compute the per-dimension correlation matrix of E over a trained run;
  correlate each coordinate against next-token entropy, perplexity delta,
  and document-boundary proximity. A single-coordinate ablation study.

Current evidence:
  NONE. No interpretability experiment has been run.

Status:
  UNTESTED.
```

---

## H3 — Prediction error is a useful internal signal

```
Hypothesis:
  The model's own prediction error at the token just consumed is an
  informative, low-dimensional-compatible signal that is worth injecting
  into a persistent state.

Motivation:
  PE is available at inference time from the prompt, is model-relative
  (unlike an external reward), and captures surprise rather than content.

Prediction:
  If true, then (a) removing the PE pathway from the drive should reduce
  whatever benefit the state provides, and (b) the PE contribution to the
  drive should be comparable in magnitude to the event contribution for
  some configurations.

Experiment:
  Ablate the PE term from the drive (keep event, keep FiLM) and re-run the
  A/B. Also correlate per-dimension PE contribution against next-token
  entropy.

Current evidence:
  AMBIGUOUS AND UNEXPLAINED. During training PE grows from 10.5 to ~36 in
  the stabilized configuration. That growth is NOT understood. It is
  currently harmless to the state (the FiLM fix decoupled them), but an
  unexplained growing input to a state signal is a poor thing to leave
  unexamined before drawing conclusions about that signal's usefulness.

Status:
  UNTESTED, with a known unexplained anomaly (see CURRENT_STATE.md).
```

---

## H4 — `dpe` (drive-prediction error) carries information about state regularity

```
Hypothesis:
  The error of predicting the next drive state from the current state and a
  pre-outcome hidden state is low when the recurrence has found real
  structure, and high when the state is noise. A useful diagnostic of
  whether the state is meaningful.

Motivation:
  If E is predictable, the recurrence is tracking something systematic. If
  it is not, the 8 dimensions may be arbitrary.

Prediction:
  If true, dpe should be substantially below the scale of E itself and
  should vary across seeds in a way that correlates with downstream loss.

Experiment:
  Track mean dpe over training for A and for B'. Currently computed and
  logged, never consumed.

Current evidence:
  NONE INTERPRETED.

Status:
  UNTESTED. Deliberately not consumed yet — see DECISIONS.md, Decision 4.
```

---

## H5 — FiLM is a sufficient conditioning route for the state

```
Hypothesis:
  Per-position FiLM modulation of the residual stream is an adequate way
  for an 8-dimensional persistent state to influence the model, without
  additional mechanisms (attention modulation, LR modulation, MoE routing).

Motivation:
  FiLM is minimal, well-understood, and makes the state-to-loss path a
  single differentiable edge that is easy to audit for causality.

Prediction:
  If FiLM is sufficient, then B' should differ from A at all. If it is
  insufficient, B' should be statistically indistinguishable from A.

Experiment:
  The 3-seed A/B itself is the test.

Current evidence:
  NONE. And the FiLM path was found to be the *cause* of a saturation
  runaway, which is a property of the mechanism, not evidence about its
  sufficiency.

Status:
  UNTESTED.
```

---

## H6 — Explains the event instability (REFUTED as an explanation)

```
Hypothesis (candidate explanation, now ruled out):
  The event term grew because the base model's hidden states drifted, so
  the LayerNorm-normalized input still produced a growing projection
  output.

Motivation:
  Plausible a priori — hidden-state norms are known to grow during
  training.

Prediction:
  If true, freezing the base model (making h byte-identical every step)
  should stop the event growth.

Experiment:
  30-step run with the base model frozen; compare ‖event_proj.W‖ and
  event magnitude against the unfrozen control.

Current evidence:
  REFUTED. With the base frozen, ‖W‖ still went 1.6317 -> 3.5092 and event
  still reached 1.2612, matching the unfrozen run to 4 decimals. The cause
  is the gradient on event_proj itself, not hidden-state drift.

Status:
  NOT THE MECHANISM.
```

---

## H7 — Explains the PE/FiLM instability (FiLM confirmed, others refuted)

```
Hypothesis A (PE projection drift):
  pe_proj.weight or pe_proj.bias grows, and that growth drives the state
  to saturation.

Hypothesis B (FiLM amplification):
  The E -> FiLM transformation becomes increasingly strong, and that
  amplification is self-exciting.

Hypothesis C (differentiable feedback loop):
  Backprop through loss -> PE -> E -> FiLM -> loss creates a pathological
  optimization loop.

Motivation:
  All three were plausible from the observed numbers; the investigation
  was designed to separate them rather than assume one.

Experiment:
  250-step ablations C1-C5 through production code, seed 42.

Current evidence:
  A — REFUTED. pe_proj.bias SHRINKS in the reference run (1.5324 ->
    1.3900), and freezing pe_proj entirely still saturates at the same
    step 217.
  B — SUPPORTED. With FiLM disabled the state never crosses 0.98. PE
    still grows to 19.6 without FiLM, so PE is downstream.
  C — REFUTED. Detaching PE's gradient gives max|E| 0.9853 and the same
    crossing step, i.e. no material change.
  Decisive extra control (C3): with the base model frozen, PE is FLAT at
    10.50 for 250 steps while pe_proj.bias grows 1.53 -> 3.71 and the
    FiLM gain grows 1.00 -> 9.57. A loop that runs with constant input is
    self-exciting.

Status:
  B SUPPORTED, A and C NOT THE MECHANISM.
```

---

## H8 — FUTURE EXTENSION, explicitly out of scope

```
Hypothesis (future, not part of the current experiment):
  Routing different drive dimensions to different sparse experts (an
  MoE-like state controller) would improve conditional specialization
  relative to dense FiLM conditioning.

Motivation:
  Different drives plausibly warrant different transformations of the
  residual stream; a single dense FiLM forces one shared mapping.

Prediction:
  Speculative. Would require a routing mechanism, a load-balancing loss,
  and a capacity limit, none of which exist.

Experiment:
  NONE DESIGNED. This is a later project.

Current evidence:
  NONE.

Status:
  UNTESTED AND OUT OF SCOPE. Listed to record the direction of travel, not
  to imply it is planned. See DECISIONS.md, Decision 6.
```

---

## Related Concepts

- [[RDA and Persistent State]] — H1 and H2 are claims about this architecture
- [[Prediction Error]] — H3, the project's primary internal signal
- [[Prediction vs Actual State]] — H4, the unused `dpe` signal
- [[FiLM Modulation]] — H5, whether this is a sufficient conditioning route
- [[State Saturation]] — H6 and H7, both refuted as explanations
- [[Control Arm]] — the experiment that would test H1
- [[Online Learning]] — H8, the out-of-scope extension

Note that H2's status vocabulary separates [[HYPOTHESES]] from
[[EXPERIMENT_LOG]]: a hypothesis is a claim that could be wrong, a fact is
something measured and recorded.
