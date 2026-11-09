# RESEARCH QUESTIONS

The scientific questions this project is trying to answer. No question
below is claimed to be answered unless `EXPERIMENT_LOG.md` records the
experiment that answers it.

---

## Primary question

> **Does a persistent low-dimensional internal state, generated from
> primitive event and prediction-error signals and coupled back into a
> language model, improve model behavior compared with a causal baseline?**

"Improve model behavior" is deliberately open. It could mean validation
bits-per-byte, but it could equally mean adaptation, calibration, or
anomaly sensitivity. The baseline for comparison is the unmodified
nanochat path, which is a strictly causal model with no persistent state
beyond its own KV cache.

This question has **not** been answered. The one earlier run that appeared
to answer it affirmatively was withdrawn: it was a single seed whose value
(1.286248) fell inside the baseline's own 3-seed range
(1.287544–1.301217), and it predated the causality fixes.

---

## Secondary questions

**Q1. Do the 8 state dimensions acquire stable, interpretable structure?**

Do the coordinates differentiate into distinguishable roles, or do they
collapse toward redundant directions? Interpretation would require either
correlating the coordinates against measurable quantities (entropy of the
next token, perplexity change, presence of a document boundary) or showing
that ablating one coordinate damages a specific behaviour. **Untested.**

**Q2. Does persistent state provide useful information beyond ordinary
transformer context?**

The state is a function of the hidden stream, so in principle a
sufficiently wide transformer could compute anything the state contains.
The interesting question is whether a *low-dimensional, persistent,
decaying* summary adds anything — for example across document boundaries,
where the context window has been exhausted but the state has not.
**Untested.**

**Q3. Does prediction error provide a useful internal state signal?**

PE is the model's own surprise at the token it just consumed. It is
available at inference time for the prompt but not for the token being
generated. Is it informative, or merely a restatement of loss?
**Untested.** Note that PE grows substantially during training (10.5 →
~36 over 250 steps) and this growth is not yet understood; see
`CURRENT_STATE.md`.

**Q4. Does self-prediction of future state (`dpe`) provide useful
information?**

`RDAPredictor` attempts to predict `E_{t+1}` from `E_t` and a pre-outcome
hidden state; `dpe` is the error of that prediction. A low `dpe` would
mean the state is predictable, i.e. the recurrence has found structure
rather than noise. **Untested, and deliberately not yet consumed** — `dpe`
is computed and logged but does not influence anything.

**Q5. Does state-conditioned modulation improve adaptation?**

FiLM is currently the only route by which the state can affect the loss.
Does conditioning the residual stream on a persistent low-dimensional state
help more than the architecture's own context would? **Untested.**

**Q6. Can sparse or MoE-like state controllers improve conditional
specialization later?**

A future extension: route different drive dimensions to different experts.
Explicitly **out of scope** for the current experiment — see
`DECISIONS.md`, Decision 6.

---

## Questions that arose from failure analysis

These were not planned; they emerged from instability work and are
genuinely open.

**Q7. Why does a bounded, self-limiting modulation still self-excite?**

The FiLM runaway occurred with the base model frozen — no varying input,
yet the gain grew 1.00 → 9.57 over 250 steps. What property of a
positive-feedback architecture allows this, and are there other loops in
this model with the same structure? **Partially answered** (the gain was
the amplifying degree of freedom; the fix bounds it), but the general
question is open.

**Q8. Why does PE grow during training?**

PE rose from 10.5 to ~36 in the stabilized configuration. This is a
symptom of something, but it is not yet diagnosed. It is plausible that
it reflects the model becoming more confident on some tokens and much
less confident on others. **Open.**

---

## What would count as answering the primary question

A valid answer requires:

1. A multi-seed B′ (RDA) run on the stabilized architecture.
2. A matching multi-seed baseline, or reuse of the existing 3-seed A
   baseline provided the training configuration is identical.
3. A report of the **spread**, not just the mean, since the earlier
   failure was precisely a case of a difference smaller than seed noise.
4. Confirmation from the 1000-step diagnostic that the state does not
   saturate over the horizon actually used in training.
5. Evidence that B′ differs from A in the hypothesized *direction*, not
   merely by a number that happens to be smaller.

---

## Related Concepts

The primary question is about whether [[RDA and Persistent State]] carries
information the base model does not already have.

- [[Control Arm]] — the experiment designed to answer it, and its three
  preconditions
- [[Prediction Error]] — the signal that makes the state non-arbitrary
- [[BPB]] — the metric the answer would be read in
- [[Ablation Study]] — the related but weaker method
- [[State Saturation]] — whether the state stays informative over the horizon
- [[Anomaly Evaluation]] — a secondary question about unusual input
