# DECISIONS

Major architectural decisions and the evidence behind them. Each records
what remains uncertain, because a decision made without that record tends
to be re-litigated from memory.

---

## Decision 1 — Use causal per-position FiLM, not future-conditioned FiLM

```
Decision:
  Make the state per-position, so position p is modulated by the state as
  it stood BEFORE the update that consumed position p. Position 0 is
  modulated by E_start.

      film_state = cat([E_start.unsqueeze(1), E_next_seq], dim=1)
      position 0 <- E_start
      position p <- E_next_seq[p-1]

Reason:
  The previous formulation broadcast one (B, n_drives) state across all
  positions, which meant position 0 was modulated by a state computed
  from the LAST token — a 1022-token lookahead at T=1024. It also made the
  architecture impossible to reproduce autoregressively: a single decode
  step has no "final state" to draw on, so the model could not be run.

Evidence:
  Commit bc3deb8. The alternative inside the training branch,
  E_next_seq[p], is a one-token lookahead, because E_next_seq[p] is built
  from pe_seq[p] which is the model's error on token p+1.

What remains uncertain:
  Nothing about correctness — the causality is verified. What is
  uncertain is whether the per-position state is the RIGHT conditioning,
  as opposed to merely a valid one. No experiment compares the two.
```

---

## Decision 2 — Normalize the event projection OUTPUT, not just its input

```
Decision:
  RMS-normalize the output of event_proj over the 8 drive axes, after the
  projection, rather than relying on the LayerNorm already applied to its
  input.

      ev   = event_proj(h_norm(hidden))
      ev   = ev / rms(ev)
      term = 0.1 * ev

Reason:
  LayerNorm pins ||h_norm|| to sqrt(d) = 27.71 — a CEILING, not a value.
  event_proj(h) can still reach ||w_row|| * 27.71, and ||w_row|| is
  LEARNED and UNBOUNDED. So the normalization was in the wrong place: it
  bounded the input magnitude but left the output magnitude free, and the
  output magnitude is what actually grew.

Evidence:
  30 steps, seed 42. LayerNorm output rms stayed at exactly 1.0000 and its
  weight norm at exactly 27.7128 for the whole run — it was working. Over
  the same run event_proj ||W|| went 1.6317 -> 3.5092 (x2.15) and event
  |max| went 0.2508 -> 1.2612 (x5.03).
  Freezing the base model changed nothing (||W|| still 3.5092, event still
  1.2612, matching to 4 decimals), which rules out hidden-state drift and
  isolates the cause to the gradient on event_proj itself.
  Over 250 steps the normalized ||W|| now SHRINKS to 0.358, because a
  global scale on W became a no-op and only its direction is learned.

  Rejected alternatives: bounding the term with tanh(k*x)/k (control C4)
  capped the contribution but ||W|| still grew x1.91, so the ratchet
  survived. Stronger decay (control C5) made it WORSE (x5.37).

What remains uncertain:
  The normalization is applied over the drive axis, so the learned
  RELATIVE scale between drives is preserved but the absolute scale is
  not. Whether the relative scale is the right thing to preserve, and
  whether some useful magnitude information is being discarded, is not
  tested. Also, an 8-vector at unit RMS has components ~0.354, so with
  event_drive_scale = 0.1 the event term is now ~0.035 — three orders of
  magnitude below the PE term. The event term may now be too small to
  matter, which would make the pathway decorative. NOT TESTED.
```

---

## Decision 3 — Normalize FiLM modulation magnitude, rather than hard-clamping the state

```
Decision:
  RMS-normalize each per-position FiLM modulation vector to a fixed budget
  (film_rms = 0.5):

      scale = 1.0 + unit_rms(film_scale(E), 0.5)
      shift =      unit_rms(film_shift(E), 0.5)

Reason:
  The FiLM gain was a free, learnable, unbounded magnitude, and it was the
  single degree of freedom through which the state could amplify itself.
  Normalizing the magnitude while keeping the direction constrains exactly
  that degree of freedom and nothing else.

  A hard clamp was explicitly rejected: clamping introduces a dead zone
  where the gradient vanishes, which would hide instability rather than
  remove it, and would make the trajectory insensitive to the fix in a way
  that is hard to interpret.

Evidence:
  250 steps, seed 42, production code.
      before: first |E|>0.98 at step 217, max|E| 0.9853, FiLM gain 2.227
      after:  first |E|>0.98 never,    max|E| 0.9402, FiLM gain 1.522
      loss:   5.805 -> 5.808 (unchanged)
  Direction preservation verified: cosine(raw, normalized) =
  1.000000000000 at every position.
  Exact identity at init verified: max |diff| = 0.00e+00, because zero
  FiLM weights remain zero after normalization. The "B' starts equal to A"
  property is preserved.
  The unchanged loss is the load-bearing detail: it shows the fix did not
  close off a loss shortcut.

  Ablation basis (see EXPERIMENT_LOG.md E10): disabling FiLM stops
  saturation entirely; freezing pe_proj does not; detaching PE's gradient
  does not. So the gain is the cause.

What remains uncertain:
  film_rms = 0.5 was chosen from two data points (0.5 and 1.0), not
  derived. At 1.0, max|E| reaches 0.9793 — under threshold, but close. The
  value is a judgement call and the margin is not characterized.
  Whether the state can still express a useful MAGNITUDE of modulation is
  not tested; only direction is learned now.
  ONE seed, 250 steps. See Decision 7.
```

---

## Decision 4 — Do not consume `dpe` yet

```
Decision:
  `dpe` — the error of RDAPredictor predicting the next drive state — is
  computed, returned and logged, and nothing consumes it. No memory, no
  learning-rate modulation, no attention modulation.

Reason:
  `dpe` is the project's main interpretability instrument: if the state is
  predictable, the recurrence has found structure; if not, the 8
  dimensions may be arbitrary. Consuming it would immediately confound
  that measurement with whatever it is being used for. It is far more
  valuable as an untouched diagnostic than as a weak training signal.

Evidence:
  Interpretive status is itself unestablished — dpe has never been
  analyzed across seeds or correlated with downstream loss.

What remains uncertain:
  Whether dpe carries any signal at all. This is recorded as open
  question Q4 in RESEARCH_QUESTIONS.md, and as hypothesis H4.
```

---

## Decision 5 — Do not modify position-0 behaviour yet

```
Decision:
  Position 0 is modulated by E_start, the state carried in from the
  previous training step (decay-only persistence). Document boundaries do
  NOT reset the state; the decay term is the only mechanism limiting how
  much prior state survives into a new document.

Reason:
  Two separate questions are entangled here — how state crosses document
  boundaries, and what position 0 should be conditioned on. Both are
  currently entangled with an unresolved stability problem. Changing
  persistence semantics before the stability question is settled would
  make any resulting measurement uninterpretable: a change in val_bpb
  could not be attributed.

Evidence:
  None directly. This is a decision to DEFER, made under uncertainty, and
  is recorded as such.

What remains uncertain:
  Whether decay-only persistence is the right choice at all. It has never
  been compared against explicit document-boundary reset. The dataloader
  packs multiple documents per row, so the cross-document behaviour is
  currently a design assumption rather than a measured choice.
```

---

## Decision 6 — Do not introduce MoE / sparse state controllers before B' is tested

```
Decision:
  Sparse or MoE-like state controllers — routing different drive
  dimensions to different experts — are explicitly out of scope. They
  appear in RESEARCH_QUESTIONS.md (Q6) and HYPOTHESES.md (H8) only to
  record the direction of travel.

Reason:
  The primary question — whether a persistent low-dimensional state helps
  at all — has not been answered. The one apparent positive result was
  withdrawn. Adding a routing mechanism, a load-balancing loss, and a
  capacity limit on top of an architecture whose basic premise is
  unestablished would produce a result that could not be interpreted: any
  effect would be confounded with the extra machinery.

Evidence:
  None. This is sequencing, not empirics.

What remains uncertain:
  Everything. No experiment has been designed. The risk being managed is
  not that the idea is wrong but that pursuing it early makes the simpler
  question unanswerable.
```

---

## Decision 7 — Gate the 3-seed A/B on a 1000-step stability run

```
Decision:
  Do not run the 3-seed B' A/B until a 1000-step production diagnostic
  shows the saturation mechanism is controlled rather than delayed.

Reason:
  Saturation in the unstabilized configuration first crossed |E| = 0.98 at
  step 217. The stabilization evidence runs to 250 steps — only about 15%
  past onset. That is not long enough to distinguish a mechanism that is
  gone from one that is merely slower. Running a multi-seed A/B on an
  architecture that might still saturate at step 400 would produce a
  number that measures implementation instability rather than the
  scientific hypothesis.

Evidence:
  The ablations support "controlled" strongly (C2 never saturates; the
  production run never crosses in 250 steps). What is missing is horizon,
  not mechanism.

What remains uncertain:
  Whether 1000 steps is itself long enough. The training runs are
  substantially longer than that, so even a clean 1000-step result leaves
  a residual risk. A stronger gate would instrument the actual training
  run and abort on saturation — the abort rule from commit 539007e already
  exists for this purpose, with RDA_DISABLE_SAT available to suppress it
  for diagnostics.
```

---

## Cross-cutting decision — diagnostics must observe, never replace

```
Decision:
  No diagnostic may reimplement a production method. Observations are made
  with forward/backward hooks on the real modules, registered and removed
  explicitly, and every run prints which implementation is active.

Reason:
  An earlier probe monkey-patched drive_seq unconditionally and silently
  shadowed a fix made to the real code, producing a byte-identical
  trajectory that read as "the fix did nothing". The same class of error
  nearly recurred. See FAILED_EXPERIMENTS.md F4 and F8.

Evidence:
  Two occurrences of the same shadowing failure, the second caught only
  because the first was written down.

What remains uncertain:
  Nothing — this one is settled by two concrete failures.
```

---

## Cross-cutting decision — treat exact equality after a change as a failure signal

```
Decision:
  When a supposedly-changed condition returns a result identical to the
  unchanged control, treat it as evidence of a broken experiment, not as
  a null result.

Reason:
  Two separate controls in this project (F4 shadowing, F5 freeze-base)
  returned digit-for-digit identical results because the experiment was
  not connected to the thing it was changing. Both were caught by
  noticing that exact equality was implausible, not by any test.

Evidence:
  F4: event 0.2715 -> 0.2690, ||W|| 1.6317 -> 2.0126, identical after a
  fix that provably changes a term's magnitude.
  F5: C3 identical to C1 in every digit because the base model was in no
  optimizer group.

What remains uncertain:
  Nothing. The heuristic is cheap and has no false-positive cost beyond
  the time spent re-checking.
```

---

## Related Concepts

- [[FiLM Modulation]] — Decision 1 (per-position causal FiLM), Decision 3 (RMS budget)
- [[Event Signal]] — Decision 2: normalise the projection output, not its input
- [[Prediction vs Actual State]] — Decision 4: do not consume `dpe` yet
- [[Residual Stream]] — Decision 2's underlying problem
- [[State Saturation]] — the failure Decision 3 addresses
- [[State Prediction]] — the module Decision 4 defers consuming
- [[Label Leakage]] — the constraint Decision 1 exists to satisfy
- [[Optimizer State]] — Decision 6: no sparse controllers before B' is tested

The cross-cutting decision that diagnostics must observe, never replace, is what
makes [[FAILED_EXPERIMENTS]] readable as evidence rather than as a list of
mistakes.
