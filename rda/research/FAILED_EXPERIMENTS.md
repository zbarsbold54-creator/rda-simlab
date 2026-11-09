# FAILED EXPERIMENTS AND RETRACTED WORK

This file is part of the reproducibility record. Failed and retracted work
is recorded here in full rather than deleted, because in this project the
failures were more informative than several of the successes — each one
changed how the next experiment was designed.

---

## F1 — The false-positive baseline comparison

```
What we tried:
  Trained B' (RDA) once, seed 42, and compared its val_bpb of 1.286248
  against a baseline figure, reporting it as an improvement.

What went wrong:
  Two independent errors, either of which alone would have invalidated it.

  1. STATISTICAL. It was a single seed. The baseline's own 3-seed spread
     turned out to be 0.0137, and 1.286248 falls INSIDE the baseline seed
     range 1.287544-1.301217. The "improvement" was smaller than the
     baseline's own noise, and in fact smaller than the second-worst
     baseline seed.

  2. STRUCTURAL. The run predated bc3deb8, so it was produced by a leaky
     model: FiLM gave position 0 a 1022-token lookahead, and PE gave the
     state a one-token lookahead. A model that can see the future is not a
     test of a causal architecture.

How it was detected:
  The baseline was later measured across 3 seeds (E1). The claimed
  improvement immediately turned out to be inside that range. The
  causality audit (E3) then showed the run predated the leak fixes.

What changed:
  Commit 8d09ecf withdrew the claim in the README and replaced it with the
  3-seed baseline plus an explicit statement that RDA does not currently
  beat the baseline. The 3-seed A/B is now gated on a stability
  demonstration rather than on a loss number.

Scientific lesson:
  A single-run comparison against a single-run baseline is not evidence,
  even when the direction is favourable. Establish the baseline's spread
  FIRST. And check the model is causal before believing any number it
  produces.
```

---

## F2 — Two temporal leaks

```
What we tried:
  An early FiLM implementation took one (B, n_drives) state and broadcast
  it across all positions, plus a state recurrence fed the current
  position's own prediction error.

What went wrong:
  Leak 1: position 0 was modulated by a state computed from the LAST
  token of the sequence — a 1022-token lookahead at T=1024. This also
  made the architecture impossible to reproduce autoregressively, since a
  single decode step has no "final state" to draw on.

  Leak 2: modulating position p with E_next_seq[p] let the logits that
  predict token p+1 depend on token p+1. A one-token lookahead, which
  INFLATES validation loss because at eval time that token does not exist.

How it was detected:
  Reading the code rather than the intended architecture. The signature
  was a shape contract that did not hold: a per-sequence state broadcast
  across positions, and a state that consumed the same position it
  conditioned.

What changed:
  bc3deb8 made the state per-position, and shifted the FiLM state by one
  so position p uses E_next_seq[p-1]. The contract is now documented in
  b19d4c7 and tested in test_pe_film_invariants.py.

Scientific lesson:
  The wrong version was SHORTER and looked correct. For a causal
  recurrent architecture the binding constraint is indexing, not
  mathematics, and the shorter formulation is usually the one with the
  leak. Write down which token generated each signal and which position
  receives it.
```

---

## F3 — The first vectorization was wrong

```
What we tried:
  Replaced the 1023-step Python recurrence with the closed form
      E[t] = w[t] * (decay*E_start + cumsum(g/w)[t]),  w[t] = decay^t

What went wrong:
  The `decay *` factor on E_start was missing, giving
      E[t] = w[t] * (E_start + cumsum(g/w)[t])
  which is the same recurrence seeded with E_start/decay — the WRONG
  initial state. Error 2.9e-02.

How it was detected:
  diagnostics/verify_vectorized_update.py compares VALUES against the
  sequential reference loop, not shapes. A shape check would have passed
  and the bug would have shipped.

What changed:
  The missing factor was restored, and the test was kept as the
  regression guard. The comment in update_seq() now names this specific
  failure so it is not repeated.

Scientific lesson:
  "Same shapes" is not "same function". Equivalence tests must compare
  values, and the test must be written against an INDEPENDENT
  implementation (the original loop), not against a restatement of the new
  one.
```

---

## F4 — Diagnostic shadowing (the most consequential error)

```
What we tried:
  diagnostics/probe_event_growth.py monkey-patched
  RDAStateModule.drive_seq with a full reimplementation, so that ablation
  flags would affect the real forward pass rather than just the
  measurement. The patch was installed UNCONDITIONALLY.

What went wrong:
  The reimplementation silently shadowed any fix made to the real rda.py.
  After implementing the event-output normalization in rda.py and
  re-running the probe, the trajectory came back BYTE-IDENTICAL to the
  pre-fix control: event 0.2715 -> 0.2690, ||W|| 1.6317 -> 2.0126, in
  every digit.

  The natural reading is "the fix did nothing". The true reading is "the
  probe never called the code I changed".

How it was detected:
  Only because a real fix producing an exactly unchanged trajectory is
  implausible. A fix that changes a term's magnitude cannot leave every
  recorded digit identical. That implausibility was treated as a signal to
  investigate rather than as a null result to report.

What changed:
  The patch is now installed ONLY when an ablation flag is set, and the
  probe prints "no ablation flag set -> measuring the REAL drive_seq from
  rda.py" when it is not. The probe's own measurement was also changed to
  mirror the real computation, so measured == computed.

  Rules adopted for all subsequent diagnostics:
    - never reimplement a production method in a diagnostic
    - observe with forward/backward HOOKS on the real modules
    - capture hooks return their input UNCHANGED
    - register and remove hooks explicitly, and assert none remain
    - print which implementation is active on every run
    - treat byte-identical results after a change as a red flag

Scientific lesson:
  This is the failure mode that invalidates a diagnostic silently, because
  the output looks completely normal. A measurement instrument that
  shadows the thing it measures will always report the instrument's own
  behaviour. The controls that prevent it (printing the active
  implementation, gating patches behind explicit flags) cost almost
  nothing and are now mandatory in this repo.
```

---

## F5 — The freeze-base probe was structurally incapable of its control

```
What we tried:
  Control C3 in the PE/FiLM investigation was meant to answer "is
  trainable base-model adaptation required for the PE increase?" by
  freezing the base model.

What went wrong:
  The probe built its optimizer from the RDA state parameters ONLY. The
  base model was never in any parameter group — so it was never updated,
  in the reference run OR in the "frozen" one. C3 came back numerically
  IDENTICAL to C1: 1stCross 71, max|E| 0.9991, peB 3.8395, filmW 107.66,
  to every digit.

  The control was not testing what it claimed. It was comparing "base
  trained" against "base also not trained".

How it was detected:
  Again by exact equality of a supposedly-changed condition. The fix
  required making the probe faithful to production:
      optimizer = m.base.setup_optimizer(unembedding_lr=0.008*scale,
          embedding_lr=0.3*scale, scalar_lr=0.5*scale,
          matrix_lr=0.02*scale, weight_decay=0.0)
      then optimizer.add_param_group(...) for each RDA group, each
      carrying kind='adamw' and initial_lr.
  A second, smaller bug: MuonAdamW's fused kernels dereference every
  gradient unconditionally, so frozen parameters (whose .grad is never
  populated) crash the step. Fixed by handing them explicit zeros, which
  is a true no-op and leaves the optimizer structure untouched.

What changed:
  The probe now uses the production optimizer. C3 became informative and
  produced the single most decisive result in the project (PE flat at
  10.50 for 250 steps while the FiLM gain grew 1.00 -> 9.57).

Scientific lesson:
  An ablation must be verified to CHANGE something before its result is
  believed. Before asking "does freezing X matter", confirm that the
  unfrozen run actually trains X. Silently identical results are the
  signature of a control that is not connected to the thing it controls.
```

---

## F6 — Three of my own regression tests were wrong

```
What we tried:
  Wrote diagnostics/test_pe_film_invariants.py with 8 test groups to lock
  in the properties the investigation established.

What went wrong:
  Three checks failed, and in every case THE TEST was wrong, not the code:

  1. Compared drive_seq()'s output (the DRIVE, a tanh) against update()'s
     output (the STATE). Different quantities; differed by construction.
  2. Compared the loop's FINAL state against every timestep of the closed
     form, instead of comparing timestep t to timestep t.
  3. Seeded the recurrence with E outside [-1, 1], violating the
     precondition the production code defends against with a clamp — and
     triggering a legitimate assertion in update_seq().

  A fourth issue was a tolerance error: the normalization's 1e-6 epsilon
  means a normalized RMS is film_rms * (1 + O(1e-6)), not exact to 1e-16.

How it was detected:
  Each failure was investigated rather than relaxed. Two of them led to a
  real investigation of update_seq() and update(), because a disagreement
  between a vectorized path and a reference loop should never be dismissed
  — the closed form was checked against a hand-computed recurrence in
  pure Python and found algebraically exact, which located the fault in
  the test's shape handling.

What changed:
  The tests were corrected to compare like with like. Final result: 18/18
  pass, with the closed form matching the reference loop to 3.33e-16.

Scientific lesson:
  "The test is wrong" is a conclusion that must be earned, every time, by
  tracing the actual values — not assumed to rescue a failing check. The
  temptation to widen a tolerance or adjust an expectation until green is
  the standard way a test suite stops protecting anything. Each of these
  three failures was a real misunderstanding of the code's contract, and
  finding them made the contract explicit.
```

---

## F7 — A diagnostic summary that contradicted its own measurement

```
What we tried:
  diagnostics/map_pe_film_graph.py printed a prose SUMMARY block stating
  the cycle loss -> PE -> E -> FiLM -> loss "IS CLOSED, and it is a
  genuine differentiable feedback path".

What went wrong:
  The script's own measured output, printed immediately above the summary,
  showed the opposite: with PE's gradient detached, film_scale ratio
  1.0000, film_shift ratio 1.0000, BASE ratio 1.0000. The gradients were
  completely unchanged.

  Both statements are defensible about DIFFERENT things, and the prose
  collapsed that distinction:
    - the CYCLE IS CLOSED: true as a statement about the forward
      computation and the computational graph.
    - the loop is EXCITING at step 0: false, because FiLM is
      zero-initialized so dL/dE is identically zero.

  Written as one sentence, the summary asserted something its own numbers
  did not support.

How it was detected:
  Re-reading the output when it failed to line up with the later
  investigation. The correct statement required measuring the ONSET, which
  is what pe_gradient_onset.py was then written to do.

What changed:
  The finding was recorded accurately: the loop is closed but
  gradient-dormant for exactly one step, reopening at step 1. The
  single-step bisection at step 0 is uninformative by construction, and
  that limitation is now stated in EXPERIMENT_LOG.md E9.

Scientific lesson:
  A diagnostic that prints both measurements and a conclusion can ship a
  conclusion its measurements do not support. When a number and a sentence
  disagree, the number is the evidence. Also: "is this loop closed?" is
  not a yes/no question for a zero-initialized architecture — it is a
  question about when, which is a different experiment.
```

---

## F8 — C6 was a hook, not the production path (a near-miss)

```
What we tried:
  Ran the FiLM-gain normalization as an OPT-IN hook (--film-norm) before
  implementing it in rda.py, to choose film_rms from evidence.

What went wrong:
  Nothing — and that is the point of recording it. After implementing the
  fix in rda.py, the first re-run of the probe came back BYTE-IDENTICAL to
  the C1 reference again (event 1.2612, epW 3.5092, every digit).

  The cause was the SAME shadowing bug as F4: the probe's monkey-patch of
  drive_seq was still installed unconditionally, so it shadowed the new
  FiLM normalization as well. The identical-output signal was recognized
  IMMEDIATELY this time, because the F4 fix had not yet been made but the
  pattern was known.

What changed:
  Same fix as F4 (conditional patching). The lesson had been recorded once
  and transferred.

Scientific lesson:
  A recorded failure is worth keeping precisely because it generalizes. The
  second occurrence was caught in one step instead of several, which is the
  entire return on writing F4 down. Note the sequence: the SAME bug
  appeared twice, which is a strong argument that "we checked this time"
  is not a substitute for a structural guard. The guard is now in the
  probe's design, not in anyone's memory.
```

---

## Summary of failure patterns

| # | Failure | Silent? | Caught by |
|---|---|---|---|
| F1 | single-seed claim from a leaky model | no | multi-seed baseline |
| F2 | two temporal leaks | partly | code inspection, not tests |
| F3 | wrong vectorized closed form | no | value-comparing test |
| F4 | diagnostic shadowed production code | **YES** | implausible exact equality |
| F5 | freeze-base control not connected | **YES** | implausible exact equality |
| F6 | three wrong regression tests | no | investigating each failure |
| F7 | summary contradicted its measurement | partly | re-reading output later |
| F8 | shadowing recurred (near-miss) | **YES** | F4 already documented |

The two most dangerous entries, F4 and F5, shared one signature: **a
result that came back exactly unchanged when something should have
changed.** That pattern — implausible exact equality between a changed and
an unchanged condition — is the most useful heuristic this project
produced, and it is recorded here so it is applied deliberately rather
than by luck.

---

## Related Concepts

Each failure below maps to a mechanism now documented separately:

- [[Label Leakage]] — F1, F2, F6; the most consequential failure
- [[Event Signal]] — F5, the freeze-base probe that could not control
- [[Prediction Error]] — the vector the leak travelled along
- [[FiLM Modulation]] — F4, diagnostic shadowing; a shadowed method returned
  unchanged results
- [[BPB]] — the metric whose readings were most often misread
- [[Weight Decay]] — F7's summary contradicted its own measurement
- [[Ablation Study]] — why two counterfactuals were confounded

The shared signature of F4 and F5 — *a result that came back exactly unchanged
when something should have changed* — is the most useful heuristic this project
produced. Implausible exact equality between a changed and an unchanged
condition means the change did not take effect.
