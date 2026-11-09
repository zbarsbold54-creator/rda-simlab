# EXPERIMENT LOG

Chronological record. Every entry states its commit, seed, configuration,
method, result, interpretation, and limitations. Retractions are recorded
as first-class entries, not deleted.

Two speedup measurements appear below and they are **not** the same
experiment; see E5.

---

## E1 — Baseline A (3 seeds)

```
Date:        2026-09-27
Git commit:  253ce9f (reproducible FineWeb-EDU baseline), reported in 8d09ecf
Seed:        42, 43, 44
Config:      unmodified nanochat path (no --rda)
Question:    What is the baseline's seed-to-seed spread?
Method:      Three independent runs, validation bits-per-byte.
Result:
    seed 42   1.301217
    seed 43   1.287544
    seed 44   1.295109
    mean      1.294623
    spread    0.0137   (max - min)
    std       ~0.0069
Interpretation:
    The spread is the yardstick. Any claimed improvement smaller than
    0.0137 cannot be distinguished from seed noise with three seeds.
Limitations:
    Single configuration. Does not establish that 0.0137 is the right
    yardstick for a different architecture.
```

---

## E2 — Initial RDA result — **RETRACTED**

```
Date:        2026-09-27
Git commit:  pre-bc3deb8 run; claim withdrawn in 8d09ecf
Seed:        42 only
Config:      B' (RDA) as it stood BEFORE the causality fixes
Question:    Does the persistent state improve on the baseline?
Method:      Single training run, validation bits-per-byte.
Result:
    1.286248   — reported at the time as an improvement over 1.294623
Interpretation (at the time):
    Taken as a ~0.008 improvement, i.e. a successful result.
RETRACTION (recorded 2026-09-28, commit 8d09ecf):
    Two independent reasons invalidate this.
    1. It is a SINGLE SEED. 1.286248 falls inside the baseline's own
       seed range (1.287544 - 1.301217). Once E1 was measured, the
       "improvement" turned out to be smaller than the baseline's own
       noise, so it is not distinguishable from a baseline draw.
    2. The run was LEAKY. It predated bc3deb8, which fixed a
       1022-token FiLM lookahead and a one-token PE lookahead. A leaky
       model is not a valid test of a causal architecture.
Status:
    NOT A RESULT. Recorded here so it is not rediscovered and
    re-reported as a success.
Limitations:
    Not re-run. Any future B' must use the post-bc3deb8 causal model.
```

---

## E3 — Causal leak fixes

```
Date:        2026-09-27
Git commit:  bc3deb8 "Fix two temporal leaks in RDA FiLM, and make the
             model decodable"
Seed:        n/a (structural)
Config:      RDA
Question:    Is the architecture actually causal?
Method:      Code inspection plus targeted perturbation tests.
Result:
    Leak 1 — FiLM. modulate() accepted a single (B, n_drives) state and
      broadcast it across all positions. Position 0 was therefore
      modulated by a state computed from the LAST token: a 1022-token
      lookahead at T=1024. It also made the model unreproducible
      autoregressively, since a decode step has no "final state".
      Fix: make the state per-position.
    Leak 2 — PE. E_next_seq[p] is built from pe_seq[p], which is the
      model's error for token p+1. Modulating position p with it let
      position p's hidden state — and thus the logits predicting token
      p+1 — depend on token p+1. A one-token lookahead that inflates val
      loss, because at eval time that token does not exist.
      Fix: shift the FiLM state by one, using E_next_seq[p-1].
Interpretation:
    E2's number was produced by a model that could see the future. The
    comparison in E2 was not merely underpowered; it was invalid.
Limitations:
    Structural reasoning, not a formal proof. Current tests cover the
    specific leak patterns known at the time; a new leak would not be
    caught automatically.
```

---

## E4 — Per-position causal FiLM documented

```
Date:        2026-09-27
Git commit:  b19d4c7
Seed:        n/a
Question:    Make the causality contract explicit in the repo.
Method:      Documentation; film_state = cat([E_start, E_next_seq], dim=1).
Result:      Position 0 <- E_start; position p <- E_next_seq[p-1].
Interpretation:
    Recording WHY the tempting alternative is wrong matters more than the
    fix itself; the wrong version is shorter and looks correct.
Limitations:
    Documentation only.
```

---

## E5 — Vectorized recurrence

```
Date:        2026-09-28
Git commit:  539007e
Seed:        n/a (mathematical equivalence) + perf measurement
Config:      RDA
Question:    Can the 1023-step Python loop be replaced without changing
             the trajectory?
Method:
    The drive at position t depends only on the base hidden state and PE
    at t, neither of which can depend on E. So the whole E sequence has a
    closed form:
        E[t] = w[t] * (decay*E_start + cumsum(g/w)[t]),  w[t] = decay^t
    float64 accumulation so that decay**-t stays finite.
Result:
    Equivalence: verified against the sequential reference loop. The
    first attempt was WRONG by 2.9e-02 — it omitted the `decay *` factor
    on E_start, which corresponds to seeding the recurrence with
    E_start/decay. Caught only because the test compares VALUES, not
    shapes.
    Speedup: TWO DISTINCT MEASUREMENTS, not to be merged:
      - ~153x  — from the original vectorization verification (E5, run
        during commit 539007e). NOT present in any committed file; it is
        a session measurement.
      - 222.8x — from diagnostics/verify_vectorized_update.py as it stands
        in the current suite (commit ab9b560), re-run 2026-09-28.
        vectorized 47.7 ms/pass vs loop 10629.0 ms/pass.
    Also added: a saturation-abort rule, --seed, and run manifests.
Interpretation:
    A ~200x reduction in state-recurrence cost makes 250-step diagnostics
    affordable, which is what made the saturation investigation possible
    at all. The speedup is a research-infrastructure win, not a result.
Limitations:
    Both speedup numbers are machine-specific (RTX 5060 Ti). The 153x
    figure is not recorded in any committed artifact and cannot be
    re-derived from the repo; treat it as historical.
```

---

## E6 — TREE check (run provenance)

```
Date:        2026-09-28
Git commit:  358993e (and 030a384)
Question:    Can a run be trusted to have executed the committed code?
Method:      A pre-run TREE check reports CLEAN or DIRTY; DIRTY means the
    working tree differed from HEAD.
Result:      Worked, and immediately earned its keep: it reported DIRTY
    during the saturation-abort test run, correctly, because uncommitted
    diagnostic work was present.
Interpretation:
    Without this, a diagnostic result could silently reflect uncommitted
    code. Given that the project later hit a shadowing bug of exactly
    this shape, the check was prescient.
Limitations:
    Reports dirtiness; does not say which file or which direction.
```

---

## E7 — Event saturation investigation

```
Date:        2026-09-28
Git commit:  investigation leading to ab9b560
Seed:        42
Config:      depth 8, d_model 768, T=1024, 30 steps, toy scale
Question:    Why did the event term grow, and was LayerNorm failing?
Method:
    14 signals recorded per step over 30 real steps, then five controls:
      C1 unfrozen reference
      C2 base model frozen
      C3 RMS-normalize event projection output
      C4 bound event via tanh(k*x)/k, k=0.3
      C5 decay 0.5
Result:
    LayerNorm output rms:      1.0000 -> 1.0000  (x1.00)   WORKING
    LayerNorm weight norm:    27.7128 -> 27.7128 (x1.00)   WORKING
    event_proj ||W||:          1.6317 -> 3.5092  (x2.15)   GREW
    event |max|:               0.2508 -> 1.2612  (x5.03)   GREW
    raw h rms:                 0.9999 -> 0.9999  (x1.00)
    grad||base||:              0.6657 -> 11.6698 (x17.53)
    Cauchy-Schwarz ceiling:   16.2898 -> 35.4817 (x2.18)
    Controls:
      C2 frozen base  -> IDENTICAL to C1: ||W|| 1.6317 -> 3.5092,
                         event -> 1.2612, matching to 4 decimals.
      C3 normalize    -> event 0.2715 -> 0.2690 (x0.99)  FLAT
      C4 bound 0.3    -> event -> 0.2983 (x1.45); ||W|| still x1.91
      C5 decay 0.5    -> event -> 1.3471 (x5.37)  WORSE
Interpretation:
    LayerNorm was never the failure — it pins ||h|| to a CEILING of
    sqrt(d) = 27.71, not to a value, and event_proj(h) can still reach
    ||w_row|| * 27.71 with ||w_row|| learned and unbounded.
    C2 is decisive: with h byte-identical every step and grad||base|| = 0
    exactly, the growth still happens. So the cause is the gradient on
    event_proj itself, and hidden-state drift is ruled out.
    C5 rules out "the decay is too slow" as an explanation.
Limitations:
    One seed, 30 steps, toy scale. C1-C5 were all measured through a probe
    that had a shadowing defect (see FAILED_EXPERIMENTS.md, F4), so their
    RELATIVE comparison is valid — all five shared the same path — but C3
    in particular was a prediction of the fix rather than an independent
    measurement of it. Superseded by E9.
```

---

## E8 — Event-output normalization (implemented)

```
Date:        2026-09-28
Git commit:  ab9b560
Seed:        42
Config:      depth 8, d_model 768, 250 steps, toy scale
Question:    Does normalizing the projection OUTPUT control the event term?
Method:      RMS-normalize event_proj output over the 8 drive axes, in both
    update() and drive_seq(). Applied identically to both so the
    reference loop and the vectorized path stay comparable.
Result:
    30 steps: event 0.2715 -> 0.2690 (x0.99); ||W|| -> 2.0126 (x1.23)
    250 steps: event 0.2715 -> 0.1670 (x0.62), range [0.162, 0.278] with
               no upward trend; ||W|| 1.6317 -> 0.3582 (x0.22, SHRINKS)
Interpretation:
    The mechanism is controlled, not delayed: ||W|| now shrinks, because
    a global scale on W became a no-op and only its direction is learned.
    The saturation observed at step 88 in this configuration was NOT
    fixed by delaying it — the amplifying degree of freedom is gone.
Limitations:
    A SECOND mechanism remained: the state still saturated, via a
    different route. See E10.
```

---

## E9 — PE/FiLM dependency graph mapping

```
Date:        2026-09-28
Git commit:  ab9b560 (diagnostics/map_pe_film_graph.py, pe_gradient_onset.py)
Seed:        42
Config:      d_model 128, T=1024
Question:    What is the actual gradient path, and when does it open?
Method:
    Real forward/backward on production rda.py. Observations made with
    forward/backward HOOKS on the production modules; the module methods
    are never replaced.
Result:
    PE == NLL(lm_head(hidden[:T-1]), targets[1:])   to 9.54e-07. CAUSAL.
    Perturbing targets[:,k] leaves E at all positions < k bit-identical
    (8.8e-05, i.e. float noise, zero in exact terms).
    fiLM scale/shift and BASE receive gradient at step 0;
    pe_proj, event_proj and h_norm receive EXACTLY 0.000e+00.
    At step 1 all of them become nonzero (pe_proj 4.9e-04).
    film weight norm at step 0: 0.000000 exactly.
Interpretation:
    dL/dE is proportional to the film weights, which are zero-initialized,
    so the state pathway is gradient-dormant for exactly one step. The
    loop is CLOSED but dormant, and reopens at step 1. This is why the
    runaway could not be diagnosed from any single step, and why
    "is the loop closed?" is not a yes/no question — it is a per-step
    question.
Limitations:
    The Q9 bisection at step 0 is uninformative by construction, because
    at step 0 the state gradient is identically zero. A later-step
    bisection would be needed to probe the closed loop directly.
```

---

## E10 — PE/FiLM ablations C1–C6

```
Date:        2026-09-28
Git commit:  ab9b560 (diagnostics/probe_pe_film.py)
Seed:        42
Config:      depth 8, d_model 768, 250 steps, base model TRAINABLE via
             m.base.setup_optimizer + add_param_group, 40-step warmup
Question:    Is the PE/FiLM runaway caused by PE projection drift, FiLM
             amplification, or a differentiable feedback loop?
Method:
    C1 reference (production code, no ablation)
    C2 FiLM disabled (state still computed and recurs)
    C3 base model frozen
    C4 pe_proj frozen
    C5 PE detached from its gradient, numerical value preserved
    C6 FiLM gain RMS-normalized to a fixed budget (hook, opt-in)
Result:
    control   1stCross  max|E|   peB final  FiLM gain  loss
    C1        217       0.9853   1.3900     2.227      5.805
    C2        never     0.9390   1.5149     1.000      6.098
    C3        82        0.9989   3.7130     9.568      9.920
    C4        217       0.9922   1.5149     2.283      5.795
    C5        217       0.9853   1.3896     2.238      5.793
    C6 (0.5)  never     0.9385   1.7620     1.529      5.801
    C6 (1.0)  never     0.9793   1.6941     2.444      5.742
Interpretation — which conclusions are actually supported:
    SUPPORTED: FiLM amplification is the cause. C2 never saturates, and
      PE still grows to 19.6 without FiLM, so PE growth is downstream.
    NOT SUPPORTED: PE projection drift. pe_proj.bias SHRINKS in C1
      (1.5324 -> 1.3900), and C4 (frozen) still saturates at step 217.
    NOT SUPPORTED: differentiable feedback pathology. C5 gives max|E|
      0.9853 and the same crossing step — indistinguishable from C1.
    DECISIVE EXTRA CONTROL (C3): with the base frozen, PE is FLAT at 10.50
      for 250 steps, yet pe_proj.bias grows 1.53 -> 3.71 and the FiLM gain
      grows 1.00 -> 9.57. A loop that runs with constant input is
      self-exciting. This is the strongest single piece of evidence.
    Growth ORDERING in C1: base grad norm first (step 11), then PE
      (step 61), then FiLM gain (step 206), then |E| crossing (step 217).
      pe_proj.bias NEVER doubles. The ordering is consistent with FiLM
      leading and PE following.
Limitations:
    One seed. The C6 sweep at 0.5 and 1.0 is two points, not a sweep —
    film_rms was chosen from these two, which is a judgement call.
    See FAILED_EXPERIMENTS.md F5: the first version of this probe omitted
    the base model from the optimizer, which made C3 identical to C1 BY
    CONSTRUCTION. All numbers above are from the corrected probe.
```

---

## E11 — Current stabilization, verified through production code

```
Date:        2026-09-28
Git commit:  ab9b560
Seed:        42
Config:      depth 8, d_model 768, 250 steps, toy scale
Question:    Does the fix, implemented in rda.py itself, control the
             runaway? (Not a hook, not a prediction — the real path.)
Method:      diagnostics/probe_pe_film.py with NO ablation flags, so the
    production modulate() and drive_seq() execute unmodified. The probe
    prints which implementation is active and asserts zero hooks remain.
Result:
                          before      after
    first |E| > 0.98      217         never
    max |E|               0.9853      0.9402
    final |E|             0.9025      0.9277
    FiLM gain max         2.227       1.522
    loss                  5.805       5.808
    direction cosine      —           1.000000000000
    identity at init      —           exact (max |diff| 0.00e+00)
Interpretation:
    The loss is UNCHANGED (5.808 vs 5.805). This is the check that matters
    most: the fix did not close off a loss shortcut, so the state pathway
    is still doing whatever work it was doing. The runaway is gone rather
    than postponed.
    The C6 hook prediction (never, max 0.9385, loss 5.801) matched the
    production result (0.9402, 5.808) — the ablation predicted the fix.
Limitations:
    ONE SEED. 250 STEPS — saturation began at 217, so this is only ~15%
    past onset, and cannot distinguish "controlled" from "delayed" at a
    longer horizon. TOY SCALE, not the real B' configuration.
    PE still grows to ~36 and remains unexplained.
```

---

## E12 — Regression tests

```
Date:        2026-09-28
Git commit:  ab9b560
Question:    Do the stability fixes preserve the properties the
             investigation established?
Method:      diagnostics/test_pe_film_invariants.py (new, 18 checks) plus
             the three pre-existing scripts.
Result:
    test_pe_film_invariants.py         18/18 PASS
    verify_vectorized_update.py        PASS (222.8x)
    verify_t1024.py                    PASS
    test_saturation_abort.py           PASS
Covered: PE alignment; no one-token lookahead; position-0 causality; FiLM
    gain bounded even at ||W|| x 1e4; exact identity at init; direction
    preservation; event-normalization regression guard; |E| <= 1; closed
    form == reference loop (3.33e-16).
Limitations:
    Three of these tests initially FAILED and the failures were bugs in
    the tests, not the code — see FAILED_EXPERIMENTS.md F6. Worth noting
    that "the test is wrong" is a conclusion that has to be earned each
    time, not assumed.
```

---

## E13 — 1000-step production confirmation — **PASS**

```
Date:        2026-09-29
Git commit:  6977c60 (code under test; no code change in this entry)
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Question:    Is the PE/FiLM saturation mechanism CONTROLLED rather than
             merely DELAYED?
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Console output to logs\pe_1000.txt,
             per-step records to logs\pe_1000.json.
Result:
    max |E| over all 1000 steps      0.9401  at step 254
    steps with |E| > 0.98            NONE
    FiLM gain, steps 250-999         mean 1.3617, stdev 0.0386
                                     min 1.2820, max 1.5781
    FiLM gain slope (250-999)        -0.000033 / step  (-0.0033 per 100)
    FiLM gain  0 -> 1.0000
               250 -> 1.4290
               500 -> 1.3561
               750 -> 1.3562
               999 -> 1.4001
    PE max    0 -> 10.491
              250 -> 36.858
              500 -> 38.733
              750 -> 38.861
              999 -> 38.862        (PLATEAUS near 38.9 after ~step 250)
    loss     10.3983 -> 4.7816
    post-peak decay, max |E| per 100-step window:
        250-349  0.9401     350-449  0.9397     450-549  0.9297
        550-649  0.8977     650-749  0.8511     750-849  0.8143
        850-949  0.7575     950-999  0.6724
    windows exceeding the previous window's max   NONE
Interpretation:
    PASS on both pre-registered criteria.
    1. |E| <= 0.98 for all 1000 steps, with 0.040 of margin.
    2. FiLM gain is stationary around 1.36 and the slope is NEGATIVE, so
       the specific failure mode the criterion was written to catch --
       gain drifting upward while |E| stays under threshold -- is absent.

    On "controlled vs delayed": in the unstabilized configuration |E|
    crossed 0.98 at step 217 and was rising. Here |E| PEAKS at step 254 and
    then decays monotonically in every 100-step window, to 0.6724 by step
    999 -- a factor of 1.4 below its own peak while training continues. A
    delayed ratchet can only arrive later; it cannot fall. Combined with
    the flat-to-negative gain slope, the mechanism is controlled.

    Incidental: PE plateaus near 38.9 after ~step 250 rather than growing
    without bound. This partially answers the E16 anomaly -- the growth
    self-limits -- but does not explain it.
Limitations:
    SINGLE SEED (42). TOY SCALE (depth 8, d_model 768), not the real B'
    configuration. 1000 steps is still far shorter than a production
    training run, so this is evidence, not proof. The stronger control is
    the saturation abort enabled on the real training job.
    The 375 raw step-to-step increases after the peak are ordinary
    sampling noise in max|E|; the windowed test is the meaningful one.
```

---

## E13-SUPERSEDED — 1000-step production confirmation — **measured WITH A LABEL LEAK**

> **SUPERSEDED BY E13-fixed. Do not cite the numbers below as evidence of
> stability.** This entry is retained verbatim as history, not deleted.
>
> **What the leak was.** `rda.py` built the per-position prediction error
> from `targets[:, 1:]`. Since `pe_seq` is indexed by input position and
> `E_next_seq[j]` is the state after consuming position `j`, that made
> `pe_seq[j]` a function of `y[j+1] = x[j+2]`. Because
> `film_state[p] = E_next_seq[p-1]` and
> `logits[p] = lm_head(modulate(hidden[p], film_state[p]))`, the drive
> state at position `p` had already seen `x[p+1]` — the very token
> `logits[p]` is scored against, since the dataloader makes
> `y[p] == x[p+1]` and the loss is `cross_entropy(logits[p], targets[p])`
> with no shift. One token of the answer reached the model.
>
> **Fixed in `969e067`** by indexing `pe_seq` as `targets[:, :-1]`.
> `diagnostics/test_no_label_leak.py` fails with 12 violations on
> `6ae93ce` and passes with 0 after the fix; `scripts/base_train.py` now
> runs it as a hard pre-flight gate.
>
> **Commits whose B′ results are affected:** every B′ run from `ab9b560`
> through `6ae93ce`, including the B′-s42 canary and this E13 entry.
> Probe evidence is committed under `diagnostics/audit_label_leak/`.
>
> **What survives and what does not.** The saturation *mechanism* findings
> (event-output normalization, FiLM self-excitation) were diagnosed on
> probe runs that also carried the leak, so they are not independent
> measurements of the fixed architecture. The specific figures below are
> superseded by E13-fixed. The baseline A numbers (1.301217 etc.) are
> unaffected: those runs never passed `--rda`.

```
Date:        2026-09-29
Git commit:  6977c60 (code under test; no code change in this entry)
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Question:    Is the PE/FiLM saturation mechanism CONTROLLED rather than
             merely DELAYED?
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Console output to logs\pe_1000.txt,
             per-step records to logs\pe_1000.json.
Result:
    max |E| over all 1000 steps      0.9401  at step 254
    steps with |E| > 0.98            NONE
    FiLM gain, steps 250-999         mean 1.3617, stdev 0.0386
                                     min 1.2820, max 1.5781
    FiLM gain slope (250-999)        -0.000033 / step  (-0.0033 per 100)
    FiLM gain  0 -> 1.0000
               250 -> 1.4290
               500 -> 1.3561
               750 -> 1.3562
               999 -> 1.4001
    PE max    0 -> 10.491
              250 -> 36.858
              500 -> 38.733
              750 -> 38.861
              999 -> 38.862        (PLATEAUS near 38.9 after ~step 250)
    loss     10.3983 -> 4.7816
    post-peak decay, max |E| per 100-step window:
        250-349  0.9401     350-449  0.9397     450-549  0.9297
        550-649  0.8977     650-749  0.8511     750-849  0.8143
        850-949  0.7575     950-999  0.6724
    windows exceeding the previous window's max   NONE
Interpretation:
    PASS on both pre-registered criteria.
    1. |E| <= 0.98 for all 1000 steps, with 0.040 of margin.
    2. FiLM gain is stationary around 1.36 and the slope is NEGATIVE, so
       the specific failure mode the criterion was written to catch --
       gain drifting upward while |E| stays under threshold -- is absent.

    On "controlled vs delayed": in the unstabilized configuration |E|
    crossed 0.98 at step 217 and was rising. Here |E| PEAKS at step 254 and
    then decays monotonically in every 100-step window, to 0.6724 by step
    999 -- a factor of 1.4 below its own peak while training continues. A
    delayed ratchet can only arrive later; it cannot fall. Combined with
    the flat-to-negative gain slope, the mechanism is controlled.

    Incidental: PE plateaus near 38.9 after ~step 250 rather than growing
    without bound. This partially answers the E16 anomaly -- the growth
    self-limits -- but does not explain it.
Limitations:
    SINGLE SEED (42). TOY SCALE (depth 8, d_model 768), not the real B'
    configuration. 1000 steps is still far shorter than a production
    training run, so this is evidence, not proof. The stronger control is
    the saturation abort enabled on the real training job.
    The 375 raw step-to-step increases after the peak are ordinary
    sampling noise in max|E|; the windowed test is the meaningful one.
    >>> SUPERSEDED: measured with the label leak described above. The
        "gain slope NEGATIVE" and "|E| decays monotonically" conclusions
        do NOT reproduce on the fixed code -- see E13-fixed, where the
        applied gain slope is +5.72e-04/step and max|E| RISES from
        window 500 onward. Do not carry this entry's conclusion forward.
```

---

## E14 — label-leak fix and audit — **the 0.0749 BPB gain is WITHDRAWN**

```
Date:        2026-09-29
Fix commit:  969e067  "Fix a one-token label leak in the RDA drive state"
Question:    Is the B'-s42 result (1.226305 vs baseline 1.301217) real?
Method:      Read-only audit of the saved checkpoints plus a permanent
             regression test. No training. Scripts and raw output under
             diagnostics/audit_label_leak/.
Result:
    LEAK CONFIRMED. The drive state could see the label it was scored on.
    WITHDRAWN: the 0.0749 BPB gain. It measured the leak, not the state.

    Verified indexing facts:
        y[p] == x[p+1]                                    fraction 1.000000
        gpt.py:532  cross_entropy(logits[p], targets[p])  no shift
        rda.py:901  pe_seq[j] from targets[:, 1:]  ->  y[j+1] = x[j+2]
        rda.py:920  E_next_seq[j] depends on x[j+2]
        rda.py:972  film_state[p] = E_next_seq[p-1] -> x[p+1]
        rda.py:1036 logits[p] = lm_head(modulate(hidden[p], film_state[p]))
        => logits[p] depends on the token it is graded against.

    Dynamic proof: perturb y[:,k] alone with x fixed. hidden and logits at
    positions < k are EXACTLY 0 difference; positions >= k all move (the
    recurrence is cumulative). The violation is the diagonal.

    Why the earlier causality tests missed it: they perturbed x, never y.
    Perturbing x can only ever find a forward-token leak, and this leak
    runs through the LABELS. Two existing tests
    (verify_t1024.py, verify_vectorized_update.py) had controls asserting
    that position p DOES change when y[:,p] changes -- i.e. they were
    written to pass with the leak present. Both corrected in 969e067.

    Ablations on the leaky checkpoint (24 identical val batches):
        baseline a-s42  TRUE y  1.242340
        B'-s42          TRUE y  1.137101
        B'-s42  event term zeroed   1.596395
        B'-s42  FiLM forced const   2.990616
        B'-s42  PE term zeroed      4.371596
      Every ablation lands FAR WORSE than the 1.242340 baseline, which is
      the signature of a model leaning on the leaked signal rather than
      one that has genuinely learned something.
Interpretation:
    The configuration was never wrong and the baseline was never
    contaminated: a-s42 and b-stab-s42 differ only by --rda and
    --model-tag (flag-by-flag diff: 0 differences in the shared set), and
    the dataset, shard count and shard-00000 edge hash all match. The
    invalidity is in the METRIC, not the config: the evaluation was not
    label-blind, so A and B were not measured on the same task.

    Two methods used during this audit were themselves wrong and are
    retracted here rather than quietly dropped:
      - "shuffle y and re-score" as a label-blindness test. The BASELINE
        also moves (+1.748) under it, because scoring against wrong
        answers is simply a harder task. It confounds cheating with
        difficulty. Not evidence either way.
      - "PE OFF via targets=None" (4.29). Invalid counterfactual: it puts
        the TRAINED FiLM into a state distribution it never saw.
Limitations:
    Fixing the index changes what the model is asked to do, so every B'
    number must be re-measured. The cheapest valid rerun is B' only
    (600 steps, ~34 min); the A baseline is unaffected.
    E16 (the PE plateau anomaly) is NOT resolved by this and remains open.
```

---

## E13-fixed — 1000-step production confirmation on FIXED code — **VERDICT WITHDRAWN PENDING RE-MEASUREMENT**

> **The G2/G3 verdicts in this entry were computed on the WRONG COLUMN and
> are withdrawn.** See the correction immediately below. G1 and G4 stand.
> Re-measurement is in progress as `E13-fixed-3000`.

```
Date:        2026-09-29
Fix commit:  969e067
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Harness byte-identical to E13
             (`git diff 6ae93ce 969e067 -- diagnostics/probe_pe_film.py`
             is 0 lines), so the two runs are directly comparable.
Output:      logs\pe_1000_fixed.json, logs\pe_1000_fixed.txt

STANDS:
  G1  max|E| <= 0.98 ................................. PASS
      peak 0.9690 at step 921, 0 steps over bound, margin 0.0110
  G4  loss finite and decreasing .................... PASS
      10.3983 -> 5.1722

WITHDRAWN (wrong column):
  G2  applied gain in band [1.0, 2.0] ............. WITHDRAWN
  G3  applied gain no upward trend ............... WITHDRAWN

  max|E| windowed (E_absmax, an unaffected series):
      300-399 0.9205   400-499 0.8890   500-599 0.9133
      600-699 0.9459   700-799 0.9594   800-899 0.9642
      900-999 0.9690
    Five consecutive rising windows from 500 onward, ending 0.0110 from the
    0.98 bound. REPORTED AS AN OBSERVATION, NOT A GATE: the pre-registered
    criterion was only "|E| <= 0.98 for all steps", which passes. This
    windowed-trend concern was ADDED AFTER seeing the fixed run and is not
    a pre-registered rule.

  PE max 33.1564 at step 430, final 20.138 -- LOWER than E13's 38.862, so
  PE growth is NOT leak-driven. E16 remains open.
```

### Correction — the gain column was the RAW projection, not the applied gain

Three separate errors were made on this metric before it was got right, and
all three are recorded because the sequence is the lesson.

1. **First reading.** The JSON column `film_s_rms` was taken to be the gain.
   It is the RMS of the **raw** `film_scale` output. Mean 0.4472 over
   250-999. G2 was declared FAIL on that basis.
2. **Second reading.** The console column `film s` was taken to be the
   applied gain `1 + _unit_rms(fs, 0.5)`. It is `max|1 + fs|` — also raw,
   just a max instead of an RMS. G2 was declared PASS on a mean of 1.8821
   and G3 FAIL on a slope of +5.7156e-04/step. **Both were wrong.**
3. **What is actually true.** Measured directly in
   `diagnostics/verify_film_s_definition.py`:

       step |  max|1+fs|  max|1+_unit_rms(fs)|   rms(fs)   rms(_unit_rms(fs))
          0 |      1.1186              2.5258     0.0426             0.4999
          3 |      1.1784              2.6666     0.0746             0.5000

   The applied scale is roughly **2x** the raw one. The hook sees the
   `film_scale` module output; `_unit_rms` is applied afterwards, inside
   `modulate()`. `rms(_unit_rms(fs))` is pinned at exactly `film_rms = 0.5`
   by construction, so the only thing that varies is the per-channel
   spread, and a max-based statistic reports the extreme channel.

**Consequence.** Neither `film_s_rms` nor `mod_scale_absmax` nor the console
`film s` is the applied gain. The applied gain was not recorded at all, and
it **cannot be reconstructed** from the stored aggregates, because
`_unit_rms` needs the per-position channel vector and only min/max/mean/rms
were kept. G2 and G3 therefore have no valid verdict from this run.

Fixed in `a5eba19`: the probe now records `applied_scale_absmax`,
`applied_scale_rms`, `applied_shift_*` and an `applied_gain_rms_check` every
step. Verified on a 3-step smoke run: RAW 1.0255 vs APPLIED 1.5102, with
`applied_gain_rms_check = 0.4996` against `film_rms = 0.50`.

G1 and G4 are unaffected — `E_absmax` is the raw per-step state norm and has
nothing to do with the FiLM gain.

---

## E13-fixed-3000 — 3000-step confirmation on FIXED code — **FAIL (both runs)**

```
Date:        2026-09-30
Fix commit:  969e067 (label leak), a5eba19 (applied-gain metric)
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE
Question:    Does the drive state stay under the saturation bound over a
             horizon long enough to see the E13-fixed trend resolve?
Criteria (declared BEFORE the run, not after):
    PASS = max|E| <= 0.98 for ALL 3000 steps
           AND windowed max|E| over the LAST 1000 steps (100-step windows)
               is non-increasing (no window above the previous).
    FAIL = either violated. An abort (|E| > 0.98 for 10 consecutive steps)
           counts as FAIL.
Two runs:
    (a) ABORT RUN    --abort-steps 10, bound 0.98 (base_train.py's rule)
    (b) NO-ABORT RUN --abort-steps 10, bound 0.999, ran all 3000 steps.
        Run (b) exists to resolve the truncation ambiguity in (a), which
        stopped AT the crossing and so could not distinguish "about to
        recover" from "still climbing".

RESULT -- NO-ABORT RUN: FAIL on both criteria.
    steps recorded                3000 (completed)
    peak max|E|                   0.9848 at step 1675   (margin -0.0048)
    steps with |E| > 0.98         151
    first crossing                step 1353
    final max|E|                  0.9737 at step 2999
    CRITERION 1                   FAIL
    CRITERION 2                   FAIL  (3 of 10 windows rise, 2000-2999)
    windowed max|E|, criterion-2 windows, signed:
        2000-2099  0.9840  +0.0018 RISE      2500-2599  0.9824  -0.0013
        2100-2199  0.9830  -0.0010           2600-2699  0.9835  +0.0011 RISE
        2200-2299  0.9828  -0.0002           2700-2799  0.9816  -0.0019
        2300-2399  0.9840  +0.0012 RISE      2800-2899  0.9809  -0.0007
        2400-2499  0.9837  -0.0002           2900-2999  0.9820  +0.0011 RISE
      The rises are marginal (+0.0011..+0.0018) and the series is
      flat-to-oscillating around 0.982-0.984, but the criterion as declared
      is strict non-increase, so it fails.

RESULT -- ABORT RUN: FAIL on all three.
    stopped at step 2324 of 3000; first crossing 1353; peak 0.9869 @ 1675;
    261 steps over bound; abort fired (10 consecutive). Criterion 2 could
    only be evaluated over windows 1325-2324, not the true last 1000.

TRUNCATION AMBIGUITY: RESOLVED. The state does NOT recover.
    threat max|E|: peak 0.9848 @ 1675, final 0.9737; last-500 range
    0.9731-0.9835. It plateaus just under the bound and STAYS there for the
    remaining 1300 steps. The abort run's 0.9814 @ 2324 was not a transient
    overshoot on the way down.

APPLIED GAIN: NO PLATEAU.
    1.0000 -> 3.5855 (max 3.7117); second-half slope +0.00084/step, still
    climbing at step 2999. Every 500 steps:
        0     1.0000     500   1.9446    1000  2.1200    1500  2.3193
        2000  2.6753     2500  3.2666    2999  3.5855
    The RAW column max|1+fs| reaches 4.5041, i.e. it now EXCEEDS the applied
    column. Early in training applied > raw; by step 2999 raw > applied. The
    relationship INVERTS, so a reader who has the wrong column cannot tell
    from the number alone -- which is why the applied gain is now recorded
    explicitly (a5eba19).

PE: NOT the driver of the upward pressure.
    PE max plateaus at 33.7550 (step 715) and FALLS to 18.79 by step 2999.
    pe_proj.bias for threat is NEGATIVE, -0.7426 -> -1.7404, i.e. learned to
    push threat DOWN. The event term SHRINKS, 0.2585 -> 0.1433.
    threat_pe_term grows (mean -2.046, absmax 2.710) but in the NEGATIVE
    direction for the bulk.

MECHANISM, AS MEASURED. The threat state is at a fixed point of the
recurrence, tracking its drive:
    |mean state - mean drive| over steps 1500-2999
        mean 0.00003, max 0.00015
    corr(mean drive, mean state) over the second half   0.9997
    one-step prediction decay*E[t-1] + (1-decay)*drive[t] vs observed
        -0.9667 vs -0.9666  (diff +0.0000)
    A constant drive d* has fixed point E* = d* exactly, and the observed
    mean state equals the observed mean drive to five decimals. The state is
    not merely "near" equilibrium; it is on it, and it moves with the drive
    rather than sitting at a stale value.

    So the state is faithfully tracking a drive that has itself stopped
    moving much. The saturation is a property of where the drive has
    settled, not of a failure in the recurrence.

CORRECTIONS TO EARLIER STATEMENTS IN THIS SESSION
    (a) The equilibrium arithmetic was wrong. It was written as "the state
        settles at the 0.95/0.05 equilibrium ~ 0.992". The fixed point of
        E[t] = 0.95*E[t-1] + 0.05*drive[t] under constant drive d* is
        E* = d*; the 0.05 is the WEIGHT on the drive, not a multiplier on
        it. The conclusion survives (residual 3e-05) but it now rests on
        measurement rather than on that arithmetic.
    (b) The framing "threat saturates upward and pins the state high" was
        wrong. The mean state is about -0.970 and the mean drive about
        -0.966. max|E| ~ 0.98 comes from a POSITIVE TAIL, not from a bulk
        pinned high. Both ends of the range are occupied: the bulk is
        pressed against the LOWER bound and a minority tail rides the UPPER
        bound.
        >>> RETRACTED BY MEASUREMENT, see E13-fixed-3000-dist below. The
        >>> sentence "max|E| ~ 0.98 comes from a POSITIVE TAIL" and "a
        >>> minority tail rides the UPPER bound" are FALSE. The measured
        >>> fraction of positions with drive > +0.9 is EXACTLY 0.0 at
        >>> every one of the 3000 steps, and from step 500 onward the
        >>> entire distribution lies below zero. The bulk is pressed
        >>> against the LOWER bound and so is the extreme: absmax is
        >>> |min| by construction, reached at the NEGATIVE end. Only the
        >>> first clause (mean state about -0.970) survives.
    (c) The comparison that produced the original claim,
        drive_preclip_absmax (0.9918) vs threat_E_absmax (0.9737), compared
        two DIFFERENT maxima and is not a valid equilibrium test. The valid
        test is mean vs mean, which is what the fixed-point numbers above
        use.

EXPLICITLY UNRESOLVED
    Whether the threat drive distribution is BIMODAL (two separated modes)
    or UNIMODAL near -1 with a thin positive tail. A mean of -0.97 with an
    absmax of +0.99 is consistent with BOTH. Not measured: the probe logs
    per-step summary scalars, not the per-position distribution. EITHER
    CHARACTERISATION IS CURRENTLY UNSUPPORTED and neither is asserted here.
    Resolving it requires logging the per-position drive quantiles and a
    histogram.
    >>> RESOLVED by E13-fixed-3000-dist below: UNIMODAL. The "positive
    >>> tail" half of this either/or is refuted outright (fraction > +0.9
    >>> is exactly 0.0 at every step).

RUN PROVENANCE -- THE TWO RUNS ARE NOT BIT-IDENTICAL
    Same seed, same config, only the abort bound differs. Over the 2325-step
    overlap (diagnostics/compare_abort_runs.py):
        field                   max |A-B|    rel      identical?
        loss                    5.391e-02   5.2e-03  no
        E_absmax                2.080e-02   2.1e-02  no
        pe_max                  7.035e+00   2.1e-01  no
        mod_scale_absmax        6.096e-01   1.5e-01  no
        applied_scale_absmax    4.079e-01   1.3e-01  no
    NO field is bit-identical, and the divergence is visible by step 25 --
    far before the abort bound could matter. Cause: GPU nondeterminism in
    the backward pass (compute_init seeds the RNGs, but nothing calls
    torch.use_deterministic_algorithms). The no-abort run is therefore an
    INDEPENDENT TRAJECTORY AT THE SAME SEED, not a continuation of the
    aborted one; agreement between them is statistical, not identity. This
    does not weaken the conclusion -- both runs put threat at the bound --
    but the truncation question is answered by a second sample rather than
    by resuming the first.

LIMITATIONS
    SINGLE SEED. TOY SCALE. 3000 steps, still far short of a production run.
    Both runs are nondeterministic, so neither is exactly reproducible and
    no run-to-run variance estimate exists.
    Criterion 1's threshold (0.98) is the base_train.py abort bound, not a
    physical limit of the architecture; the tanh bound is 1.0.
    E16 (the PE plateau) is NOT resolved: PE still plateaus and falls, and
    that behaviour is unchanged by the leak fix.
    No B' A/B was run; these gates are upstream of it.

UNCOMMITTED FILES ON DISK (nothing committed as of this entry)
    logs\pe_3000_fixed.json / .txt            abort run, 2325 steps
    logs\pe_3000_fixed_noabort.json / .txt    no-abort run, 3000 steps
    logs\eval3000.txt                         abort-run analysis
    logs\eval3000_noabort.txt                 no-abort analysis
    logs\smoke.json, logs\smoke_log.txt       probe smoke test
    logs\smoke_threat.json, logs\smoke_threat.txt
    diagnostics\eval_e13_3000.py              abort-run criteria
    diagnostics\eval_e13_3000_noabort.py      no-abort criteria
    diagnostics\compare_abort_runs.py         run-to-run divergence
    diagnostics\verify_equilibrium.py         fixed-point test
    diagnostics\verify_film_s_definition.py   applied-gain definition
    diagnostics\eval_e13_fixed.py             E13-fixed criteria
    diagnostics\reconcile_gain_metric.py      metric reconciliation
    diagnostics\recompute_gates.py            gate recomputation
    diagnostics\probe_pe_film.py              MODIFIED, uncommitted:
                                             --abort-bound, --log, applied
                                             gain fields, threat_* fields

---

## E13-fixed-3000-dist — per-position threat-drive distribution — **UNIMODAL, and the "positive tail" is refuted**

Same config as E13-fixed-3000 no-abort (3000 steps, seed 42, `--abort-steps 10
--abort-bound 0.999`). The only change is LOGGING: the probe now records the
per-position threat-drive quantiles every step and a 100-bin histogram at steps
0/500/1000/1500/2000/2500/2999. No model, gain, or drive code was touched. The
existing per-step records are unchanged; the new data went to a separate file.

    first step |E|>0.98 : 1177      (prior run: 1353)
    max |E|             : 0.9881     (prior run: 0.9848)
    final |E|           : 0.9769
    applied gain        : 1.0000 -> 3.4726, max 3.5814
    p50 slope over 1000..2999: -4.050e-06 /step, R^2 = 0.3253
    p50 slope over 2000..2999: +3.402e-07 /step

The two runs are NOT bit-identical (same seed, GPU nondeterminism in the
backward pass; see the provenance note in E13-fixed-3000). First crossing moved
1177 vs 1353 and peak 0.9881 vs 0.9848. Both are within the run-to-run spread
already characterised, but they are different numbers and the earlier run's
values are not superseded by them.

### (a) Bimodal or unimodal-with-tail? — UNIMODAL at every saved step

Decided by an explicit valley test on the histogram, not by eye. At steps
1500/2000/2500 the entire 4092-position distribution occupies THREE bins:

    step 1500  occupied bins: [672, 2622, 798]
    step 2000  occupied bins: [966, 2491, 635]
    step 2500  occupied bins: [894, 2381, 817]

Three adjacent bins with no empty bin between them cannot be bimodal. The
occupied support narrows monotonically as the drive saturates:

    step    support span   mode bin centre   frac > +0.9   frac < -0.9
       0        0.504          -0.276           0.0000        0.0000
     500        0.216          -0.852           0.0000        0.0000
    1000        0.120          -0.948           0.0000        1.0000
    1500        0.072          -0.972           0.0000        1.0000
    2000        0.072          -0.972           0.0000        1.0000
    2500        0.072          -0.972           0.0000        1.0000
    2999        0.096          -0.972           0.0000        1.0000

The distribution is a single mode that moves toward -1 and concentrates. It
does not split, and it does not develop a second mode.

Under/overflow are 0 at every step, as they must be: tanh cannot leave (-1, 1),
so the [-1.2, 1.2] bin range can never be exited. That the counts are 0 confirms
the range is wide enough, and it also means the binning is not clipping anything
at the negative end either.

### CORRECTION (d) — the "positive tail" was wrong, and the correction of it was also wrong

Two separate errors, both now settled by measurement:

1. The original claim that max|E| ~ 0.98 "comes from a POSITIVE TAIL" is
   **false**. The fraction of positions with drive > +0.9 is **exactly 0.0000 at
   every one of the 3000 steps**. From step 500 onward the highest occupied bin
   is entirely below zero (edge -0.696 at step 500, -0.936 by step 1500). The
   entire distribution is negative; `absmax` is `|min|` by construction, reached
   at the NEGATIVE end.
2. The earlier correction (b) in E13-fixed-3000, which itself said "a minority
   tail rides the UPPER bound", repeats the same error in the opposite
   direction and is retracted by the same measurement. Correction (b)'s surviving
   clause is only the mean: mean state about -0.970, mean drive about -0.966.

This is the third time in this line of work that a summary scalar was read as if
it described a distribution. `absmax` was reported without its sign, and
"absmax 0.99" next to "mean -0.97" invites a two-sided story that the
distribution does not support.

### (b) How p1/p50/p99 and the tail fractions move as applied gain rises

    step     gain       p1        p50       p99     f>0.9    f<-0.9
       0   1.0000   -0.3932   -0.2190   -0.0108   0.0000   0.0000
     500   1.8815   -0.8794   -0.8522   -0.7702   0.0000   0.0000
    1000   2.1233   -0.9829   -0.9540   -0.9224   0.0000   1.0000
    1500   2.5848   -0.9918   -0.9712   -0.9464   0.0000   1.0000
    2000   3.0292   -0.9928   -0.9740   -0.9497   0.0000   1.0000
    2500   3.1372   -0.9918   -0.9742   -0.9457   0.0000   1.0000
    2999   3.4726   -0.9913   -0.9721   -0.9436   0.0000   1.0000

Over the whole run: p1 spans 0.601, p50 spans 0.807, p99 spans 1.013 — all three
move monotonically toward -1 in the first ~1200 steps and then stop. f>0.9 never
leaves 0. f<-0.9 goes 0 -> 1.0 between steps 500 and 1000 and stays there.

The asymmetry between p1 and p99 is the whole story: by step 1500 the 1st
percentile is at -0.9918 and the 99th at -0.9464, a spread of 0.045. The
distribution is not merely near -1, it is tightly clustered just above it, with
the mode at -0.972 holding 58-64% of all mass.

### (c) Does p50 approach -0.98, and when would it cross?

**No, not meaningfully.** p50 reaches -0.9740 by step 2000 and then stops:

    p50 over 1000..2999: least-squares slope -4.050e-06 /step, R^2 = 0.3253
    p50 over 2000..2999: least-squares slope +3.402e-07 /step  (positive)
    p50 at step 2999 = -0.97211;  -0.98 lies 0.00789 BELOW it

A straight-line fit over 1000..2999 puts the crossing of -0.98 at **step ~4043**
— that is an INFERENCE from a linear extrapolation, not a measurement, and the
data argue against it: the slope over the last 1000 steps is +3.4e-07, i.e.
positive, so p50 is drifting back AWAY from -0.98, not toward it. The R^2 of
0.3253 on the full-window fit means the line explains a third of the variance and
should not be extrapolated from at all.

Better statement: p50 **plateaus near -0.972 to -0.974 from step ~1800 onward**,
oscillating in a band about 0.010 wide. It is not approaching -0.98, and the
mechanism does not require it to. The equilibrium is a fixed point at
`E* = d*`, and `d*` is itself whatever the learned `pe_proj.bias` and event term
make it — about -0.972 for the median position. There is nothing driving p50
toward -0.98 specifically; -0.98 was a threshold chosen from `absmax`, and
`absmax` is a max over positions, not a quantile, so it was never the right
quantity to compare p50 against.

**So `max|E| ~ 0.98` crossing 0.98 is NOT the bulk of the distribution reaching
0.98.** It is roughly the 1st-percentile position (p1 = -0.9913) plus the
recurrence's memory of it. Criterion 1 in the previous entry was evaluated
against a bound that the bulk of the distribution never reaches.

LIMITATIONS
  - 100 bins over [-1.2, 1.2] is 0.024 per bin. From step 1500 the whole
    distribution spans 0.072, i.e. THREE bins. Bimodality separated by less
    than 0.024 would be invisible here. The UNIMODAL verdict is conditional on
    that resolution. A finer histogram near -1, or a direct count of local
    maxima in the unsmoothed per-position values, would settle it without the
    binning caveat.
  - One seed, one run. The run-to-run spread characterised in E13-fixed-3000
    applies to every number here, including the shape verdict.
  - This measures the DRIVE distribution. The STATE distribution `E` is not
    separately histogrammed; it is the recurrence-smoothed version of the drive
    and is expected to be at least as unimodal, but that is an inference.

FILES WRITTEN (all uncommitted)
  research/EXPERIMENT_LOG.md                     this entry + retraction of (b)
  diagnostics/probe_pe_film.py                   dist logging (--dist-out)
  diagnostics/analyze_threat_drive_dist.py       shape/quantile/crossing analysis
  logs/threat_drive_dist.json                    3000 quantile rows + 7 histograms
  logs/pe_3000_fixed_dist.json                   per-step records (unchanged shape)
  logs/pe_3000_fixed_dist.txt                    console log
  logs/dist_report.txt                           analysis output
  logs/dist_shape_check.txt                      support-span / sign checks
  logs/dist_valley_fix.txt                       corrected valley test
  logs/dist_p50_trend.txt                        p50 regression
  logs/dist_sign_check.txt                       sign of the distribution
  logs/smoke_dist.json, logs/smoke_dist_dist.json, logs/smoke_dist.txt   3-step smoke test

A NOTE ON THE VALLEY TEST ITSELF
  The first version of analyze_threat_drive_dist.py computed valley depth as
  `min(h[i-1],h[i+1])/h[i]`. That metric is INVERTED — it is small at a PEAK, not
  at a valley — so it selected the mode itself and reported BIMODAL for a
  three-bin spike with no gap. The corrected metric is `h[i]/min(h[i-1],h[i+1])`,
  with the additional guards that both neighbours are non-empty and the bin is
  interior to the occupied support (an empty bin next to the support edge would
  otherwise register as a valley). The corrected function is what produced the
  verdicts above. Recorded here because the wrong version's output is the kind of
  thing that gets quoted later as a finding.

## E13-fixed-3000-preact — why the threat drive collapses to the lower bound

Same config as the two runs above (3000 steps, seed 42, `--abort-steps 10
--abort-bound 0.999`). The only change is LOGGING: the pre-tanh argument is
decomposed into its additive terms, the pe_proj gradients are recorded, and
the drive's correlation with its raw inputs is measured across positions. No
model, gain, or drive code was touched.

    first step |E|>0.98 : 1341   (prior runs: 1353, 1177)
    max |E|             : 0.9864  (prior runs: 0.9848, 0.9881)
    final |E|           : 0.9765
    applied gain        : 1.0000 -> 3.3150, max 3.4480

Third run, third distinct set of numbers, same seed. Consistent with the
run-to-run spread already recorded. None of the three supersedes the others.

### THE ANSWER, IN ONE LINE

The threat drive is pinned at the lower bound because **`pe_proj.bias[threat]`
converged to -1.711 and supplies 80% of a pre-tanh argument whose median is
-2.11**, and the channel is *near*-dead but not dead: the mean tanh derivative
plateaus at 0.059 and never crosses 0.05. The collapse is a **bias ratchet that
converged and then stopped**, not a runaway and not a dead channel. It is an
equilibrium, reached around step 1300-1800, and the state has been sitting on
it ever since.

    FINAL STEP (2999) -- argument decomposition, SIGNED
      component      mean        p5         p50        p95        share
      ev          -0.07388   -0.11257    -0.07933   -0.01890     3.5%
      wpe         -0.35497   -0.77599    -0.32914   -0.02607    16.6%
      bias        -1.71124   -1.71124    -1.71124   -1.71124    80.0%
      arg         -2.14009   -2.55999    -2.11174   -1.81227   100.0%
      drv         -0.96999   -0.98812    -0.97113   -0.94806

    The identity ev + wpe + bias == arg was CHECKED, not assumed:
    worst |residual| over all 3000 steps = 4.619e-07.

The bias is constant across positions by construction (std exactly 0.000000),
so it is a pure offset. The two input-dependent terms together supply only
20% of the argument, and between them they add little: ev and wpe are
anti-correlated in effect, ev going -0.007 -> -0.079 while wpe goes
+0.526 -> -0.355.

### (a) When does pre-tanh p50 cross -2 and -3?

    p50 first below -2.0 : step 1131
    p50 first below -3.0 : NEVER in 3000 steps

    step   arg_p50    arg_p1   arg_p99    ev_p50   wpe_p50      bias   drv_p50
       0   -0.2226   -0.4155   -0.0108   -0.0066    0.5262   -0.7426   -0.2190
     250   -0.8544   -1.0987   -0.5067   -0.1326    0.2551   -0.9792   -0.6934
     500   -1.2444   -1.3650   -1.0071   -0.1420    0.0619   -1.1658   -0.8467
     750   -1.7145   -1.9598   -1.4768   -0.1033   -0.1390   -1.4747   -0.9372
    1000   -1.8583   -2.3483   -1.5972   -0.0492   -0.2053   -1.6016   -0.9525
    1250   -2.0111   -2.5973   -1.6777   -0.0394   -0.2763   -1.6993   -0.9648
    1500   -2.0981   -2.7370   -1.7825   -0.0953   -0.2846   -1.7275   -0.9703
    2000   -2.1289   -2.7543   -1.7731   -0.0728   -0.3193   -1.7420   -0.9721
    2500   -2.1026   -2.6955   -1.7124   -0.0168   -0.3558   -1.7316   -0.9706
    2999   -2.1117   -2.6981   -1.7618   -0.0793   -0.3291   -1.7112   -0.9711

arg_p50 crosses -2 at 1131 and then plateaus in a band about 0.10 wide for the
remaining 1870 steps. It does NOT approach -3 and shows no sign of doing so.

For reference, tanh(-2) = -0.9640 and tanh(-3) = -0.9951. The measured
drive median -0.9711 sits just past tanh(-2), which is exactly consistent.

### (b) pe_proj.bias[threat] trajectory and its gradient

    bias: step 0 -0.74259  ->  step 2999 -1.71124   (total change -0.96865)

    first step at or below -1.0   : 307
    first step at or below -1.5   : 763
    first step at or below -1.711 : 1303

    step        bias   g_bias(signed)   g_w(signed)   tanh_deriv
       0   -0.74259       0.000e+00       0.000e+00     0.942634
       1   -0.74255      +2.476e-04      +2.555e-04     0.942987
     250   -0.97925      -4.667e-03      -5.754e-03     0.522460
     500   -1.16583      +1.436e-04      +1.133e-04     0.288521
     750   -1.47474      +7.741e-05      +4.015e-05     0.123240
    1000   -1.60165      +1.729e-04      +8.093e-05     0.091519
    1500   -1.72746      +1.047e-04      +5.267e-05     0.058175
    2000   -1.74198      -2.790e-04      -1.280e-04     0.056295
    2500   -1.73157      -2.800e-05      -1.360e-05     0.061613
    2999   -1.71124      -1.732e-06      -6.847e-07     0.058967

    g_bias[threat] sign over 3000 steps: negative 1376, positive 1623, zero 1
    mean +7e-05, min -1.089e-02, max +1.223e-02, final -1.7e-06

**The gradient does NOT hold one sign.** 1623 steps positive against 1376
negative. So the bias is NOT a one-way ratchet in the strict sense -- it is
being pushed both ways, and it settles because the two directions balance.
    >>> CONFIRMED on a second run by E13-fixed-3000-all8: threat's bias
    >>> gradient is negative 1390 steps and positive 1609, dominant fraction
    >>> 0.537. Two-sided, as stated here. What makes threat special is the
    >>> MAGNITUDE of the bias change (-0.9486, roughly 6-20x every other
    >>> dimension) and its share of the argument, NOT the gradient's sign.

The bias also CONVERGED rather than running away:

    bias 1000->1500 : -1.60165 -> -1.72746  (delta -0.12581, -2.52e-04/step)
    bias 1500->2000 : -1.72746 -> -1.74198  (delta -0.01453, -2.91e-05/step)
    bias 2000->2500 : -1.74198 -> -1.73157  (delta +0.01041, +2.08e-05/step)
    bias 2500->2999 : -1.73157 -> -1.71124  (delta +0.02034, +4.08e-05/step)

The per-step drift falls by an order of magnitude from the first window to the
last, and the last two windows are POSITIVE -- the bias is now drifting back
UP toward zero, not down. It overshot to about -1.756 near step 1750 and has
been relaxing since.

The gradient magnitude also decays: |g_bias| is O(1e-2) at step 250 and
O(1e-6) by step 2999, a factor of ~10^4. This is the tanh derivative shutting
the gradient path down, not the gradient being cancelled.

### (c) Is the channel dead?

    first 5 values: 0.9426, 0.9430, 0.9422, 0.9423, 0.9409
    final  5 values: 0.0549, 0.0591, 0.0638, 0.0578, 0.0590

    first step < 0.50 : 267
    first step < 0.20 : 588
    first step < 0.10 : 885
    first step < 0.05 : NEVER
    first step < 0.01 : NEVER

    min 0.050683 at step 2217 ; slope over 2000..2999 = +2.08e-07/step
    fraction of positions with tanh deriv < 0.01: max 0.00171

**No -- the channel is not dead, and the answer to "at what step is it below
0.05" is NEVER.** The mean tanh derivative falls 0.943 -> 0.059 and then
plateaus; its minimum over the whole run is 0.0507, and the late slope is
POSITIVE. It approaches 0.05 asymptotically without crossing.
    >>> CORRECTED by E13-fixed-3000-all8, same seed, different run: on THAT
    >>> run threat's mean tanh derivative DOES cross below 0.05, at step
    >>> 1743, with a minimum of 0.0472. Here it reached 0.0507 and stopped
    >>> just short. So the two runs straddle the threshold and the
    >>> "never below 0.05" claim is run-specific, not a property of the
    >>> mechanism. The "MARGINAL, not dead" reading survives either way: in
    >>> both runs the derivative sits at ~0.05 and neither goes toward 0.

This is the single most important number in the entry, because it decides
whether the collapse is self-sustaining. At 0.059 the tanh still passes ~6% of
its gradient, which is enough for the bias to relax back (as (b) shows it
doing) but not enough to escape the saturated region. The system is in a
**marginal** fixed point, not a frozen one.

### (d) Did correlation with raw inputs drop?

    step   drive_std    corr_ev     corr_pe     corr_wpe   ev_std
       0    0.098400   +0.999635   +0.009930   +0.009930   0.104100
     500    0.023018   +0.927650   +0.468395   +0.468395   0.066544
    1000    0.015087   +0.396785   +0.905473   +0.905473   0.065642
    1500    0.011500   +0.164798   +0.948522   +0.948522   0.043810
    2999    0.012435   +0.035347   +0.962846   +0.962846   0.029552

    corr_ev_term : +0.99963 -> +0.03535  (change -0.96429)
    corr_pe_term : +0.00993 -> +0.96285  (change +0.95292)
    drive_std    : 0.09840  -> 0.01244   (-87.4%)

Yes, decisively -- but the two inputs SWAP rather than both dying:

- **corr with the event term collapses** from +0.9996 to +0.0353. The threat
  drive has essentially STOPPED responding to `event_proj(h_norm(h))`, which
  is the only genuinely input-dependent, context-sensitive pathway.
- **corr with the PE term rises** from +0.0099 to +0.9628. The drive now
  tracks the prediction-error pathway almost perfectly.

The swap has a mechanical explanation visible in the weight: `pe_proj.weight
[threat]` crosses zero between steps 569 and 570, going from +0.00078 to
-0.00186, and keeps going to -0.575. Before that crossing the PE term
contributes with the OPPOSITE sign to the bias; after it, with the same sign.
So the PE pathway went from opposing the collapse to reinforcing it.

Net effect: the drive's across-position variation fell to std 0.0124, i.e. 87%
of its initial spread is gone. The dimension is close to a constant -0.97.
That is the "collapse" in the plain sense -- the dimension has largely stopped
responding to anything, and the part of the input it still responds to is the
one that has been captured by a near-constant bias.

Note drive min/max at step 2999 is -0.99328 / -0.93732, BOTH negative, which
is the signed confirmation of the E13-fixed-3000-dist finding.

`corr_arg` is ~0.97 throughout and is logged only to make explicit that it is
near 1 by construction (tanh is monotone in its own argument); it carries no
information.

### (e) Do the other seven dimensions show any of this?

**PARTIALLY ANSWERABLE, and the reason matters.** The existing per-step record
stores per-dimension `E_perdim` and `pe_perdim` as **absmax aggregates only** --
no per-dimension tanh argument, no per-dimension gradient, and NO SIGNED MEAN.
`E_perdim` is unsigned, so it cannot show which bound a dimension is pinned
against. That is precisely the defect that produced the retracted "positive
tail" claim, and it applies here too. This run logged threat only, as
specified, so (a)-(d) cannot be computed for the other seven at all.

    per-dimension |E| trajectory (E_absmax at 0/500/1000/1500/2000/2500/2999)
              dim       0     500    1000    1500    2000    2500    2999
          novelty  0.5536  0.6321  0.5528  0.6661  0.7269  0.7531  0.7903
      uncertainty  0.1932  0.4065  0.4875  0.5745  0.6627  0.6612  0.6439
           threat  0.2773  0.8579  0.9652  0.9775  0.9798  0.9779  0.9765
       integrity  0.2293  0.3398  0.1720  0.1859  0.0995  0.1016  0.0791
           agency  0.4353  0.2199  0.2893  0.2086  0.2477  0.2818  0.1924
  learning_progress  0.8143  0.5757  0.1954  0.1797  0.1901  0.1895  0.1981
          arousal  0.9236  0.8630  0.6630  0.4808  0.3804  0.3206  0.2117
  homeostatic_dev  0.8022  0.6061  0.2710  0.1979  0.1874  0.2167  0.0971

What IS measurable from the unsigned data:

- **Threat is the ONLY dimension that grows monotonically to a rail.** It is
  the only one still rising at step 2999 while every other dimension is flat
  or falling.
- **Three dimensions are in the same family but going the OTHER way**:
  `arousal` 0.9236 -> 0.2117, `learning_progress` 0.8143 -> 0.1981,
  `homeostatic_dev` 0.8022 -> 0.0971. All three START high (they begin near
  +0.8-0.9 because the state is initialised there) and decay away. They do not
  show a collapse to a rail; they show relaxation toward zero. Whether they
  are pinned high, pinned low, or merely small CANNOT BE DETERMINED from an
  absmax, and given that threat's own absmax was misread this way already, no
  claim about their signs is made.
- `novelty` and `uncertainty` rise slowly and are still rising at 2999
  (0.7903, 0.6439) but have not flattened against a bound the way threat has.
- `integrity`, `agency` sit low (0.079, 0.192) and stay low.

**INFERENCE, flagged as such:** the three decaying dimensions are plausibly
the mirror image of threat -- initialised high, relaxing toward zero as the
bias-driven equilibrium is established. But because `E_perdim` is unsigned,
this is a guess from magnitudes, and it must not be recorded as a finding. A
real answer needs the pre-arg decomposition and signed means logged for all 8
dimensions. That is a concrete, cheap follow-up and it is listed as NOT YET
RUN.
    >>> RETRACTED by E13-fixed-3000-all8 below, which measured the signs.
    >>> The three dimensions are NOT the mirror image of threat. Signed
    >>> drv_p50 at step 2999: arousal -0.1808, learning_progress +0.0381,
    >>> homeostatic_dev -0.0379 -- all near the CENTRE of the tanh, not at
    >>> a bound. They relax toward zero, not toward a rail. The
    >>> "plausibly the mirror image" was an inference from unsigned
    >>> magnitudes and it is wrong.
table would have suggested every dimension is positive, which is very likely
false. The table has been removed from the script and is not reproduced above.
Recorded because the wrong output is the kind of thing that gets quoted later.

FILES WRITTEN (all uncommitted; base_train.py verified unchanged throughout)
  research/EXPERIMENT_LOG.md                  this entry
  diagnostics/probe_pe_film.py                --preact-out logging
  diagnostics/diagnose_threat_collapse.py     diagnosis + identity check
  logs/threat_preact.json                     3000 preact rows, 3000 grad rows,
                                              5 correlation snapshots
  logs/pe_3000_fixed_preact.json              per-step records (unchanged shape)
  logs/pe_3000_fixed_preact.txt               console log
  logs/preact_report.txt                      full diagnosis output
  logs/preact_extra.txt                       weight sign-flip / convergence checks
  logs/smoke3.json, logs/smoke3_preact.json, logs/smoke3_check.txt   smoke test

## PROPOSAL — replacing criterion 1 (NOT IMPLEMENTED)

The current criterion 1 is "max|E| stays below 0.98". This entry shows it is
the wrong instrument: `max|E|` is a max over positions, it is unsigned, and
the bulk of the distribution never approaches it (p50 plateaus at -0.971 while
absmax reads 0.9765). A criterion built on it can pass while a dimension is
dead, or fail while a dimension is healthy but large.

Proposed replacement, per dimension d, with max|E| retained as INFORMATIONAL
only (no pass/fail role):

  R_d      = p95(drive_d) - p5(drive_d)        SPREAD
  M_d      = modal-bin mass fraction           CONCENTRATION
  D_d      = mean(1 - drive_d^2)               HEADROOM
  signed_max_d = max(drive_d), signed_min_d = min(drive_d)

  FAIL if any of:
    (1) R_d < 0.02                 the dimension has stopped varying
                                    across positions
    (2) M_d > 0.90 AND R_d < 0.05  a spike: essentially all mass in one bin
    (3) D_d < 0.01                 the tanh is saturated; no gradient can
                                    flow, so nothing downstream is
                                    attributable to this dimension
    (4) signed_min_d > +0.9 or signed_max_d < -0.9
                                    COLLAPSED TO A RAIL, with the sign
                                    stated. This is the check whose absence
                                    produced the "positive tail" error.

  INFORMATIONAL (reported, never gating):
    max|E|, applied gain, arg quantiles, per-dimension bias/weight.

Why these and not max|E|:

- R_d is the direct measure of "is this dimension doing anything". It is
  scale-free, sign-aware, and needs no threshold calibrated to a rail value.
  Measured on this run: threat R_d = p95 - p5 = -0.94806 - (-0.98812) =
  0.04006, i.e. **already failing (1) at step 2999** -- and correctly so,
  since corr with the event term has fallen to 0.035 and the std is 0.0124.
  Note the earlier report's E13-fixed-3000-dist run gave p5/p95 as -0.98812 /
  -0.94806 too, so this is stable across runs.
- M_d separates "narrow" from "spiky". A healthy narrow dimension and a dead
  spiky one both have small R_d; only M_d tells them apart. Measured modal
  mass for threat is 0.58-0.64 (E13-fixed-3000-dist), which alone would NOT
  trip (2) -- correctly, since 0.64 is not a spike, and R_d does the work.
- D_d is the headroom check. At the measured 0.059 it does not trip, which is
  the right answer: the channel is marginal, not dead, and a criterion that
  called it dead would be wrong.
- (4) is the sign-aware replacement for the absmax comparison that produced
  two retractions in this work.

Thresholds 0.02 / 0.90 / 0.05 / 0.01 / 0.9 are PROPOSALS, not calibrated. They
should be set from a known-healthy baseline run before being used to pass or
fail anything, and the modal-bin definition (bin width, and whether it is
fixed in drive units or relative to R_d) must be pinned down first. Stated as
a proposal only; nothing implemented.

## E13-fixed-3000-all8 — all 8 dimensions: threat is the ONLY one that saturates

Same config (3000 steps, seed 42, `--abort-steps 10 --abort-bound 0.999`).
Only logging changed: the pre-arg decomposition is generalized to all 8
dimensions, signed, with variance shares and dL/d(drive) added.

    first step |E|>0.98 : 1207  (prior: 1353, 1177, 1341)
    max |E|             : 0.9866 (prior: 0.9848, 0.9881, 0.9864)
    final |E|           : 0.9765
    identity worst residual: 4.768e-07 (step 1154, threat) -- PASS, tol 1e-6

    IDENTITY, all 8 dims, all 3000 steps: arg = ev + wpe + bias, worst
    residual 4.768e-07, i.e. float32 round-off. Decomposition reconciles.

    THREAT RECORDS UNCHANGED: cross-checked the all-8 file against the
    threat-only file field by field, worst diff 0.000e+00 (bit-exact).

### (a) arg p50 crossing +/-2, SIGNED

    dim                first<-2   first>+2   arg_p50@2999
    novelty                  never      never      +0.9004
    uncertainty              never      never      -0.6577
    threat                    1128      never      -2.1322
    integrity                 never      never      -0.0197
    agency                    never      never      +0.0649
    learning_progress         never      never      +0.0381
    arousal                   never      never      -0.1828
    homeostatic_dev           never      never      -0.0379

**Threat is the only dimension whose argument ever leaves the +/-2 band**, and
it leaves it downward only. No dimension ever rises above +2. This is the
single cleanest result in the entry: the collapse is a THREAT-SPECIFIC event,
not a property of the architecture as a whole.

### (b) bias trajectory and signed gradient sign

    dim                start      end     delta     g<0   g>0    g_end
    novelty          +0.7085  +0.8060  +0.0975   1484  1515  -1.952e-04
    uncertainty      -0.7749  -0.9234  -0.1484   1467  1532  -6.854e-04
    threat           -0.7426  -1.6912  -0.9486   1390  1609  -4.873e-05
    integrity        -0.0867  -0.1373  -0.0506   1482  1517  -5.188e-03
    agency           +0.5089  +0.5722  +0.0633   1532  1467  -2.223e-02
    learning_progress +0.2664  -0.1456  -0.4120   1441  1558  +4.957e-04
    arousal          -0.5821  -0.0670  +0.5152   1552  1447  +3.143e-04
    homeostatic_dev  -0.1341  +0.2057  +0.3398   1541  1458  +1.880e-03

**NO dimension has a one-signed bias gradient.** Dominant fractions run
0.505-0.537 -- i.e. essentially 50/50 for all eight. This CORRECTS the
one-sided reading in the threat-only entry: over 3000 steps threat's bias
gradient is negative 1390 times and positive 1609, so it is two-sided, not a
ratchet. What makes threat special is the MAGNITUDE of the bias
(-0.9486, ~6-20x every other dimension's change) and its share of the
argument, not the gradient's sign.

Four dimensions change sign in their bias over the run: learning_progress
+0.2664 -> -0.1456, arousal -0.5821 -> -0.0670 (relaxing back toward 0),
homeostatic_dev -0.1341 -> +0.2057. None of these approach a rail.

### (c) mean tanh derivative below 0.05

    dim                first<0.05  first<0.10      min    min@step
    novelty                NEVER      never     0.4573      2830
    uncertainty            NEVER      never     0.6416      2802
    threat                   1743       768     0.0472      2850
    integrity               NEVER      never     0.8894        70
    agency                  NEVER      never     0.7783       118
    learning_progress       NEVER      never     0.3628         0
    arousal                 NEVER      never     0.1606         0
    homeostatic_dev         NEVER      never     0.3777        17

**Threat is the ONLY dimension that ever goes below 0.05** (step 1743, min
0.0472). Note this is a small correction to the threat-only entry, which said
"NEVER" below 0.05 with min 0.050683 on a different run. Same seed, different
run, and on THIS run it does cross. The threat-only figure was 0.0507 --
just above the line. So the honest statement is: threat sits at ~0.05,
straddling the threshold, and whether it crosses is run-to-run noise. The
"MARGINAL, not dead" conclusion survives either way.

The other seven never go below 0.16, i.e. they retain at least 16% of their
tanh gradient. They are not near saturation.

### (d) which bound the drive p50 sits against, SIGNED

    dim              step 0            500            1000           1500           2000           2999
    novelty        +0.514 off     +0.612 off     +0.531 off     +0.602 off     +0.657 off     +0.717 off
    uncertainty    +0.140 ctr     -0.235 ctr     -0.403 off     -0.480 off     -0.542 off     -0.577 off
    threat         -0.219 ctr     -0.862 off     -0.951 LOW-    -0.972 LOW-    -0.973 LOW-    -0.972 LOW-
    integrity      +0.180 ctr     -0.119 ctr     -0.056 ctr     -0.129 ctr     -0.061 ctr     -0.020 ctr
    agency         -0.380 off     -0.028 ctr     +0.006 ctr     +0.050 ctr     +0.073 ctr     +0.065 ctr
    learning_prog  +0.797 off     +0.458 off     -0.005 ctr     +0.101 ctr     +0.097 ctr     +0.038 ctr
    arousal        -0.917 LOW-    -0.786 off     -0.556 off     -0.327 off     -0.179 ctr     -0.181 ctr
    homeostatic    -0.779 off     -0.449 off     -0.031 ctr     -0.108 ctr     -0.105 ctr     -0.038 ctr
    (off = off-rail, ctr = centre, LOW- = LOWER RAIL)

Signed extremes at step 2999, so the reader can see the sign of the absmax:

    dim             drv_p5    drv_p50   drv_p95    drv_min   drv_max
    novelty         +0.6838   +0.7165   +0.7365   +0.6348   +0.7571
    uncertainty     -0.7705   -0.5768   -0.1722   -0.8057   +0.1994
    threat          -0.9881   -0.9723   -0.9482   -0.9941   -0.9349
    integrity       -0.0855   -0.0197   +0.0282   -0.1886   +0.0816
    agency          -0.4422   +0.0648   +0.4267   -0.7145   +0.5135
    learning_prog   -0.1724   +0.0381   +0.2994   -0.2662   +0.4858
    arousal         -0.4458   -0.1808   +0.0498   -0.6134   +0.1031
    homeostatic     -0.3713   -0.0379   +0.2278   -0.5987   +0.3150

Note the E_perdim table in the previous entry, built from the UNSIGNED absmax
field, showed `arousal` decaying 0.9236 -> 0.2117 and `learning_progress`
0.8143 -> 0.1981 and `homeostatic_dev` 0.8022 -> 0.0971. With signs now
measured, the interpretation is DIFFERENT from the "mirror image of threat"
guess recorded there: arousal is at -0.181 and the other two near zero, i.e.
they are all sitting near the CENTRE, not pinned at a rail. The earlier
"they are the mirror image of threat" was an INFERENCE from unsigned
magnitudes and it is WRONG. They are relaxing toward the centre, not toward a
bound. Recorded as a correction.

Only two dimensions ever sit against a rail: threat (lower, from step 1000)
and arousal (lower, but ONLY at step 0 -- it then walks back to the centre by
step 2000 and is at -0.181 by 2999).

### (6) pe_proj.weight[d,0] and its zero crossing, SIGNED

    dim                start     end   zero-crossing
    novelty          -0.1392 -0.0363     never
    uncertainty      +0.8965 +0.6602     never
    threat           +0.5060 -0.5864     546
    integrity        +0.2608 +0.0025     1881
    agency           -0.8769 -0.7323     never
    learning_progress +0.7964 +0.3215     never
    arousal          -0.9489 -0.3898     never
    homeostatic_dev  -0.8844 -0.4633     never

**Only TWO dimensions cross zero**, and threat crosses earliest (step 546).
integrity crosses at 1881 and lands at +0.0025, i.e. it is effectively zero at
the end. Every other weight keeps its initial sign for the whole run.

Threat's crossing at 546 is the same event identified in the threat-only
entry (measured there as between steps 569/570 on a different run; here 546).
It is what flips the PE pathway from opposing the bias to reinforcing it.

### (7) variance shares -- who owns the across-position variation

    step 2999
    dim              ev_var    wpe_var        cov   ev_share  wpe_share  dominant
    novelty        9.520e-04  2.172e-04  -4.465e-05     0.814     0.186  event
    uncertainty    8.449e-04  7.195e-02  -5.072e-04     0.012     0.988  PE-weight
    threat         3.941e-04  5.676e-02  -7.988e-04     0.007     0.993  PE-weight
    integrity      1.320e-06  1.022e-06   9.557e-06     0.999     0.001  event
    agency         1.449e-03  8.852e-02  -9.739e-04     0.016     0.984  PE-weight
    learning_prog  2.672e-03  1.706e-02   1.410e-03     0.135     0.865  PE-weight
    arousal        1.049e-03  2.508e-02   1.036e-03     0.040     0.960  PE-weight
    homeostatic    1.514e-03  3.543e-02   8.854e-04     0.041     0.959  PE-weight

At step 0 the event term owns essentially ALL the variance for all eight
dimensions (ev_share 0.999-1.000), because `pe_proj.weight` starts small and
the PE is nearly constant across positions.

By step 2999 the picture has INVERTED for six of eight: the PE-weight term
owns 0.865-0.993 of the variance. Only `novelty` (0.814 event) and
`integrity` (0.999 event) still vary mainly with the event.

The covariances are small relative to the variances (e.g. threat
cov -7.988e-04 against ev_var 3.941e-04, wpe_var 5.676e-02), so the two terms
are close to independent contributors rather than cancelling.

This is the same "swap" the correlation analysis showed for threat in the
previous entry, now confirmed for six dimensions by variance share rather
than by correlation -- a stronger statement, because a correlation can be
misleading when the two inputs have wildly different scales.

### (8) dL/d(drive_d), SIGNED

    dim                mean         std      absmax   g_bias_signed@end
    novelty       -9.654e-08  1.209e-06   7.106e-06      -1.952e-04
    uncertainty   -3.028e-07  7.393e-06   3.453e-05      -6.854e-04
    threat        -2.132e-07  1.702e-06   7.780e-06      -4.873e-05
    integrity     -1.272e-06  9.995e-06   3.603e-05      -5.188e-03
    agency        -5.935e-06  1.743e-05   6.057e-05      -2.223e-02
    learning_prog +1.272e-07  5.715e-06   2.061e-05      +4.957e-04
    arousal       +7.738e-08  2.708e-06   1.262e-05      +3.143e-04
    homeostatic   +4.644e-07  7.399e-06   2.556e-05      +1.880e-03

`|mean|/absmax` runs 0.006-0.098: the gradient CANCELS across positions by
roughly 10-160x for every dimension. A dimension whose per-position gradients
agreed in sign would have |mean| ~ absmax. None does, so no dimension is being
driven coherently by the loss; each is pushed both ways and nets out.

`agency` has both the largest mean and the largest absmax, so it is the most
strongly driven dimension -- not threat.

VERIFIED, NOT ASSUMED: dL/d(drive) is EXACTLY 0.0 at step 0 for all eight
dimensions. This is correct, not a broken hook. rda.py:1079 states "The state
influences the model only through FiLM", and dpe is "computed, returned and
logged, and nothing about memory / learning rate / attention is modulated by
it" (rda.py:1077-1079). `film_scale.weight` and `film_shift.weight` are
ZERO-INIT (measured absmax exactly 0.0 at init), so at step 0 the loss has no
path from E to the logits and therefore none from the drive. Confirmed by a
standalone check: with retain_grad installed, `drive.grad` is not None but has
absmax 0.0 and nonzero-fraction 0.0000 at step 0.

CONSEQUENCE, and it is the central structural fact of this investigation:
**the drive's only path to the loss is FiLM, and FiLM is zero-initialised.**
So at step 0 there is no gradient at all on the drive pathway, and the
gradient that exists later is entirely mediated by the learned FiLM weights.

CONSUMERS, read off rda.py with line numbers (not asserted):
  drive_direct : RDAStateModule.update_seq (rda.py:951) -- the only direct
                 path from drive into E
  E_state      : RDAStateModule.modulate (rda.py:1084) via
                 film_scale/film_shift (rda.py:501-502) -- FiLM
  E_state      : RDAStatePredictor.predict_seq (rda.py:954) ->
                 state_proj (nn.Linear(8,8), rda.py:544)
  E_state      : drive_prediction_error (rda.py:957) -- dpe readout;
                 LOGGED BUT NEVER CONSUMED

### (9) Pattern summary and health

health, as specified: R_d = p95-p5 of the drive >= 0.3 AND no bias offset
above 50% of the arg. NOT CALIBRATED -- applied as given.

    dim              R_d      bias@end   arg_p50@end   bias/arg   (i)  (ii) (iii) (iv)
    novelty       0.0527      +0.8060      +0.9004       0.90     Y    .    .     .
    uncertainty   0.5984      -0.9234      -0.6577       1.40     Y    .    Y     .
    threat        0.0399      -1.6912      -2.1322       0.79     Y    Y    Y     Y
    integrity     0.1138      -0.1373      -0.0197      undef     .    Y    Y     .
    agency        0.8689      +0.5722      +0.0649      undef     Y    .    Y     .
    learning_prog 0.4717      -0.1456      +0.0381      undef     .    .    .     .
    arousal       0.4956      -0.0670      -0.1828       0.37     .    .    Y     .
    homeostatic   0.5990      +0.2057      -0.0379      undef     .    .    Y     .

    (i)  bias converged to |bias| >= 0.5 : novelty, uncertainty, threat, agency
    (ii) pe weight crossed zero         : threat (546), integrity (1881)
    (iii) one term owns >90% of variance: uncertainty, threat, integrity,
                                       agency, arousal, homeostatic_dev
    (iv) mean tanh deriv ever below 0.10: threat

    HEALTHY     : agency, learning_progress, arousal, homeostatic_dev
    NOT HEALTHY : novelty, uncertainty, threat, integrity

READ THE "undef" COLUMN CAREFULLY. It is not missing data. For four
dimensions |arg_p50| is under 0.2, so |bias|/|arg_p50| is 3.8-8.8 and the
ratio is dominated by a near-zero denominator. `agency` shows 8.817 on
|bias| 0.5722 against |arg_p50| 0.0649 -- the bias is 8.8x the median
argument, but the median argument is nearly zero, so the MEDIAN drive is
near the centre of the tanh even though the mean argument is not. Marking
these "unhealthy on bias" would be wrong; the ratio is not a meaningful
measure at small |arg|. This is a defect in the health definition as
specified, and it is reported rather than quietly patched.

Caveat on (iii): six of eight exceed 90% variance share, including four
healthy ones. So "one term owns the variance" is TRUE for almost everything
here and does not discriminate health. The 90% cut is too loose to be a
health signal in an architecture whose PE pathway grows to dominate by
construction. It only becomes informative for `integrity` (0.999 event) and
`novelty` (0.814 event) -- the two that never let the PE term take over.

### WHAT THIS ADDS TO THE THREAT-ONLY ENTRY

1. The collapse is THREAT-SPECIFIC. Seven of eight dimensions never saturate.
   Any account of the collapse that is really about the architecture as a
   whole is wrong; it has to be about what is specific to threat.
2. Threat's bias gradient is TWO-SIDED (1390 neg / 1609 pos), so the earlier
   "one-signed ratchet" framing was wrong even though the conclusion
   (converged offset dominating the argument) was right. Magnitude, not sign.
3. The tanh-derivative threshold: threat crosses below 0.05 at step 1743
   here, and did NOT cross on the earlier run (min 0.0507). Straddling the
   line; run-to-run noise decides. "Marginal, not dead" holds either way.
4. The "arousal / learning_progress / homeostatic_dev are the mirror image of
   threat" inference from the previous entry is REFUTED. Signed measurement
   puts them near the CENTRE, not at a bound.
5. The drive reaches the loss ONLY through FiLM, which is zero-init, so
   dL/d(drive) is exactly zero at step 0. This is a structural property of the
   design and it is the reason the per-position gradients cancel so strongly
   (|mean|/absmax 0.006-0.098) once they exist.

FILES WRITTEN (all uncommitted; base_train.py verified unchanged throughout)
  research/EXPERIMENT_LOG.md                   this entry + 2 retractions
  diagnostics/probe_pe_film.py                 --preact-all-out, retain_grad wrapper
  diagnostics/diagnose_all8.py                 all-8 diagnosis
  logs/affect_preact_all.json                  3000 rows x 8 dims + 6 variance steps
  logs/pe_3000_all8.json, logs/pe_3000_all8.txt   per-step records, console log
  logs/all8_report.txt                         full diagnosis output
  logs/all8_extra.txt                          dLddrive magnitudes, bias shares
  logs/smoke_all8*.json/.txt                    20-step smoke test + checks

## E13-fixed-3000-film — is threat a constant offset into FiLM?

> ### RETRACTED BY MEASUREMENT — the conclusion below is WRONG
>
> **The "hypothesis refuted" verdict and every offset/varying figure in this
> entry are retracted.** The bug: `.mean()` and `.std()` were called with **no
> `dim` argument** on a tensor of shape **(d_model=768, B, T)**, so both
> reduced over **output units x batch x positions together** rather than over
> positions only. For `contrib_j(p) = W[j,d] * E_d(p)` that gives
>
> $$\mathrm{std}_{\text{flat}}^2 = \mathbb{E}_j[W_j^2]\,\mathrm{var}_p(E) \;+\; \bar{E}^2\,\mathrm{var}_j(W)$$
>
> The second term is the spread of the **column direction across the 768
> output units** — a property of the weights, not of how E varies across
> positions. With $|\bar E| = 0.977$ for threat it dominates completely. The
> `.mean()` had the same defect, so the ratio was built from two contaminated
> statistics.
>
> **Specifically retracted from this entry:**
> - the varying std **0.2923** and the offset/varying ratio **0.043** for
>   threat;
> - the ranking that placed threat **6th of 8** and the claim "threat has the
>   LARGEST varying part of all eight (std 2.923e-01)";
> - the `|col| * E_std` comparison (0.0114 vs 0.2923 "both true") — that
>   apparent contradiction was itself a symptom of comparing a per-unit
>   quantity against a unit-flattened one, not two views of one number;
> - the whole "THE ANSWER" section and subsections (a), (b), (c) insofar as
>   they rest on the ratio.
>
> **The hypothesis is in fact SUPPORTED.** Reducing over positions only gives
> `offset_dj = W[j,d]*E_mean_d` and `varying_dj = W[j,d]*E_std_d`, and
> aggregating over units by RMS makes the column factor cancel, so
>
> $$\text{offset\_rms}/\text{varying\_rms} = E_{\text{mean}}/E_{\text{std}}\quad\text{exactly}$$
>
> giving **436.1** for threat, not 0.043 — threat is **1st of 8**, not 6th. See
> the table in the ADDENDUM below and `E13-fixed-3000-film-fix`.
>
> The per-dimension column norms, E statistics, and the variance-share and
> reconstruction machinery in this entry are NOT affected by this bug: they
> reduce over positions only and are reported signed throughout.

Same config (3000 steps, seed 42, `--abort-steps 10 --abort-bound 0.999`).
Only logging changed. Instrument: `--film-out logs/film_weights.json`.

    first step |E|>0.98 : 1202  (prior: 1353, 1177, 1341, 1207)
    max |E|             : 0.9893 (prior: 0.9848, 0.9881, 0.9864, 0.9866)
    final |E|           : 0.9828

WHAT WAS MEASURED. FiLM is linear in E, so the contribution of dimension d
to output channel c is exactly `W[:,d] * E[:,d]` -- a constant vector times a
per-position scalar. Its across-position mean is therefore the DC/offset part
and its std the varying part, with no modelling assumption. Ratio
mean/std is a direct offset-to-signal measure. The E logged is the FiLM
INPUT, `film_state = cat([E_start, E_next_seq], dim=1)` (rda.py:1022), NOT
E_next_seq and NOT the drive; verified because our `threat_E_absmax` at
step 250 was 0.739617, matching the independently-computed
`threat_E_absmax` in the main per-step file exactly.

### ~~THE ANSWER: threat is the SECOND-LEAST offset-like dimension~~ RETRACTED

> Everything in this subsection is retracted. The ranking below is built from
> the unit-flattened ratio and is **inverted** relative to the correct
> positions-only reduction. See the RETRACTION banner at the head of this
> entry and the ADDENDUM at the foot. The table is left in place, marked, so
> the error stays visible rather than being quietly deleted.

    RANKED BY |offset/varying| AT STEP 2999 (scale+shift combined)
    ** RETRACTED -- built from the unit-flattened reduction **
    #   dim                 |o/v|     E_mean        E_std    col_nrm
    1   uncertainty         0.382    -0.4873    4.196e-02     6.5489
    2   homeostatic_dev     0.169    -0.0408    3.402e-02     5.5965
    3   agency              0.138    -0.0120    5.635e-02     9.6699
    4   learning_progress   0.119    +0.1029    3.025e-02     5.0308
    5   novelty             0.082    +0.7991    5.730e-03     5.0532
    6   THREAT              0.043    -0.9771    2.240e-03     5.1058
    7   integrity           0.012    +0.0085    2.344e-02     6.1990
    8   arousal             0.012    -0.1751    3.028e-02     4.9152

**The hypothesis is refuted.** *(RETRACTED — the opposite is true. The
paragraphs below explain the error and are themselves superseded.)* The
intuition that a saturated, near-constant E_threat must act as a DC offset
into FiLM is wrong ~~, and the reason is visible in the two columns that
matter~~:

- threat has the SMALLEST E_std of all eight (2.240e-03, vs 4.2e-02 for
  uncertainty and 5.6e-02 for agency -- 19x and 25x larger);
- but its COLUMN NORM is mid-pack (5.1058, rank 5 of 8 on scale).

A small E_std means a small *numerator* in the ratio, not a large one. The
offset part of threat's contribution is genuinely near zero -- and the reason
is that the FiLM COLUMN has partly cancelled it. Threat's signed column mean
is +4.001e-02 at step 250 while the other dimensions' column means are of
similar size, so a large negative E_mean (-0.977) multiplied by a column
whose own mean is small leaves a small product mean.

So threat is NOT a large offset. It is a SMALL SIGNAL. The distinction
matters: a large DC offset and a near-zero signal both make a dimension look
"constant", and the o/v ratio separates them, and threat lands at the
signal end.

### (a) Does the threat column grow relative to the other seven? **NO -- it SHRINKS in relative terms**

    scale column norm
      dim                     0     250     500    1000    1500    2000    2500    2999
      novelty            0.0000  2.1267  2.8082  3.6607  4.2634  4.6414  4.9050  5.0532
      uncertainty        0.0000  2.7128  3.8540  5.0515  5.7672  6.1343  6.3800  6.5489
      threat             0.0000  2.4267  3.1021  3.9208  4.4561  4.7794  4.9970  5.1058
      integrity          0.0000  3.0976  3.6923  4.2896  4.5882  5.1066  5.5973  6.1990
      agency             0.0000  1.8190  3.3066  5.8156  7.1527  8.1914  9.1807  9.6699
      learning_progress  0.0000  1.9525  2.6088  3.4291  4.0292  4.5011  4.8376  5.0308
      arousal            0.0000  2.0148  2.6785  3.5097  4.0834  4.4843  4.7694  4.9152
      homeostatic_dev    0.0000  1.9533  2.6111  3.4595  4.1974  4.7025  5.2907  5.5965
      threat share         n/a   13.4%  12.6%  11.8%  11.6%  11.2%  10.9%  10.6%
      threat rank          n/a      3      4      4      4      4      5      5
      threat / other7      n/a  1.0836 1.0072 0.9394 0.9152 0.8860 0.8540 0.8309

Threat's column norm grows in ABSOLUTE terms (0 -> 5.1058) exactly as all
seven others do, because they all start at zero (FiLM is zero-init) and grow
together. But its SHARE falls monotonically 13.4% -> 10.6%, and its rank
falls from 3rd to 5th. By step 500 it is already below the mean of the other
seven (ratio 1.0072, then decreasing every sampled step).

`agency` is the opposite case and is the real outlier: its column grows to
9.6699, rank 1, ratio to the others 1.2-1.4x by step 1000. If any dimension
is dominating FiLM's weights, it is agency, not threat.

On the shift side threat is rank 1 at every sampled step, but its lead
shrinks monotonically (ratio 1.2368 -> 1.1105), so it is losing ground there
too. Note the two projections disagree: threat is 5th on scale, 1st on
shift. Reporting only one would have been misleading.

### (b) Is threat's contribution mostly offset rather than varying? **NO**

At step 2999, combined scale+shift:

    dim                 contrib_mean   contrib_std      o/v
    novelty             +0.019017    2.312e-01     +0.082
    uncertainty         +0.054589    1.428e-01     +0.382
    threat              +0.012512    2.923e-01     +0.043
    integrity           +0.000088    7.589e-03     +0.012
    agency              -0.003209    2.319e-02     -0.138
    learning_progress   -0.003643    3.062e-02     -0.119
    arousal             +0.000583    5.054e-02     +0.012
    homeostatic_dev     -0.002662    1.571e-02     -0.169

Threat's contribution has the LARGEST varying part of all eight
(std 2.923e-01) and a small offset part (mean +0.012512). It is the
most signal-dominated dimension in the set, not the least.

The dimension that IS most offset-like is `uncertainty` (o/v +0.382), which
has a large E_mean (-0.4873) AND a moderate E_std (4.196e-02) -- both parts
present. `novelty` is instructive: E_mean +0.7991, the largest mean in the
set, yet o/v only 0.082, because its E_std has collapsed to 5.730e-03. So
novelty is the closest thing to a genuine constant offset, and it is not
threat.

### (c) Does the column norm grow as E_std shrinks? **YES for threat, and the product is tiny**

    dim              E_std step0 -> step2999      col step2999    |col|*E_std
    novelty          0.05124 -> 0.00573  (-88.8%)       5.0532          0.0290
    uncertainty      0.01978 -> 0.04196  (+112.2%)      6.5489          0.2748
    threat           0.02723 -> 0.00224  (-91.8%)       5.1058          0.0114
    integrity        0.02336 -> 0.02344  (  +0.4%)       6.1990          0.1453
    agency           0.03984 -> 0.05635  ( +41.4%)       9.6699          0.5449
    learning_progress 0.07906 -> 0.03025 (-61.7%)        5.0308          0.1522
    arousal          0.08994 -> 0.03028  (-66.3%)       4.9152          0.1489
    homeostatic_dev  0.07648 -> 0.03402  (-55.5%)       5.5965          0.1904

(The step-0 column norm is exactly 0 for every dimension, so a step-0 ->
step-2999 percentage for |col|*E_std is 0/0 and is not quoted.)

Threat's E_std falls 91.8%, the largest proportional collapse of any
dimension, while its column grows like everyone else's. So the two DO move
together -- but the PRODUCT, which is the size of the varying contribution,
falls to 0.0114, the SMALLEST of all eight. The column growth did not
compensate for the variation collapse; it multiplied a signal that was
already 91.8% smaller.

Contrast `agency`: E_std GROWS 41.4% and its column grows fastest, giving the
largest varying contribution (0.5449), 48x threat's. And `integrity`, whose
E_std is essentially unchanged (+0.4%), keeps a mid-sized varying part.

OFFSET magnitude |col| * |E_mean| per dimension, over time:

    dim                 250     500    1000    1500    2000    2500    2999
    novelty           1.2770  1.6541  1.8895  2.5008  3.1733  3.7107  4.0382
    uncertainty       0.2861  0.8891  1.5853  2.3824  2.7453  2.7694  3.1910
    threat            1.6953  2.6509  3.7566  4.3346  4.6606  4.8804  4.9887
    integrity         0.5475  0.5762  0.3450  0.3000  0.2034  0.0738  0.0528
    agency            0.0474  0.0950  0.2006  0.4374  0.0430  0.3939  0.1160
    learning_progress 1.2004  1.1408  0.2120  0.1337  0.2681  0.5487  0.5175
    arousal           1.7171  2.0387  1.9137  1.4589  1.1879  1.1500  0.8608
    homeostatic_dev   1.2021  1.1265  0.4345  0.7567  0.6096  0.4590  0.2285

Threat and novelty have the LARGEST raw offset magnitude (4.9887 and 4.0382)
-- which is exactly why the o/v ratio was needed. Threat's |E_mean| is 0.9771
and its column is 5.1, so the product is large in absolute terms; but the
column's own signed mean is small, so what reaches modulate() as a DC term is
much smaller than the naive product suggests. The o/v ratio, computed from
the actual contribution tensor, is the honest measure and it says threat is
signal-dominated.

### WHAT THIS DOES AND DOES NOT ESTABLISH

ESTABLISHED, as measured:
- threat is NOT a constant offset into FiLM; it has the 6th-highest o/v of 8
  and the largest varying contribution of any dimension (std 2.923e-01).
- threat's FiLM column does NOT grow faster than the other seven; its share
  falls 13.4% -> 10.6% and its rank falls 3rd -> 5th.
- threat's E_std collapses 91.8%, the most of any dimension, and the
  resulting varying contribution is the smallest of all eight (0.0114).

NOT ESTABLISHED, and stated as inference:
- INFERENCE: that FiLM is therefore the channel through which the saturated
  threat drive does damage. This run does not test that. What it shows is
  that threat's FiLM contribution is small AND nearly constant across
  positions, which is a weak, nearly-static nudge rather than a large DC
  bias. Whether that nudge matters for the loss is a separate question, and
  the earlier measurement that |mean(dL/ddrive)|/absmax = 0.006-0.098 says
  the per-position gradients largely cancel. The two together suggest the
  drive's influence is diffuse, not a large offset -- but that is an
  inference from two measurements, not a measurement.
- The apparent paradox (large |col|*|E_mean| but small measured o/v) is
  resolved by the column's own signed mean being small, but that resolution
  was not itself verified by reconstructing the FiLM output tensor and
  comparing to the captured `film_scale_out`. That check was not run and
  should be if this decomposition is to be relied on further.

FILES WRITTEN (all uncommitted; base_train.py verified unchanged throughout)
  research/EXPERIMENT_LOG.md                   this entry
  diagnostics/probe_pe_film.py                 --film-out, modulate wrapper
  diagnostics/analyze_film_offset.py           offset/varying analysis
  logs/film_weights.json                       8 steps x 8 dims
  logs/pe_3000_film.json, logs/pe_3000_film.txt   per-step records, console log
  logs/film_report.txt                          full analysis output
  logs/smoke_film*.json/.txt, logs/smoke_film250*   smoke tests

## E13-fixed-3000-film-fix — axis correction, and the reconstruction residual

### PART 1 — the correct positions-only decomposition, and what it does to the conclusion

Reducing over POSITIONS ONLY and aggregating over units by RMS:

    offset_dj  = W[j,d] * E_mean_d          (per output unit j)
    varying_dj = W[j,d] * E_std_d
    offset_rms  = sqrt(mean_j (W[j,d] E_mean_d)^2) = |E_mean| * ||W[:,d]||_rms
    varying_rms = sqrt(mean_j (W[j,d] E_std_d)^2)  =  E_std  * ||W[:,d]||_rms

The column factor **cancels in the ratio**, so the correct measure is exactly
`|E_mean| / E_std` and is independent of the column norm.

| |E_mean| / E_std per dimension at step 2999 — E ONLY, no weights involved

    dim                E_mean      E_std    |E_mean|/E_std   class
    THREAT            -0.97706   0.002240        436.1        offset-dominated
    novelty           +0.79915   0.005730        139.5        offset-dominated
    uncertainty       -0.48726   0.041961         11.6        offset-dominated
    arousal           -0.17513   0.030284          5.8        in between
    learning_progress +0.10287   0.030251          3.4        in between
    homeostatic_dev   -0.04083   0.034020          1.2        in between
    integrity         +0.00852   0.023443          0.36       signal-dominated
    agency            -0.01200   0.056350          0.21       signal-dominated

**Threat is 1st of 8, not 6th.** The hypothesis that the threat channel acts
as a constant offset into FiLM is **SUPPORTED** by this measure, and the
previous entry's refutation of it was caused by the unit-flattened reduction.

Why threat and novelty dominate: both have a large |E_mean| AND a collapsed
E_std. threat's E_std fell 91.8% over the run (the largest proportional drop
of any dimension), and its column grew like everyone else's, so the product
`|col| * E_std` — the varying magnitude — fell to 0.0114, the smallest of all
eight. Small varying magnitude with a large offset magnitude is precisely the
signature of an offset-dominated channel, which is what the unit-flattened
ratio was hiding.

CAVEATS on this table, stated rather than buried:
- It uses **E alone** — no weights, so it is NOT behind the reconstruction
  gate and is not invalidated by it. But it is also not a statement about
  FiLM: it says what the state looks like, and FiLM's contribution scales it
  by `||W[:,d]||`, which happens to cancel only in the RATIO.
- It is **one batch at one step**, not a run average. The run-to-run spread
  documented in E13-fixed-3000 applies.
- The sign is carried: the ratio is |E_mean|/E_std so it is non-negative by
  construction, while E_mean itself is signed and shown.

### PART 2 — the per-unit decomposition: gate FAILED, then the gate was found to be wrong

The reconstruction gate (`arg = sum_d W[:,d] E_d + b`, compared against the
captured `film_scale_out` / `film_shift_out`) initially FAILED:

    max |sum-over-dims - module|   scale 3.207e-04   shift 3.431e-04
    max |sum-over-dims - captured| scale 3.207e-04   shift 3.431e-04
    captured - fresh recompute     scale 0.000e+00   shift 0.000e+00
    output absmax                  scale 8.116e-01   shift 1.325e+00
    relative residual              scale 3.4e-04     shift 2.7e-04

Ruled out at that point: a stale captured view (`captured_vs_fresh` is exactly
0.0); non-fp32 dtypes (weight, output and FiLM input are all `torch.float32`);
and plain fp32 accumulation order (a controlled two-ordering test differed by
only 1.4e-06 on outputs of magnitude 12.5, ~30x too small).

**RESOLVED in `E13-fixed-3000-film-residual` below: the forward pass runs with
TF32 enabled (`float32_matmul_precision='high'`, `allow_tf32=True`), and
replaying the reconstruction under those same settings reproduces the captured
output to 0.000e+00, bit-exact. The 1e-4 absolute gate was finer than TF32's
~4.9e-4 relative resolution can represent, so it was mis-specified rather than
the decomposition being wrong.**

**The per-unit quantities are therefore VERIFIED, not unverified:**
`offset_rms`, `varying_rms`, the per-unit offset/varying ratio, and the
per-unit absmaxes all reproduce FiLM exactly under the forward's own
arithmetic. The earlier "UNVERIFIED" marking on them is withdrawn.

## E13-fixed-3000-film-residual — the residual was TF32, not a decomposition error

**Verdict: hypothesis B-prime -- the forward pass's own arithmetic. A, B and C
are all ruled out. The decomposition was correct; the GATE was wrong to fail.**

The reconstruction is now **bit-exact (0.000e+00)** when replayed under the
precision the forward pass actually used. Measured on the 260-step smoke
configuration (not the 3000-step run, per instruction).

### THE THREE TESTS

| test | what it reconstructs | worst residual | verdict |
|---|---|---|---|
| **A** | float64, from forward-time weight clones | 3.916e-04 | FAIL vs 1e-4 |
| **B** | `F.linear` at `highest` + `allow_tf32=False` | 3.916e-04 | FAIL vs 1e-4 |
| **B'** | `F.linear` at the **forward's own** settings | **0.000e+00** | **PASS** |
| **C** | identity of the E tensor at check time | `torch.equal` True, `data_ptr` match True | PASS |

**A rules out weight drift.** Cloning `W_scale`, `b_scale`, `W_shift`,
`b_shift` inside the `modulate` wrapper (at forward time, same call) and
reconstructing in float64 from those clones still leaves 3.916e-04. Measured
directly: `W_scale_values_drifted_vs_live = 0.0` and
`W_shift_values_drifted_vs_live = 0.0`, so the live parameters had not been
updated between forward and check. Hypothesis A is not the cause.

**B rules out "just use higher precision".** Forcing
`float32_matmul_precision='highest'` and `allow_tf32=False` and recomputing
`F.linear` from the same clones gives **3.916e-04 — bit-identical to the
float64 result**. That is the diagnostic signature of the residual being *a
different arithmetic rather than a more accurate view of the same one*:
precision and TF32 changes move it by exactly zero, so neither is implicated.

**C rules out E substitution.** At every check, the E used in the
reconstruction is the same tensor `modulate` received: `torch.equal` True and
`data_ptr` identical (E shape `[4, 1024, 8]`, i.e. batch 4, not the 2 used in
an earlier standalone test — the two are not comparable, and the earlier
standalone scripts that "reconstructed at exactly 0.0" were producing
all-zero outputs and were vacuous).

**B' identifies it.** The environment read INSIDE the `modulate` call:

    device_name                NVIDIA GeForce RTX 5060 Ti
    capability                 [12, 0]        (Blackwell, sm_120)
    float32_matmul_precision   high            <-- NOT "highest"
    cuda_matmul_allow_tf32     True
    cudnn_allow_tf32           True
    autocast_enabled           False
    autocast_dtype             torch.float16   (configured, not active)
    E_dtype / W_dtype          torch.float32 / torch.float32

So the forward pass ran fp32 matmuls with **TF32 enabled** (precision `high`
implies TF32 for fp32 matmul on this architecture). Reconstructing under
exactly those settings reproduces the captured output to **0.000e+00** --
bit-identical, not merely close.

Independent confirmation on synthetic data with the same aspect
(768x8 matmul, E ~ O(1), 1024 positions):

    |out(high, tf32=True) - fp64exact| relative  4.3323e-04
    |out(high, tf32=True) - out(highest, tf32=False)| relative 4.3319e-04

The synthetic gap (4.33e-04) matches the observed relative residual
(3.45e-04 scale, 2.94e-04 shift) in magnitude, as it must if TF32 is the
whole story: TF32 carries 10 mantissa bits, giving ~4.9e-04 relative
resolution, which is the scale of the discrepancy.

### WHY THE ORIGINAL GATE FAILED, AND WHAT THAT MEANS

The 1e-4 absolute tolerance was applied to a comparison between a **TF32**
forward and a **float64** reconstruction. TF32's relative resolution is
~4.9e-4, so on outputs of magnitude 0.82 and 1.33 the expected absolute
discrepancy is 4.0e-4 and 6.5e-4. The observed 2.81e-04 and 3.91e-04 are
*below* that bound. **The gate was asking for agreement finer than the
arithmetic that produced the data can represent.** The gate was mis-specified,
not the decomposition.

Consequences, stated plainly:

- The per-unit decomposition (`offset_rms`, `varying_rms`, per-unit ratios) is
  **VERIFIED**, not merely unrefuted. It reproduces FiLM exactly under the
  forward's own precision.
- The E-only table in `E13-fixed-3000-film-fix` is unaffected and stands:
  threat `|E_mean|/E_std` = **436.1**, 1st of 8, so the threat channel **is**
  offset-dominated into FiLM. The refutation in the previous entry was an
  axis error and is retracted.
- **Training precision was not changed.** The `highest`/TF32 overrides in
  test B and B' are set and restored inside `try`/`finally`, and
  `flags_restored` / `flags_restored_after_tf32` are logged and verified True
  at every step. The forward pass's own settings are what the training uses.

RECOMMENDED GATE (not implemented -- no fixes proposed or applied): the
tolerance should be **relative** and set to the arithmetic's resolution, e.g.
`atol = 1e-3 * max|output|`, rather than a flat 1e-4 absolute. Under that, the
observed relative residuals of 3.45e-04 / 2.94e-04 pass on the correct
arithmetic and would still catch a genuine structural error, which would show
up as O(1) relative disagreement rather than O(1e-4).

FILES WRITTEN (all uncommitted; base_train.py verified unchanged throughout)
  research/EXPERIMENT_LOG.md              retraction banner, E-only table,
                                           this entry
  diagnostics/probe_pe_film.py            forward-time weight clones, env
                                           snapshot, tests A/B/B'/C
  logs/smoke_resid_film.json              260-step run, tests A/B/B'/C + env
  logs/smoke_resid.json                   per-step records
  logs/residual_report.txt                A/B/C + environment dump
  logs/tf32_test.txt                      independent synthetic confirmation

## E15 — end-of-run per-dimension ablation — **SEE CORRECTION BELOW**

Run 2026-10-01: seed 42, 3000 steps, `--abort-steps 10 --abort-bound 0.999`,
8 held-out val batches (B=4, T=1024), weights frozen, `torch.no_grad()`, no
optimizer step, carry state restored identically before every condition.
Full write-up: `research/E15_ABLATION.md`. Outputs: `logs/abl_3000_ablation.json`.

Two defects found before the numbers were trustworthy, both recorded in E15_ABLATION.md:

1. **The ablation was computed and then discarded.** The committed
   `--ablate`/`--ablate-out` code built the ablated tensor `_E_use` and then
   returned `_orig_modulate(x, E_t, ...)` — the ORIGINAL `E_t`. Every delta
   would have been exactly 0.0 and all eight dimensions would have falsely
   read as inert. Fixed in the same commit as this entry; when disarmed
   `_E_use is E_t` so the training forward is bit-for-bit unchanged.
2. **`per_batch_state` was recorded but never serialized** (`_eval_condition`
   returns only `ABL["state_used"]`). STILL UNFIXED as of this entry, so the
   reported per-sequence means are from the FIRST held-out batch only.

### CORRECTION to E15 — sign convention, super-additivity, and the noise floor

> **The reading of the E15 table in the first commit of this entry was wrong in
> three places. This block supersedes it. Nothing is deleted.**

**1. SIGN CONVENTION — negative delta = the loss DECREASED when the channel was
removed, i.e. removing the channel HELPED.**

    zero-ablate threat      -0.0017176   <- loss DROPPED; channel was HARMFUL
    zero-ablate novelty     -0.0007911   <- loss DROPPED; channel was HARMFUL

    >>> WITHDRAWN: the E15 first-pass conclusion that "position variation is
    >>> not used but the OFFSET IS USED" for threat and novelty. That reading
    >>> assumed a positive delta meant the channel was doing useful work. For
    >>> these two the deltas are NEGATIVE: zeroing threat made the loss
    >>> BETTER by 0.0017, and zeroing novelty made it BETTER by 0.0008. On
    >>> this checkpoint those channels are a net cost to the loss, not a
    >>> benefit. Only `uncertainty` (+0.0176988) and `agency` (+0.0016174)
    >>> have a clearly USEFUL offset.

    Full signed zero-ablate table (baseline loss 4.247949):
        uncertainty  +0.0176988  useful        arousal           -0.0004416
        agency       +0.0016174  useful        homeostatic_dev   -0.0000830
        integrity    -0.0000505                learning_progress -0.0000639
        novelty      -0.0007911  HARMFUL       threat            -0.0017176 HARMFUL

**2. SUPER-ADDITIVE, NOT 420x. The "420x" figure and its explanation are
WITHDRAWN.**

    sum of the 8 zero-ablate singles   +0.0161686
    zero_ablate_ALL                    +0.0317354
    ALL / largest single (uncertainty) = 1.793x
    ALL / sum of the singles           1.963x

    >>> WITHDRAWN: the E15 first-pass claim that zero_ablate_ALL is "420x the
    >>> largest single dimension" and that the effects "nearly cancel" because
    >>> "the columns are strongly anti-correlated". Both are wrong. 420x was an
    >>> arithmetic error on my part. The measured relationship is
    >>> SUPER-ADDITIVE: removing all eight at once costs ~1.8x the largest
    >>> single effect and ~2.0x the sum of the singles. That is the opposite
    >>> of cancellation. The pairwise interaction structure is NOT MEASURED;
    >>> no claim about it is made here.

**3. mean-ablate_ALL is +0.0004377, against a sum of the mean-ablate singles of
about -0.0002. Position variation is almost entirely unused** — this stands.
Per-dimension mean-ablate deltas span -2.3e-5 to +3.9e-5, i.e. 1e-7 to 1e-5
relative on a loss of 4.25. Read as "indistinguishable from zero at this
precision", NOT as "exactly zero".

**4. THE NOISE FLOOR OF 0.0 IS NOT SAMPLING ERROR.** baseline and baseline_repeat
are bit-identical because there is no backward pass in this evaluation, so the
GPU nondeterminism documented in E13-fixed-3000 does not apply. The floor of 0.0
reflects a DETERMINISTIC evaluation, and it bounds nothing about batch-to-batch
variation. **The `within_noise` field in `logs/abl_3000_ablation.json` is
MEANINGLESS and must not be read** — with a floor of 0.0 every nonzero delta is
flagged "not within noise".

**KNOWN GAPS, restated so they are not lost:**
- `per_batch_state` not serialized; per-sequence means are from batch 1 only.
- ONE seed, ONE checkpoint, 8 batches (32768 tokens), toy scale (depth 8,
  d_model 768). No seed replication.
- The interaction structure behind super-additivity is unmeasured.

## NOT YET RUN

**E14 — 3-seed B' A/B comparison.** GATE MET by E13, but not launched.
Awaiting explicit go-ahead. Do not launch seeds 43/44 without instruction.

**From E15 (2026-10-01): no B' A/B has ever run on post-leak-fix code.** The
withdrawn 1.226305 came from `b-stab-s42` at commit `6ae93ce`, which is the
commit immediately BEFORE the fix `969e067`. Read-only detail in
`research/AB_SETUP_READONLY.md`.

## E17-AB-v2 (post-leak-fix A/B) — **B' BEATS BASELINE; the effect is real**

> ID note: this is **E17**, not E16. E16 is referenced five times in this file
> as "the E16 anomaly" (the PE plateau, still open) but has no entry heading of
> its own, so the number is treated as taken.

Date: 2026-10-01. Sweep commit `f2729386cdb74930a520914a90ba541ed6f459dd`.
Pre-flight `diagnostics/test_no_label_leak.py` **PASS**, exit 0, all three
causality checks. `git merge-base --is-ancestor 969e067 HEAD` confirms the leak
fix is in HEAD. Driver `run_ab_v2.ps1`.

METRIC: validation BPB at the FINAL step (600), parsed from
`Step 00600 | Validation bpb:` (`scripts/base_train.py:740`). Deliberately NOT
`Minimum validation bpb` (line 1065), which is the best checkpoint. Both arms
use the identical `evaluate_bpb(model, val_loader, eval_steps, token_bytes)`
path. The `{:05d}` format renders step 600 as `00600`, not `06000`.

Flags, identical in both arms except `--rda`: `--depth=8 --aspect-ratio=96
--head-dim=128 --max-seq-len=1024 --window-pattern=L --device-batch-size=4
--total-batch-size=32768 --num-iterations=600 --eval-every=50
--eval-tokens=524288 --core-metric-every=-1 --sample-every=-1`.
So d_model = 768, T = 1024. Order interleaved to limit machine drift:
A42 B42 A43 B43 A44 B44 A42-repeat.

| seed | baseline | B' | paired diff (B' - baseline) |
|---|---|---|---|
| 42 | 1.300643 | 1.278289 | **-0.022354** |
| 43 | 1.287503 | 1.274878 | **-0.012625** |
| 44 | 1.294832 | 1.275739 | **-0.019093** |

    baseline s42 repeat   1.300576   difference -0.000067
    mean |paired diff|               0.018024

VERDICT (criterion declared before the sweep: all three paired diffs the same
sign AND mean |diff| above both the repeat difference and 0.0137):

    all three same sign      YES (all negative)
    mean |diff| > |repeat|   0.018024 > 0.000067   (269x)
    mean |diff| > 0.0137      YES
    ==> EFFECT IS REAL

**SEED 43 ALONE IS BELOW 0.0137** (-0.012625). It clears only because it is
averaged with seeds 42 and 44. A single-seed B/B' comparison at this step count
would NOT have established the effect.

**THE REPEAT IS NOT THE SEED SPREAD.** The s42 repeat measures run-to-run
NONDETERMINISM on ONE seed and came out at 0.000067. The 0.0137 figure is
seed-to-seed variation across three DIFFERENT seeds (E1). They are different
quantities and are not interchangeable; the criterion uses both because they
bound different things. The 269x ratio between them is a fact about this sweep,
not a general constant.

**EQUAL STEPS ARE NOT EQUAL COMPUTE.** Baseline runs took 12.2-12.3 min;
B' runs took 24.1-29.1 min, i.e. roughly 2.4x. The B' arm is doing
substantially more work per step (the FiLM path plus the vectorized recurrence),
so this A/B compares two different compute budgets, not two architectures at
matched cost. A cost-matched comparison is NOT what was run and would be a
different experiment.

### TWO INFRASTRUCTURE FINDINGS

**1. A LATENT NameError MADE THE BASELINE ARM UNRUNNABLE.** `base_train.py`
referenced `_rda_diag_lr_scale` at line 941, in the per-step LR rescale that
runs for EVERY step of EVERY run, but defined it at line 680 nested inside
`if rda_active:` (line 536). Any run WITHOUT `--rda` died at the first optimizer
step with `NameError: name '_rda_diag_lr_scale' is not defined`.

`git log -S"_rda_diag_lr_scale" -- scripts/base_train.py` returns exactly ONE
commit, `ab9b560` ("Stabilize the RDA state", 2026-09-28), and `git blame`
attributes both halves to it. Its diff shows
`-group["lr"] = group["initial_lr"] * lrm` becoming
`+group["lr"] = group["initial_lr"] * lrm * _rda_diag_lr_scale`, with the
definition indented under the RDA block. Invisible to the arm being run at the
time, because `b-stab-s42` passed `--rda`.

Fixed in `f272938` by initialising it above the `if rda_active:` block. Default
stays 1.0, so the value reaching line 941 is identical for `--rda` runs (same
default, same `RDA_STATE_LR` override path), and a baseline's learning rates are
UNCHANGED (`initial_lr * lrm * 1.0` is the pre-`ab9b560` expression).

**CONFIRMED EMPIRICALLY BY THIS SWEEP:** the three baselines reproduce E1 to
within 0.00057 (1.300643/1.287503/1.294832 vs 1.301217/1.287544/1.295109).
That is independent evidence the fix changed no baseline arithmetic.

**2. THE OLD BASELINE LOGS WERE NOT TRUNCATED — THEY ARE MIXED-ENCODING.**
This corrects an earlier claim in this work that `logs/a_s42.log`, `a_s43.log`
and `a_s44.log` were "truncated" by a pipe closing early. They are NOT
truncated and did NOT fail: each ran the full 600 steps and ends with
`Minimum validation bpb:` 1.301217 / 1.287544 / 1.295109, `Total training time:`
9.98 / 9.94 / 9.93 m, and a saved `optim_000600_rank0.pt`. They ran at commit
`030a384`, which PREDATES `ab9b560`, which is why they were unaffected.

They *appeared* empty because each file is MIXED-ENCODING: a UTF-8 header
written by `Set-Content -Encoding utf8`, followed by a UTF-16LE body appended by
`Tee-Object`. Read as UTF-8 you get the header and nothing else; read as
`[System.Text.Encoding]::Unicode.GetString(bytes)` you get everything. That is a
third defect, in `run_sweep.ps1`'s logging, not in training. `research/
AB_SETUP_READONLY.md` has been corrected accordingly.

### WHAT THIS DOES AND DOES NOT SHOW

**IT DOES NOT SHOW THAT THE 8 STATES MATTER.** B' adds parameters AND a
recurrence AND the FiLM path relative to the baseline. This A/B measures the
effect of that WHOLE package. It cannot separate "the 8 regulatory states carry
information the model uses" from "the extra capacity / extra compute / the
extra pathway helps". Any causal claim about the states requires a control that
keeps the parameters and the recurrence but removes the INFORMATION, which this
entry does not contain.

**SCOPE CAVEAT.** 600 steps, depth 8, d_model 768, T=1024, 19.7M tokens, one
RTX 5060 Ti. Not production scale. Per E13-fixed-3000 the `threat` drive is
already pinned against the lower rail by ~step 1000-1350 at this scale, and
this A/B never ran past step 600, so it says NOTHING about whether the state
stays healthy at longer horizons.

LIMITATIONS
- Single machine. Runs are not bit-reproducible (GPU nondeterminism in the
  backward pass), which is why the repeat exists.
- Not cost-matched (see above).
- The three seeds share one shard order and one dataset; no data variation.
- Only three seeds. The per-seed spread of the paired diffs is
  0.022354/0.012625/0.019093, a range of 0.0097, which is comparable to the
  effect itself and would not survive many more seeds without tightening.

FILES: `run_ab_v2.ps1`, `dry_test_wrapper.ps1` (both committed);
`logs/ab_v2_<label>_s<seed>.txt` x7, `logs/ab_v2_results.json`,
`logs/ab_v2_smoke_A.txt`, `logs/ab_v2_smoke_B.txt`,
`logs/ab_v2_aborted_attempt1.txt`, `logs/ab_v2_crash_attempt2.txt`,
`logs/dry/` (uncommitted, gitignored).


---

## E18-ARM-C-DIAG — arm C is 2.5x slower than B'' per step; cause OPEN

Diagnostic only. Two independent pairs of 60-step runs, identical flags to the
real runs except `--num-iterations=60` and `--eval-every=50` kept (evals do not
change training numerics). No model, gain, drive or training code was changed.

### (a) Measured pace and GPU state

| | B' s42 | C s42 |
|---|---|---|
| exit | 0 | 0 |
| wall, pair 1 / pair 2 | 2.95 / 2.88 min | 5.92 / 5.97 min |
| trainer `dt` (steps >=5, n=55) | 2.100 / 2.013 s/step | 5.162 / 5.181 s/step |
| tok/sec step 0 | 18,844 | 19,336 |
| tok/sec step 10 | 16,706 | 7,446 |
| tok/sec step 30 | 16,853 | 6,718 |
| tok/sec step 59 | 16,709 | 7,117 |
| peak VRAM | 7,727 MiB | 7,744 MiB |
| VRAM trajectory | 356 -> 3166 -> 7727 -> flat | 335 -> 3166 -> 7727 -> flat |
| mean utilisation | 88.8% | 93.9% |
| mean SM clock | 2,738 MHz | 2,835 MHz |
| mean power | 105.1 W | 70.9 W |
| max temperature | 66 C | 63 C |

Reproduced in two pairs; the C/B' ratio is stable to within 3% across four runs
(2.5x, 2.57x, 2.46x, 2.57x). Sampling via `diagnostics/diag_gpu_sample.ps1`
at 15 s into `logs/diag2_<arm>_gpu.csv`.

C is arm-specific, NOT environmental: B' at 2.0-2.1 s/step matches ab_v2's
20.7-25.9 min for 600 steps. A machine-wide slowdown would show in both arms.

### (b) RETRACTION of the "GPU memory pressure" explanation

An earlier pace report attributed C's slowness to VRAM pressure (96% of 8 GB,
repeated allocator flushes). **That explanation is withdrawn.** Memory is high
but STATIC and effectively IDENTICAL across arms: both arms reach ~7.7 GB
within the first ~30 s (allocator warm-up) and then hold flat to the last
sample, with peaks differing by 17 MiB (7727 vs 7744) and a final wobble of
+/-17 MiB that is noise, not a trend. There is no growth and no asymmetry to
explain a progressive 2.5x slowdown.

Also ruled out by measurement: sysmem/shared-memory fallback. `Shared Usage`
totalled 14.5 MiB, so the card did NOT spill to shared memory. Thermal
throttling is excluded (63-66 C, near-max clocks). HAGS is not set, i.e.
Windows default, and is not a lever here. Idle VRAM with nothing of ours
running was 507 MiB, so there was no pre-existing contention: the ~7.7 GB is
the model's own footprint.

The earlier 9.06 s/step wall figure (and the ~6 h projection built on it) came
from a log that was block-buffering and therefore reporting stale step counts.
At the measured 5.18 s/step, 600 steps projects to ~52 min, inside the new
90-minute cap.

### (c) CORRECTION — gradient pruning in C is narrower than previously stated

E17-Arm-C-setup and the earlier session notes recorded that in arm C "backward
does not traverse the drive path" and that C's per-step cost sat near
baseline's. **Both are wrong, and the second inference was drawn from the
first.**

What is actually true: replacing `film_state` with a constant removes the
loss gradient to `event_proj`, `pe_proj` and `h_norm`, and autograd prunes
those backward edges (verified: `p.grad is None`, not a zero tensor). That
pruning does NOT remove the FORWARD work. `drive_seq`/`update_seq`
(`rda.py:951`), `predict_seq` (`rda.py:954`) and the `dpe` computation still
execute every micro-step, and forward-call counts are identical in C and B'
(1 call/step each for `update_seq`, `predict_seq`, `event_proj`, `drive_seq`,
`modulate`). In C that work builds a graph that is then discarded, so C pays
for the forward and skips only part of the backward.

### (e) OPEN — why is C 2.5x slower per step?

Not identified. Candidates, neither confirmed:
  - wasted forward work building a dead graph (per (c));
  - host-side synchronisation.

Signature is suggestive but not decisive: C shows HIGHER utilisation (93.9% vs
88.8%), HIGHER clocks (2835 vs 2738 MHz) and LOWER power (70.9 vs 105.1 W),
which reads as waiting rather than computing. High util + high clocks + low
power + halved throughput. A targeted profile is needed to separate the two;
no profiling was run, so neither is asserted.

NOT BLOCKING the A/B: validation BPB at a fixed step count is unaffected by
wall-clock pace, and the C/B' comparison is like-for-like at equal steps.

### (f) The grad-free RDA modules are "untrained AND decayed", not "at init"

In arm C, `event_proj`, `pe_proj` and `h_norm` receive NO gradient (autograd
pruned) and are added to the grad-free list only so
`MuonAdamW._compute_adamw` does not crash calling `grad.float()` on `None`
(`optim.py:73`). They are NOT frozen. `grad_free` is read only by the zero-grad
priming at `base_train.py:935-940`; `optim.py` never sees the key, and
`adamw_step_fused` applies decoupled decay unconditionally
(`optim.py:75`, `p32.mul_(1 - lr*wd)`) before examining the gradient.

Measured over 3 steps at lr 3e-3, wd 0.01 (`diagnostics/probe_arm_c_decay.py`):
all six tensors shrank by exactly `(1 - 3e-5)^3`, observed ratio 0.999910178
vs predicted 0.999910003, matching to 1.8e-7. `h_norm.bias` did not move only
because it is 0.0 to begin with.

CORRECTION to an earlier claim in this session that these are "bit-identical
to init": that was an artifact of `probe_arm_c.py` excluding them from its own
optimizer with `weight_decay=0.0`, which is not the training configuration.
An earlier comment in `base_train.py` asserting they are "neither updated nor
decayed" was wrong and has been rewritten.

### (d) CORRECTION — the step-60 BPB difference is not evidence either way

Recorded correctly as measured: at 60 steps C 1.904722 vs B' 1.904281,
difference 0.000441 (pair 2: 1.904637 vs 1.904288, difference 0.000349).

An earlier note characterised that as "not what you would see if the states
were contributing". **That characterisation is withdrawn.** C being worse than
B' is the direction CONSISTENT with the states contributing; the size at 60
steps is negligible and well inside run-to-run noise, so it is not evidence in
either direction. The A/B verdict must come from the 600-step three-seed
comparison, not from this 60-step diagnostic.

### Driver changes made for this phase (not model/gain/drive code)

- `run_ab_v3.ps1`: 90-minute per-run timeout (`c103606`). On expiry kills the
  process TREE (`Stop-Process` plus `taskkill /T`, because the venv shim spawns
  the real interpreter as a child and killing only the parent orphans it and
  keeps the GPU held), appends `TIMEOUT after 90 min`, returns 124, and records
  `halted.timed_out` in the results JSON before stopping the sequence.
  Two bugs found by self-test rather than assumed away: `Wait-Process
  -Timeout` raises when the PID is gone, which misreported an instantly-exited
  process as a timeout (replaced with a `HasExited` polling loop); and
  `Start-Process -PassThru` reports `ExitCode` as `$null` unless `$p.Handle` is
  cached first (verified: `cmd /c exit 5` gave `[]` uncached, `[5]` cached).
- `run_ab_v3.ps1`: `PYTHONUNBUFFERED=1` set on `$env` so it is inherited by the
  child. Verified by A/B with a payload printing one line per second WITHOUT
  flush: unset = 0 lines readable while running, set = live (1,2,3,4 lines at
  t=+1..4 s). An earlier version of that probe used `flush=True` and so
  reported a false pass in both arms; the probe had to be fixed before it could
  discriminate.

---

## Concept Notes

The entries above are chronological. For the mechanism each one investigates,
see:

- [[RDA and Persistent State]] — the architecture across all entries
- [[Event Signal]] — E7, E8: saturation investigation and the output-normalisation fix
- [[Prediction Error]] — E3, E14: the PE pathway and the leak fix
- [[FiLM Modulation]] — E10: the C1–C6 ablation suite
- [[State Saturation]] — E11, E13-fixed: stabilisation evidence
- [[State Prediction]] — E9: the dependency graph that tracked `dpe`
- [[Prediction vs Actual State]] — the `dpe` signal
- [[Label Leakage]] — E2, E3, E14: the retractions
- [[Ablation Study]] — E10, E15
- [[Control Arm]] — E17, E18
- [[BPB]] — the metric every entry reports
- [[Weight Decay]] — E18: the decay measurement
- [[Optimizer State]] — E6: the TREE provenance check

Retracted entries (E2, E13-SUPERSEDED, E14) are documented in
[[FAILED_EXPERIMENTS]].
