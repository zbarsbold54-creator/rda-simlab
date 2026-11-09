# EXPERIMENTS

Experiments **as specified**. Results live in `../research/EXPERIMENT_LOG.md`
and will be reported in `RESULTS.md` only after being run.

Each experiment below states its status honestly.

---

## E13 — 1000-step stability confirmation — **NOT RUN**

```
Status:  NOT RUN. THIS IS THE GATE.
Question:
  Is the PE/FiLM saturation mechanism controlled, or merely delayed?
Why 1000 and not 250:
  Saturation in the unstabilized configuration first crossed |E| = 0.98 at
  step 217. Evidence at 250 steps covers ~15% past onset and cannot
  distinguish a mechanism that is gone from one that is slower.
Method:
  diagnostics/probe_pe_film.py, 1000 steps, seed 42, production rda.py
  with no ablation flags. The probe prints which implementation is active
  and asserts zero hooks remain.
Command:
  cd C:\AI\nanochat
  $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
  $env:PYTHONUNBUFFERED="1"; $env:PYTHONPATH="C:\AI\nanochat"
  $env:NANOCHAT_DATASET="fineweb-edu"
  .\.venv\Scripts\python.exe diagnostics\probe_pe_film.py --steps 1000 `
    --label "E13 1000-step production confirmation" --out logs\pe_1000.json
Report:
  first step |E| > 0.98, max|E|, FiLM gain max, loss curve.
Pass condition:
  |E| never exceeds 0.98, and the FiLM gain stays near 1.5 rather than
  trending upward.
On failure:
  Do NOT run the A/B. The next suspect is PE itself, which still grows to
  ~36 and is not understood. It was harmless once FiLM was bounded, so it
  should not be touched until FiLM is confirmed stable.
```

---

## E14 — 3-seed B' A/B comparison — **NOT RUN, BLOCKED**

```
Status:  NOT RUN. Blocked on E13.
Question:
  Does the persistent state improve validation bits-per-byte relative to
  the baseline, by more than the baseline's own seed spread?
Method:
  Seeds 42, 43, 44. Identical configuration to the baseline runs apart
  from --rda. TREE must be CLEAN. Saturation abort must be enabled.
Pass condition (pre-registered, see METHODS.md section 5):
  mean(B') - mean(A) is larger in magnitude than 0.0137, in the
  hypothesized direction, AND no B' run saturated.
On a negative result:
  Report "not supported at this scale". NOT "refuted" — three seeds and a
  small model cannot refute a general claim.
On a positive result:
  Report it WITH the spread, with the seed count, and with the
  qualification that three seeds at one scale is weak evidence.
Known trap:
  The project's first result was 1.286248, which was inside the baseline
  range and came from a leaky model. Any new number must be checked
  against BOTH failure modes: is it outside the spread, and is the model
  causal?
```

---

## E15 — State interpretability — **NOT DESIGNED**

```
Status:  NOT DESIGNED. Requires a trained B' run.
Question:
  Do the 8 coordinates differentiate into distinguishable roles, or
  collapse toward redundancy?
Would involve:
  - per-dimension correlation matrix of E over a trained run
  - correlation of each coordinate against next-token entropy,
    perplexity delta, and document-boundary proximity
  - single-coordinate ablation: damage one drive, observe what changes
Blocked on:
  E14, because interpretability of an untrained state is uninformative.
Note:
  This is the experiment that would give the dimension NAMES any
  justification. Until it is run, the names are a hypothesis (H2).
```

---

## E16 — PE anomaly — **NOT RUN, OPEN**

```
Status:  NOT RUN. Unexplained observation.
Observation:
  PE grows from 10.5 to ~36 over 250 steps in the stabilized
  configuration. It no longer drives state saturation, but its growth is
  not explained.
Question:
  Is this the model becoming more confident on some tokens and much less
  confident on others, or is it a sign of something structural?
Would involve:
  - distribution of PE, not just its mean and max
  - per-token PE against per-token loss
  - whether PE growth tracks the LR schedule
Why it matters:
  An unexplained growing input to a state signal is a poor thing to leave
  unexamined before drawing conclusions about that signal's usefulness.
  See hypothesis H3.
Priority:
  After E13, and only if E13 is clean. Do not modify PE before FiLM
  stability is confirmed.
```

---

## E17 — dpe as an interpretability instrument — **NOT RUN**

```
Status:  NOT RUN. Deliberately unconsumed (DECISIONS.md, Decision 4).
Question:
  Is the state predictable? If dpe is low, the recurrence has found
  structure; if high, the dimensions may be arbitrary.
Would involve:
  mean dpe over training, compared across A and B' and across seeds
Note:
  dpe is currently computed and logged but consumed by nothing, which is
  what makes it usable as a clean instrument. Consuming it would
  immediately confound the measurement.
```

---

## Already completed

Recorded in full in `../research/EXPERIMENT_LOG.md`:

| ID | Description | Outcome |
|---|---|---|
| E1 | Baseline A, 3 seeds | mean 1.294623, spread 0.0137 |
| E2 | Initial RDA result | **RETRACTED** — single seed, inside baseline range, leaky |
| E3 | Two temporal leaks found and fixed | commit `bc3deb8` |
| E5 | Vectorized recurrence | ~200x, verified to 3.33e-16 |
| E7 | Event saturation investigation | cause isolated: `event_proj` gradient |
| E8 | Event-output normalization | mechanism controlled, \|\|W\|\| now shrinks |
| E9 | PE/FiLM graph mapping | closed but gradient-dormant at step 0 |
| E10 | PE/FiLM ablations C1-C6 | cause isolated: FiLM amplification |
| E11 | Production-code verification | never crosses 0.98 in 250 steps |
| E12 | Regression tests | 18/18 pass |

The two most substantial results so far are negative. They are about
whether the architecture is *valid*, not about whether the hypothesis
holds.
