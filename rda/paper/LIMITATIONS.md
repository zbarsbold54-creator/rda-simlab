# LIMITATIONS

What this work cannot claim. Written to be read before the paper is
judged, so that the boundaries are visible rather than discovered.

---

## 1. The stabilization evidence does not cover enough time

**The single most important limitation.**

Saturation in the unstabilized configuration first crossed `|E| = 0.98` at
step **217**. The evidence that the fix works runs to **250 steps** — about
15% past onset.

That is not enough to distinguish a mechanism that is *gone* from one that
is *slower*. A ratchet that took 217 steps to cross a threshold could,
after a 2x suppression, take 434 — still comfortably inside a training run
of thousands of steps, and entirely consistent with the 250-step result.

**What would fix it:** the 1000-step run (E13). It is not optional and it
has not been run.

**Residual risk even after E13:** 1000 steps is still shorter than the real
training runs. The stronger control is to instrument the actual training
run with the saturation abort enabled, which already exists.

---

## 2. One seed

Every stabilization number in this project comes from seed 42. No variance
estimate exists for the stabilized configuration.

For the *ablation* comparisons this is less serious, because the
contrasts are large (max|E| 0.9853 vs 0.9402; FiLM gain 9.568 vs 1.522)
and the mechanisms are structural rather than statistical. For the
*forward-looking claim* that B' is stable, it is serious.

> **Both contrast numbers above were measured on the LEAKY code** (pre
> `969e067`) and are retained only as evidence that the runaway mechanism
> existed and was bounded. They are not measurements of the fixed
> architecture. The forward-looking stability claim is in fact **not**
> supported: see §4b — the fixed code FAILS its 1000-step trend gate.

---

## 3. One scale, and it is not the treatment scale

Stability was established at depth 8, d_model 768, T=1024. The final B'
configuration is larger. There is no reason to expect the same behaviour,
and no evidence that it will be the same.

> The 1000-step confirmation this refers to is E13, which was measured with
> the label leak and is superseded. On the fixed code the same 1000-step
> configuration does not clear its trend gate — see §4b.

A mechanism that depends on a gain being free to grow could reappear at a
different scale if the gradient magnitudes differ.

---

## 4. The primary question is unanswered

The claim that a persistent low-dimensional state improves a language
model has **not been tested**. The one run that appeared to support it was
withdrawn for two independent reasons: it was a single seed whose value
(1.286248) fell inside the baseline's own range (1.287544–1.301217), and it
was produced by a model with two temporal leaks.

Everything this project has established is about whether the architecture
is *valid*, not about whether it is *useful*.

---

## 4a. A third leak, found later, invalidated every B′ number

The two leaks above were in the FiLM pathway and were found and fixed in
`539007e`. A **third** leak was found on 2026-09-29, in the PE pathway, and
it is the reason this limitation is stricter than it was.

`rda.py` built the per-position prediction error from `targets[:, 1:]`.
Because `pe_seq` is indexed by input position and `E_next_seq[j]` is the
state after consuming position `j`, this made `pe_seq[j]` a function of
`y[j+1] = x[j+2]`. With `film_state[p] = E_next_seq[p-1]`, the drive state
modulating position `p` had already seen `x[p+1]`.

The dataloader makes `y[p] == x[p+1]` exactly (verified, fraction
1.000000), and the loss is `cross_entropy(logits[p], targets[p])` with no
shift. So `logits[p]` is the distribution for `x[p+1]` and is graded
against `x[p+1]`. **One token of the answer reached the model.**

Fixed in `969e067` by indexing `pe_seq` as `targets[:, :-1]`.

**Withdrawn as a result:** the B′-s42 val BPB of 1.226305 against a
baseline of 1.301217 — a 0.0749 gain, larger than the 0.0137 seed spread
and therefore nominally significant. It was an artifact. Every B′ run
between `ab9b560` and `6ae93ce` is affected, including that canary and the
E13 stability confirmation.

**Why this is a limitation and not just a bug.** The leak improved the
metric. A model that can see one token of the answer scores better than one
that cannot, so the failure mode is not "training diverges" — it is
"training looks fine and the number goes the right way for the wrong
reason." The three existing causality checks all missed it because they
perturbed `x` and never `y`; two of them had controls that *asserted the
leaky behaviour*, so they were written to pass with it present. A green test
suite was not evidence of correctness.

**What is unaffected:** the A baseline. Those runs never passed `--rda`,
and `rda.py` is imported only under `--rda`, so no baseline number moves.
Only B′ comparisons are void.

**Standing risk.** `diagnostics/test_no_label_leak.py` is now a hard
pre-flight gate in `scripts/base_train.py`, so a run cannot start on
leaking code. It is checked in the entry point rather than only in the
gitignored `run_sweep.ps1`, because a gate that does not travel with the
repository is not a gate. The escape hatch `RDA_NO_LEAK_CHECK=1` exists for
debugging the test itself and prints a warning that the run is not
reportable.

---

## 4b. The fixed code has not yet cleared its own stability gate

The 1000-step confirmation re-run on the fixed code (`E13-fixed`) is
**incomplete**. Two gates stand and one is withdrawn.

**Stand:** `max|E|` peaks at 0.9690 against a 0.98 bound — a margin of
0.011 — with zero steps over the bound, and the loss falls normally
(10.3983 → 5.1722). The pre-registered criterion `|E| <= 0.98 for all
steps` is satisfied.

**Also stands, as an observation rather than a gate:** `max|E|` *rises*
across five consecutive 100-step windows from step 500 onward, where the
leaky E13 run decayed. The windowed-trend criterion was formulated **after**
seeing the fixed run, so it is not pre-registered and is not presented as
though it were. It is nonetheless the observation most warranting follow-up,
being the same self-excitation signature the original criterion was written
to catch.

**Withdrawn:** the two gain gates. The probe recorded the *raw* FiLM
projection, not the applied gain — `_unit_rms` is applied inside
`modulate()`, after the forward hook that captures the module output, and
the two differ by roughly 2× (measured: `max|1+fs|` = 1.12–1.21 against
`max|1+_unit_rms(fs,0.5)|` = 2.53–2.67). The applied gain was never stored
and cannot be reconstructed from the aggregates that were. The probe now
records it (`a5eba19`), and re-measurement is in progress. Three
successive readings of this metric were all wrong before it was pinned
down; the sequence is recorded in `research/EXPERIMENT_LOG.md` because the
error is the kind that a confident, plausible-sounding number hides.

**E16 remains open.** The PE-plateau anomaly is not explained by the leak:
the fixed run plateaus *earlier and lower* (max 33.16, final 20.14) than
the leaky run (max 38.86), so PE growth is not leak-driven.

---

## 5. Three seeds is a weak experimental design

The baseline spread of 0.0137 is estimated from n=3, so the spread estimate
is itself unstable. The pre-registered comparison rule ("a difference
smaller than the spread is not a result") is a floor on rigour, not a
substitute for adequate power.

At n=3, a real but small effect could easily be missed. A null result here
should be read as "not demonstrated at this scale", never as "refuted".

---

## 6. Related work has not been surveyed

The central design claim — that a small persistent summary adds information
a bounded context window does not carry — has not been checked against the
recurrent-model literature. See `RELATED_WORK.md`.

This is a limitation on the *framing*, not just on the citations. The
design may be redundant with, or already refuted by, prior work, and that
has not been determined.

---

## 7. The dimension names are unjustified

The 8 coordinates are named `novelty`, `uncertainty`, `threat`, `integrity`,
`agency`, `learning_progress`, `arousal`, `homeostatic_dev`. Nothing
supervises them and no interpretability experiment has been run.

They are motivated names, and the names are doing real work in the
write-up — they set what a reader expects to see. Until E15, they are a
hypothesis about what the coordinates will acquire, not a description of
what they are.

They are explicitly **not** emotion labels, and no correspondence to human
affective categories is claimed or tested.

---

## 8. PE behaviour is unexplained

PE grows from 10.5 to ~36 over 250 steps in the stabilized configuration.
It no longer drives state saturation, but why it grows is not understood.

An unexplained growing input to a state signal is a genuine problem for
any claim about that signal's usefulness, because the state is being
driven by a signal whose dynamics are unknown. See E16.

---

## 9. `dpe` is unconsumed and uninterpreted

`dpe` is computed and logged and nothing uses it. This is deliberate (see
`../research/DECISIONS.md` Decision 4) — it is preserved as a clean
instrument — but it means the project has no evidence about whether the
state is structured or arbitrary.

---

## 10. Bits-per-byte may be the wrong metric

Validation bits-per-byte measures predictive quality. If the persistent
state helps with calibration, adaptation, or anomaly sensitivity, this
metric will not detect it.

That is a possible **false negative** mechanism in the primary
experiment, and it is not addressed by running more seeds. It is worth
pre-registering how such an outcome would be interpreted before the runs
are launched, precisely so it cannot be reinterpreted afterwards as a
positive.

---

## 11. The measurement instruments failed in ways that were not obvious

Two controls in this project returned results identical to their
comparisons in every digit, because the experiment was not connected to
the thing it was changing. Both were caught only by treating implausible
exact equality as a signal.

This is recorded because it bounds confidence in the positive results
too. The same class of error could produce a result that looks
*different* when it should be identical, and nothing in the current test
suite would distinguish that. See `../research/FAILED_EXPERIMENTS.md`.
