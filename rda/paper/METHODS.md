# METHODS

Experimental protocol. This file specifies what will be done; it does not
report what was found.

---

## 1. Systems

### 1.1 Baseline (A)

The unmodified nanochat path. Identical code, identical hyperparameters,
`--rda` not passed. RDA parameters do not exist in this configuration, so
there is no possibility of the treatment leaking into the control.

This is the same code path as upstream, tracked as the `upstream` remote.
Its only project-specific change is the added `--seed` flag and run
manifest, which do not affect computation.

### 1.2 Treatment (B')

RDA enabled, on the stabilized architecture:

- event projection output RMS-normalized
- FiLM modulation vector RMS-normalized to `film_rms = 0.5`
- per-position causal FiLM
- **per-position prediction error indexed as `targets[:, :-1]`, so the drive
  state at position `p` has not seen `y[p]`** (added in `969e067`; see
  LIMITATIONS §4a. Before this, `targets[:, 1:]` gave the state a one-token
  view of the label it was scored on, and every B′ number measured with it is
  withdrawn)
- decay-only persistence, `decay = 0.95`

Everything else is identical to A. In particular B' does **not** change
the learning rate, the weight decay, the batch size, the data, or the
tokenizer.

### 1.2a The label-blindness invariant

The PE pathway is the only route by which the targets reach the model, so
one condition governs the whole treatment:

    film_state[p] may depend on y[j] only for j < p.

Since `y[j] == x[j+1]`, this is equivalent to `film_state[p]` depending only
on `x[0..p]` — exactly what a causal transformer position is entitled to see.
`diagnostics/test_no_label_leak.py` enforces it and runs as a hard
pre-flight gate in `scripts/base_train.py`. It fails on `6ae93ce` (12
violations) and passes on `969e067`.

### 1.3 The identity-at-init property

Because the FiLM weights are zero-initialized and remain zero under the
normalization, B' is **bit-identical to A at step 0**. The two systems
diverge only as the FiLM parameters receive gradient. This is verified
(max |diff| = 0.00e+00) and is a useful check that the treatment is doing
what it claims: it does not perturb initialization.

---

## 2. Configuration

Held identical across A and B' and across seeds:

| Parameter | Value |
|---|---|
| dataset | FineWeb-Edu, local shards |
| max sequence length | 1024 |
| total batch size | 32768 |
| device batch size | 4 |
| gradient accumulation | 8 |
| depth | 8 (diagnostic configuration) |
| window pattern | `L` |
| optimizer | MuonAdamW, as upstream |
| warmup | 40 steps |
| seeds | 42, 43, 44 |

The diagnostic configuration (depth 8, d_model 768) is **not** the final
B' configuration. Stability was established at this scale only; the
production configuration must be re-verified.

### 2.1 A configuration trap already hit once

An early run omitted `--eval-tokens`, silently defaulting to 80x the
intended evaluation budget, and appeared to hang for 74 minutes. Any
change to a training command line must be diffed against the baseline's
command line before the run is allowed to proceed unattended.

---

## 3. Metrics

### 3.1 Primary

Validation bits-per-byte, from the same evaluation suite as the baseline.

### 3.2 Stability diagnostics (recorded every step, not just at the end)

- `max|E|` and per-dimension `|E|`
- post-tanh drive magnitude
- FiLM gain and shift magnitude
- PE magnitude and per-dimension PE contribution
- `event_proj` weight norm, `pe_proj` bias norm
- gradient norms for each parameter group
- loss

### 3.3 The comparison rule

The claim under test is that B' differs from A. With three seeds, the
baseline's own spread is **0.0137** (max − min over 1.287544, 1.295109,
1.301217).

> A difference smaller than the seed-to-seed spread is **not** a result.

This rule is not a formality. It exists because the project's first
headline number, 1.286248, was smaller than the baseline's spread and was
withdrawn. Any future claim must state the spread alongside the mean, and
must state whether the difference exceeds it.

---

## 4. Validity controls

### 4.1 Causal validation

- No future information may reach any position. Tested by perturbing
  `targets[:, k]` and confirming every state position `< k` and every
  hidden position `< k` is bit-identical.
- The PE pathway must not introduce a one-token lookahead. Tested
  directly; `PE == NLL(logits[:T-1], targets[1:])` to 9.5e-07.

### 4.2 Run provenance

Every run reports a TREE state, CLEAN or DIRTY. A DIRTY run did not
execute the committed code alone and must not be reported as a result.

### 4.3 Saturation abort

If any drive dimension holds `|E| > 0.98` for 10 consecutive steps, the
run aborts and names the offending dimensions. A saturated tanh cannot
respond, so loss improvements after that point are not attributable to
the drive. `RDA_DISABLE_SAT=1` suppresses this for diagnostics only and
must never be set for a reported run.

### 4.4 The exact-equality rule

If a supposedly-changed condition produces a result identical to its
control in every digit, the experiment is treated as broken and
investigated before the result is believed. Two controls in this project
failed this way, both silently.

---

## 5. Analysis plan

Stated in advance, so that the analysis cannot be chosen after seeing the
numbers.

1. Run A on seeds 42, 43, 44. Report all three values, the mean, the
   spread, and the standard deviation. (Already done: mean 1.294623,
   spread 0.0137.)
2. Run B' on the same three seeds.
3. Compare: is (mean B' − mean A) larger than 0.0137?
4. If yes, the primary hypothesis is supported *at this scale, on these
   seeds*, and that qualification must appear in the conclusion.
5. If no, the hypothesis is not supported at this scale. It is **not**
   refuted — three seeds and a small model cannot refute a general claim —
   and the paper must say "not supported", not "refuted".
6. Independently, report whether the state saturated during any B' run.
   A saturated run invalidates its own loss number.

### 5.1 Pre-registered gate

The B' A/B does not begin until the 1000-step production diagnostic shows
the saturation mechanism is controlled rather than delayed. Saturation
previously began at step 217; 250 steps of evidence covers only ~15% past
onset and cannot distinguish the two.

---

## 6. Known methodological weaknesses

- **Three seeds is few.** The spread estimate itself is unstable at n=3.
- **One scale.** Stability was established at depth 8 / d_model 768. The
  final configuration may behave differently.
- **The diagnostic configuration is not the treatment configuration.**
  This is a real gap and is stated rather than glossed.
- **Evaluation is bits-per-byte**, which is insensitive to improvements in
  calibration or adaptation. If the state's effect is not a loss effect,
  this metric will not detect it — which would be a false negative, and
  one worth pre-registering how to interpret.
