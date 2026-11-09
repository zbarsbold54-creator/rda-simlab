# BPB

## What it is

**BPB (bits per byte)** is the project's primary evaluation metric. Lower is
better.

Roughly: it measures **how many bits the model needs to describe each byte of
text**. A model that predicts text well needs fewer bits per byte, so it scores
lower.

Concretely, it is derived from the cross-entropy loss, converted from
nats/token into bits/byte using the byte length of each token. The practical
effect of the conversion is that the number lands in a human-readable range
around 1.2–1.8 for this model size rather than the raw loss around 7–9.

## Why it matters here

Every claim in this project is a BPB difference. There is no other metric. The
entire experimental record reduces to "this arm's BPB versus that arm's BPB,
at the same step count, on the same validation batch".

## The rules that make it trustworthy

### Final step, not best checkpoint

The metric is read at the **final** step (`Step 00600 | Validation bpb:`), not
from `Minimum validation bpb`, which is the *best* checkpoint.

This matters because "best" is selected using the validation set — reading it
introduces selection bias. The final-step value is what the run actually ended
on.

Note the parsing subtlety: `{:05d}` renders step 600 as `00600`, so the parser
must anchor on `Step 00600` exactly. A pattern for `Step 06000` silently
matches nothing.

### Identical evaluation conditions

Both arms must use the same `--eval-tokens` and the same split. The val shard is
the **last** parquet file, and `base_eval` uses split `"val"`.

### A measured noise floor

A baseline **repeat** at seed 42 gave 1.300643 versus 1.300576 — a difference of
**0.000067**. That number is what makes the 0.018024 effect interpretable: it is
269× the repeat noise.

Without a measured repeat, there is no way to know whether 0.018 is a signal or
ordinary run-to-run variation.

## Key measured values

### The main A/B (600 steps, three seeds, post-leak-fix)

| seed | baseline | B' | difference |
|---|---|---|---|
| 42 | 1.300643 | 1.278289 | −0.022354 |
| 43 | 1.287503 | 1.274878 | −0.012625 |
| 44 | 1.294832 | 1.275739 | −0.019093 |

Mean |difference| = **0.018024**

### Post-leak-fix ablations (24 identical validation batches)

| configuration | BPB |
|---|---|
| baseline | 1.242340 |
| B' | 1.137101 |
| [[FiLM Modulation]] zeroed | 2.990616 |
| [[Prediction Error]] zeroed | 4.371596 |
| [[Event Signal]] zeroed | 1.596395 |

These show the state pathway carries most of the model's ability to predict,
not a small addition to it.

### The 200-step preview (seed 42)

| step | B' | baseline | C |
|---|---|---|---|
| 50 | 1.924210 | 1.944570 | 1.924313 |
| 100 | 1.800259 | 1.807158 | 1.802097 |
| 150 | 1.701248 | 1.720822 | 1.704882 |
| 200 | 1.645905 | 1.674698 | (timed out) |

At step 150, the last step all three arms reached:

- B' − baseline = **−0.019574**
- C − B' = **+0.003634**
- baseline − C = **+0.015940**
- fraction of B's gain that C retains = **81.4%**

## The threshold — a proposal, not a fact

The verdict rule uses **0.0137**:

| verdict | criterion |
|---|---|
| C matches B' | \|C − B'\| < 0.0137 on all three seeds |
| states contribute | C worse than B' by > 0.0137 on all three seeds |
| otherwise | inconclusive |

> **This number has not been calibrated.** It is roughly 4× the measured repeat
> noise and comparable to the per-seed spread of the paired B'-baseline diffs
> (0.0097). It was not derived from a power calculation. With three seeds, a
> result near the threshold should not be trusted in either direction.

## A limitation worth stating

The per-seed spread of the B'-baseline differences is 0.0097 — comparable to the
effect itself (0.018). That means three seeds is not enough to make the effect
precise, even though the effect is clearly larger than the noise floor.

## Related concepts

- [[Control Arm]] — the experiment judged on this metric
- [[Ablation Study]] — the component-removal variant of the same idea
- [[Label Leakage]] — invalidated a whole generation of BPB measurements
- [[Anomaly Evaluation]] — the other evaluation used in this project

## Evidence / references

- [[E15_ABLATION]] — per-dimension BPB deltas
- [[AB_SETUP_READONLY]] — the metric definition and step-count verification
- [[EXPERIMENT_LOG]] — E17: the A/B result
- [[RDA_ARCHITECTURE_GUIDE]] — Section 14, measurement history
