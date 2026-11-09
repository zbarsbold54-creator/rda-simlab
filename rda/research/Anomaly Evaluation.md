# Anomaly Evaluation

## What it is

**Anomaly evaluation** means measuring how well a model handles inputs that do
not resemble its training distribution — corrupted text, out-of-domain content,
or inputs drawn to be unusual.

It is a separate question from language modelling: not "how well does it predict
text" but "does it behave sensibly when the text is weird".

## Why it matters here

RDA's [[RDA and Persistent State]] has two dimensions whose *intended* roles are
directly about unusual input:

- `novelty` — how much the input departs from what was expected
- `homeostatic_dev` — drift from the model's settled baseline

These are named for anomaly-adjacent behaviour. But naming them is a
hypothesis, not a result, and the project has **not** tested whether the
coordinates acquire the behaviour their names suggest.

The distinction matters because it is easy to slide from "the state has a
dimension called novelty" to "the state detects novelty". Only the second is a
claim requiring evidence.

## Status: not yet done

`EXPERIMENT_LOG.md` records no completed anomaly-evaluation experiment. The
relevant files under `eval/` are referenced in `.gitignore`
(`eval/eval_anomaly_set.jsonl`, `eval/recall_comparison.json`), indicating the
setup exists but the results are either in progress or not part of the current
record.

## What it would take

A meaningful test would need:

1. an anomaly set that is genuinely unlike the training distribution,
2. a metric that separates "handled well" from "handled badly",
3. a comparison between arms, since the interesting question is whether the
   state makes the model *better* at anomalies or merely different.

The third is the one that matters scientifically, and it is the same shape as
the [[Control Arm]] question: does the state *information* help, or only the
parameters?

## Related concepts

- [[RDA and Persistent State]] — the dimensions named for this behaviour
- [[Online Learning]] — the other sequential task in the project
- [[Control Arm]] — the experiment shape an anomaly comparison would take
- [[Ablation Study]] — the related method of isolating one component

## Evidence / references

- [[CURRENT_STATE]] — the dimension table with intended roles
- [[RESEARCH_QUESTIONS]] — secondary questions
- [[HYPOTHESES]] — H2: the 8 dimensions acquire stable, interpretable structure
