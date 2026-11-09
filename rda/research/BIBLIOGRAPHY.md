# BIBLIOGRAPHY

Structured research bibliography. Each entry uses a fixed template so that
adding a paper is mechanical and gaps are visible.

---

## Rules for this file

1. **No invented citations.** A paper appears here only if its metadata
   has been verified from a reliable source. An unverified entry is worse
   than a missing one, because it will be cited later by someone who
   assumes it was checked.
2. **Record the verification method** in the Notes field. "Verified from
   the arXiv listing on 2026-09-28" is meaningful; "I think this is
   right" is not.
3. **Similarity and Difference are required, and must be specific.**
   "Related work" without a stated difference is not a positioning.
4. **"Potential experiment inspired by it"** should name a concrete
   experiment, not a direction.
5. The first entry is the only paper currently verified. It is the
   upstream project and is a hard dependency, not merely related work.

---

## karpathy/nanochat

```
Title:
  nanochat — the minimal but complete chat LM training repository

Authors:
  Andrej Karpathy

Year:
  2025 (repository; this project tracks it as the `upstream` remote)

URL:
  https://github.com/karpathy/nanochat

Venue:
  Source repository, not a publication

Core idea:
  A small, complete, dependency-light implementation of the full chat-LM
  training pipeline: tokenizer training, pretraining on FineWeb-Edu,
  supervised fine-tuning, and a model-specific evaluation suite. The
  design value to this project is that every stage is short enough to read
  end to end, which is what makes instrumenting the forward pass
  practical.

Relation to RDA:
  RDA is a fork. The unmodified nanochat path IS the experimental
  baseline ("A"). The baseline numbers in EXPERIMENT_LOG.md E1 are
  produced by this code with no RDA modifications. RDA adds an 8-dimensional
  recurrent state to the residual stream, which is the only architectural
  change.

Similarity:
  Same tokenizer, same data pipeline, same optimizer, same evaluation.

Difference:
  RDA adds a persistent recurrent state, a prediction-error pathway, and
  FiLM conditioning of the residual stream. It also changes the loss
  computation: the base forward's loss is DISCARDED and recomputed from
  the FiLM-modulated hidden states, so that the state influences the loss
  only through the modulation.

Potential experiment inspired by it:
  Already realized — A is the control for every B' comparison. The
  relevant practice inherited from upstream is the TREE cleanliness
  check, extended in commit 358993e to make run provenance explicit.

Notes:
  Verified 2026-09-28 via `git remote -v` and the `upstream` remote
  present in this repository. The LICENSE file in this repository is
  Karpathy's MIT license, inherited from upstream and unmodified.
```

---

## Open slots

The following categories are known to need coverage. They are listed as
placeholders with the search criteria, NOT as citations, because no paper
in them has been verified yet.

```
Slot: Internal / persistent state in recurrent or non-transformer models

  Search criteria: recurrent language models with an explicit compressed
  state carried across tokens; state-space and memory-augmented models;
  compression RNNs and their successors.
  Why it matters: RDA's central claim is that a small persistent summary
  adds information the context window does not already carry. This
  literature either supports that or already contains the counterexample
  that would refute it, and RDA should not be developed without having
  read it.
  Status: NOT YET SEARCHED. No citation recorded.

Slot: Layer normalization and scale control in residual streams

  Search criteria: work on residual-stream magnitude growth during
  training; normalization placement; weight-norm / output normalization
  reparameterizations.
  Why it matters: both stabilizations in this project are output-scale
  reparameterizations, and both were found by measurement rather than
  from the literature. If the placement argument is already known, the
  project should cite it.
  Status: NOT YET SEARCHED. No citation recorded.

Slot: Feature-wise linear modulation

  Search criteria: FiLM conditioning; per-feature affine conditioning of
  hidden states.
  Why it matters: FiLM is the only route from state to loss, and the
  runaway found in Decision 3 is a property of that route.
  Status: NOT YET SEARCHED. No citation recorded.

Slot: Feedback loops created by differentiable inputs derived from the loss

  Search criteria: self-referential / meta-learning architectures where a
  quantity derived from the loss is fed back as an input; meta-learning
  without a second-order derivative; inner-loop constructions.
  Why it matters: the PE pathway is exactly this shape — a quantity
  computed from the loss, fed forward as an input. The investigation
  established that the loop is closed but not currently exciting; the
  literature is the right place to find out whether that is robust or
  lucky.
  Status: NOT YET SEARCHED. No citation recorded.

Slot: Prediction error as an internal learning signal

  Search criteria: self-prediction error, eligibility traces,
  prediction-error-driven plasticity, surprise-based gating.
  Why it matters: PE is the second of the two state inputs and its
  growth during training (10.5 -> ~36) is still unexplained.
  Status: NOT YET SEARCHED. No citation recorded.

Slot: Affective / motivational state in artificial agents

  Search criteria: intrinsic motivation architectures; drives and
  motivational variables in reinforcement learning; emotion-tagged agent
  state.
  Why it matters: the 8 dimensions were named after this literature. That
  naming is a hypothesis (H2), and reading the source literature is the
  way to find out whether the names are meaningful or merely evocative.
  Note that RDA's dimensions are explicitly NOT supervised emotion
  labels, so this is motivational prior, not a claim of correspondence.
  Status: NOT YET SEARCHED. No citation recorded.
```

---

## A note on the empty slots

The temptation, given a project under time pressure, is to populate these
from memory. That is exactly the failure mode this file exists to prevent:
a plausible-looking citation that has not been checked will be copied into
a paper, and the error is invisible to everyone downstream.

The slots above are deliberately explicit about what is being searched for
and why, so that the next person can execute the searches rather than
rediscover the need for them.

---

## Related Concepts

The bibliography is currently empty of verified entries. The concepts below are
the ones this project would need sources for, in rough order of priority:

- [[RDA and Persistent State]] — the architecture under test
- [[FiLM Modulation]] — the conditioning mechanism
- [[Prediction Error]] — the drive input, and the PE literature
- [[Online Learning]] — continual/adaptive architectures
- [[Control Arm]] — experimental design references
- [[Ablation Study]] — method references
- [[BPB]] — standard language-modelling evaluation

See [[HYPOTHESES]] for which claims are hypotheses rather than results, and
[[BIBLIOGRAPHY]]'s own rules: no entry appears until its metadata is verified.
An unverified citation is worse than a missing one, because it will be cited
later by someone who trusts it.
