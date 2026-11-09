# RELATED WORK

**Status: not yet surveyed.** This file records what must be covered and
why, rather than what has been read. It is a working plan, not a
positioning statement.

The search criteria and rationale for each slot are recorded in
`../research/BIBLIOGRAPHY.md` under "Open slots". This file is where the
resulting citations and the actual positioning argument will go.

---

## What RDA must position itself against

### 1. Recurrent and state-carrying language models

The central claim of RDA is that a small persistent summary carries
information that a causal transformer's bounded context does not. This is
a claim about a well-populated literature, and RDA cannot be evaluated
until it has read that literature — in particular whether the
"compressed state across arbitrary distance" claim has been made before,
and whether it has already been refuted at this scale.

**Risk if skipped:** the paper's central contribution is not new, or is
known not to work, and neither was noticed.

**Required output:** a paragraph stating precisely what RDA assumes that
this literature does not, and vice versa.

### 2. Loss-derived inputs and feedback architectures

RDA's `pe` is a quantity computed from the loss and fed forward as an
input to the same computation. That is a specific and somewhat unusual
shape, and this project found that it is closed but dormant under
zero-initialization.

The literature on inner loops and meta-learning is where the general
question lives: when is a loss-derived input a legitimate computational
device, and when does it become a shortcut? RDA currently has an
empirical answer for one architecture and no theoretical framing.

**Risk if skipped:** the paper cannot explain why its own instability
happened, only that it did.

**Required output:** whether loss-derived inputs are conventionally
stopped, and if so, why — which would bear directly on the open decision
recorded in `../research/DECISIONS.md`.

### 3. Normalization and residual-stream scale growth

Both stabilizations in this project are output-scale
reparameterizations, and both were found by measurement rather than from
the literature. The measurement showed that normalizing a *projection's
input* does not bound its *output*, because the weight norm remains free.

**Risk if skipped:** the paper presents as novel something that is
standard practice, and — worse — misses established reasons why the
placement matters.

**Required output:** whether output normalization of this form is known,
and what the standard argument for its placement is.

### 4. Feature-wise linear modulation

FiLM is the only path from state to loss here, and the runaway found in
this project is a property of that path specifically.

**Required output:** the standard FiLM formulation, so that the
normalization applied to it can be described as a modification of
something recognizable.

### 5. Affective and motivational state in artificial agents

RDA's eight dimensions are named after this literature. That naming is
currently a hypothesis with no empirical support, and the names are doing
real work in the write-up — they determine what a reader expects to see in
Section 6.

**Required output:** whether the chosen dimensions correspond to anything
in the established taxonomy, and whether the taxonomy offers a better
starting set. Note that RDA's dimensions are explicitly *not* supervised
emotion labels, so the relationship is motivational prior rather than
correspondence.

---

## Honest assessment of the gap

Related work is the weakest section of this project, and it is weak in a
way that is currently invisible: the paper reads as though the design
space had been surveyed, when it has not.

Every claim in `DRAFT.md` Section 1 and 3 that reads as a design choice
("a small, slowly-decaying summary", "the interesting difficulty is...")
is currently a position taken without having read the alternatives. That is
the most likely way for this work to be judged as unsound, and it is not
fixable by adding citations at the end — the design may need to change.

**Recommendation: complete slots 1 and 2 before the B' A/B is run.** Both
bear on whether the experiment is worth running at the current scale.
