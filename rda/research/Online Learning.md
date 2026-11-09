# Online Learning

## What it is

**Online learning** means the model keeps its state between examples. It is not
trained on a fixed dataset in one pass; it processes a stream, and what it
learned at step `t` affects how it behaves at step `t+1`.

RDA is an online-learning architecture in a specific and narrow sense: the
[[RDA and Persistent State]] vector persists across micro-steps under
**decay-only persistence**.

## Why it matters here

RDA's central claim is about *amortized adaptation* — carrying a summary across
positions rather than recomputing it. That is an online-learning claim, and it
is the mechanism the whole architecture exists to test.

## Decay-only persistence

Under training, the state is carried forward on every step after the first:

```python
E_t_external = out["E_next"].detach()      # carried to the next micro-step
```

The state is **never hard-reset at a document boundary**. The `decay` term
(`0.95`) is the *only* mechanism governing how much prior state survives into a
new document.

That is a deliberate design choice, and it has a consequence worth stating
plainly:

> The recurrence still runs **per row**, because a packed row concatenates
> independent documents. Mixing their states within a row would be meaningless.
> The only cross-document effect is the decay term.

Document starts are counted and logged (`document_start_mask`, based on the BOS
tokens the dataloader already prefixes to every document) so a carry that crossed
a boundary is distinguishable from one that stayed within a document. Under
decay-only persistence this is **reporting only** — it triggers no reset.

## Why "decay-only" and not "reset at boundaries"

The alternative — resetting the state at each document start — would make the
state a *within-document* summary only. That is cleaner semantically, but it
removes exactly the long-range carry the architecture exists to provide.

The trade is deliberate: with `decay = 0.95`, the state's memory half-life is
about 13.5 steps, so long documents accumulate substantial carry while short
ones reset most of it.

## Two things this is NOT

**It is not a KV cache.** The KV cache is per-sequence inference state for
efficiency. RDA's state is *trained* and *persistent across documents*.

**It is not reinforcement learning.** Nothing here observes rewards or
outcomes. The state is updated by a fixed recurrence on measured signals.

## Related concepts

- [[RDA and Persistent State]] — the state being carried
- [[State Saturation]] — what bounds the carried value
- [[Prediction Error]] — the signal that updates it each step
- [[Anomaly Evaluation]] — the other sequential task in this project

## Evidence / references

- [[CURRENT_STATE]] — recurrence and persistence as implemented
- [[DECISIONS]] — Decision 5, do not modify position-0 behaviour yet
- [[RDA_ARCHITECTURE_GUIDE]] — Section 3, recurrence and the cost of decay
