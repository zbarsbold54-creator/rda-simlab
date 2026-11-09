# Residual Stream

## What it is

A **residual stream** is the running vector that passes through every layer of a
transformer. Each layer does one thing to it:

```
h  ←  h + F(h)
```

The stream is not *replaced* by `F(h)` — it is *augmented* by it. That single
design decision is why deep networks train at all.

## Why it matters here

RDA treats the residual stream as a **shared bus**:

- the base model **writes** to it,
- [[Event Signal]] **reads** from it,
- [[FiLM Modulation]] **writes back** to it.

That makes the stream where the project's one real architectural risk lives —
and where most of its debugging happened.

## The cost: no bounded scale

Because `h` only ever accumulates, `||h||` is set by the base model's habit
rather than by any constraint.

If some component reads `h`, it reads a vector whose magnitude drifts for
reasons unrelated to that component's purpose. This is not hypothetical — it
broke the event pathway:

```
event_proj output:  mean |.| = 2.11,  max 4.11  over 30 steps
                   past atanh(0.99) = 2.647 on 26 of 30 steps
                   → state railed within ~130 steps
```

Nothing was "broken". The projection faithfully read a stream that had grown,
and fed an unbounded value into a squashing nonlinearity.

> Whenever a module reads a residual stream, ask what bounds the magnitude of
> what it is reading. If the answer is "nothing", measure `||h||` over training
> **first** — not the gradient.

## Why the residual framing earns its keep

Two properties follow automatically:

**Signal preservation.** The identity path guarantees `h` stays reachable.
Layers can only *add* to what came before, so information at layer 0 is present
at layer 12, merely augmented.

**Gradient preservation.** Differentiating gives a Jacobian containing an
identity term that does not decay through depth. Gradient signal propagating
backwards retains an unmodified component. This is the difference between a
network that trains and one that does not.

## Why RDA needs the unmodulated stream

The event pathway reads the **unmodulated** residual stream. That is structural,
not incidental:

```
drive = drive_seq(pre_outcome_hidden, pe_seq)   # computed FIRST
hidden_mod = modulate(hidden, film_state)        # modulation SECOND
```

Because the drive is computed before modulation, the state can never see its
own downstream effect through the event term.

This ordering is what makes the recurrence's inputs **loop-free** — and
therefore what makes the closed-form vectorisation possible. Had the drive been
a function of the modulated stream, the closed form would be impossible.

## Sliding-window attention

This project uses `window_pattern="L"`, so every layer attends to a sliding
window. nanochat's `has_ve(layer_idx, n_layer)` decides whether a layer has a
value embedding, and the pattern string assigns sliding-window or full attention
per layer.

Consequence: local information reaches far less far than with full attention, so
anything requiring long-range integration must flow through the residual
stream. RDA's state is precisely such a mechanism — an 8-dim vector carrying
information across arbitrarily many positions.

## Related concepts

- [[FiLM Modulation]] — what writes to the stream
- [[Event Signal]] — what reads from it
- [[RDA and Persistent State]] — the state being transported
- [[State Saturation]] — the failure this unboundedness caused
- [[Label Leakage]] — how state reached the logits

## Evidence / references

- [[CURRENT_STATE]] — verified pathway ordering
- [[DECISIONS]] — Decision 2, normalise the projection output
- [[FAILED_EXPERIMENTS]] — F4, diagnostic shadowing
- [[RDA_ARCHITECTURE_GUIDE]] — Section 1
