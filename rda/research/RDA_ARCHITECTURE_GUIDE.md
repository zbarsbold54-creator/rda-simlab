# RDA: Architecture, Theory, and Measurement

### A learning document that doubles as a map of this project

---

## How to use this document

This has two jobs, and it is worth being explicit about which is which.

**Job 1 — teaching.** Sections 1 through 9 build the ideas from first principles:
what a residual stream is, why attention is a mixing operation, what FiLM is,
why recurrences go unstable, why a closed form beats a loop. These are the
concepts you need in order to *predict* what your code will do before you run
it. They are written to be read in order.

**Job 2 — orientation.** Sections 10 through 16 are about *this specific
codebase*: what RDA is, how it is wired, what has been measured, what has been
retracted, and what is still unknown. If you want to understand what the
project is actually doing rather than how the general machinery works, start
here and read forward.

Throughout, the running example is the real experiment: the RDA affective-state
model on nanochat, with the 8-dim state, the leaky recurrence, the FiLM
modulation, and the three-arm A/B that decides whether the state is doing any
work at all. Real measured numbers appear inline, including the ones that
contradict earlier claims — those are kept deliberately rather than tidied
away, because in this project the corrections *are* the result.

---

## Table of contents

**Part I — Foundations**

1. [The transformer as a residual stream](#1-the-transformer-as-a-residual-stream)
2. [Attention, and what "mixing" means](#2-attention-and-what-mixing-means)
3. [The training loop's hidden state: recurrence](#3-the-training-loops-hidden-state-recurrence)
4. [Feature-wise modulation (FiLM)](#4-feature-wise-modulation-film)
5. [Boundedness: why tanh saturates and why that matters](#5-boundedness-why-tanh-saturates-and-why-that-matters)
6. [Reparameterization as a stability tool](#6-reparameterization-as-a-stability-tool)
7. [Closed-form recurrences](#7-closed-form-recurrences)
8. [Causality and the label-leak surface](#8-causality-and-the-label-leak-surface)
9. [Experimental design: what a control arm is for](#9-experimental-design-what-a-control-arm-is-for)

**Part II — This project**

10. [What RDA is](#10-what-rda-is)
11. [The state module, line by line](#11-the-state-module-line-by-line)
12. [The forward pass, end to end](#12-the-forward-pass-end-to-end)
13. [Arm C: the control that tests the idea](#13-arm-c-the-control-that-tests-the-idea)
14. [Measurement history: what we learned](#14-measurement-history-what-we-learned)
15. [Retractions and corrections](#15-retractions-and-corrections)
16. [Open questions](#16-open-questions)
17. [Glossary](#17-glossary)

---

# Part I — Foundations

## 1. The transformer as a residual stream

A transformer layer is, in essence, one line:

$$
h \leftarrow h + F(h)
$$

where $F$ is some function — attention, a feed-forward network, or in our case a
modulation. The key word is **residual**. The stream $h$ is not replaced by
$F(h)$; it is *augmented* by it. Everything about how these networks behave
follows from that one design decision.

### Why the residual framing earns its keep

Consider what happens if you compose many such operations without the skip. Each
layer transforms its input, and the composition of $N$ arbitrary maps is
arbitrary — gradients either vanish toward the input or explode toward the
output. There is no reason for information to survive 12 layers.

With the skip, two things become automatic:

**Signal preservation.** The identity path guarantees that $h$ remains
reachable. Every layer can only *add* to what came before, so information
present at layer 0 is present at layer 12, merely augmented. This is why
stacking is nearly free of the degradation that plagues deep plain networks.

**Gradient preservation.** Differentiating $h_{n+1} = h_n + F(h_n)$:

$$
\frac{\partial h_{n+1}}{\partial h_n} = I + \frac{\partial F}{\partial h_n}
$$

The Jacobian contains an identity term that does not decay through depth. A
gradient signal propagating backwards through a 12-layer stack retains a
component that is at least unmodified. The other terms may vanish or explode;
the identity does not. In practice this is the difference between a network
that trains and one that does not.

### The cost: the stream has no bounded scale

Here is the trap that shaped most of this project's debugging. Because $h$ only
ever accumulates, $\lVert h \rVert$ is set by the base model's habit rather than
by any constraint. If some later component decides it wants to read $h$, it is
reading a vector whose magnitude drifts according to factors that have nothing
to do with the component's purpose.

In this project that showed up concretely. `event_proj` reads the residual
stream to build the "event" drive. Its output reached mean $|\cdot| = 2.11$ and
max 4.11 over 30 real steps — past $\text{atanh}(0.99) = 2.647$ on 26 of 30
steps. The state was then pinned at its rail within about 130 steps. Nothing was
"broken"; the projection was faithfully reading a stream that had grown, and
feeding an unbounded value into a squashing nonlinearity.

**Lesson to carry forward:** whenever a module reads a residual stream, ask
what bounds the magnitude of what it is reading. If the answer is "nothing",
the first thing to measure is $\lVert h \rVert$ over training, not the
gradient.

### What this means for RDA

RDA treats the residual stream as a *shared bus*. The base model writes to it;
RDA reads from it and writes back through FiLM. Section 12 returns to this
interaction, because it is where the project's one real architectural risk
lives.

---

## 2. Attention, and what "mixing" means

Attention is the only component in the network that moves information *between*
positions. Everything else — projections, feed-forward, LayerNorm — acts on each
position independently. This distinction is what makes the leakage question in
Section 8 tractable.

### The mechanism

For query $q$ at position $p$, key $k$ at position $j$, value $v$ at position
$j$:

$$
\text{score}_{pj} = \frac{q_p \cdot k_j}{\sqrt{d_k}}, \qquad
a_{pj} = \text{softmax}_j(\text{score}_{pj}), \qquad
o_p = \sum_j a_{pj} v_j
$$

Causality is enforced by masking $a_{pj} = 0$ for $j > p$. Position $p$ can
attend to itself and anything before it.

### One crucial detail for a recurrent architecture

The value $v_j$ at position $j$ is produced by the base model *at* position $j$.
In a plain transformer that is fine, because position $j$'s representation
depends only on tokens $\le j$.

But RDA needs the hidden states *before* the state is updated, to compute the
drive that determines the state. So it slices:

```python
pre_outcome_hidden = hidden[:, :-1, :]   # (B, T-1, d_model)
pooled_post_outcome = hidden[:, 1:, :]   # (B, T-1, d_model)
```

Two views of the *same* positions, shifted by one. The whole causality argument
then rests on which view feeds which consumer — and that is exactly where the
project's one retracted claim came from (Sections 8 and 15).

### Sliding-window attention

This project uses `window_pattern="L"`, so every layer attends to a sliding
window. nanochat's `has_ve(layer_idx, n_layer)` decides whether a layer has a
value embedding, and the pattern string assigns sliding-window or full
attention per layer. The "L" pattern means sliding-window throughout.

Practical consequence: local information reaches far less far than it would with
full attention, so anything requiring long-range integration must flow through
the residual stream. RDA's state is precisely such a mechanism — an 8-dim
vector carrying information across arbitrarily many positions.

---

## 3. The training loop's hidden state: recurrence

A transformer is stateless between tokens. RDA adds memory by making the
affective state a carried vector.

### The trivial recurrence and why it fails

Write it as you would first:

$$
E_t = \tanh(W_e h_t + W_p \epsilon_t)
$$

Run it for 600 steps and something breaks. Two failure modes, both observed in
this project:

**Saturation.** Because $\tanh' = 1 - \tanh^2$, once $|E|$ is large the gradient
vanishes. Measured over 500 real steps: 5 of the 8 dimensions spent **more than
94% of steps with $|E| > 0.9$**. An untrained module saturated about 50% of the
time — proving the problem was structural, not an artifact of a strong learned
signal.

**Ratcheting.** A self-exciting loop. The state modulates the residual stream
(`h`), which changes the state's own future inputs, which modulates harder.
Measured: the FiLM gain $\lvert 1 + \text{film\_scale}(E) \rvert$ grew
$1.00 \to 2.23$ with the base model trainable, and $1.00 \to 9.57$ with the
base model **frozen**. That last number is the decisive one — with a frozen
base there is no varying external input at all, yet the loop still ran away. A
loop driven by no varying signal can only be self-exciting.

### The fix: convex combination

The project's answer, in `rda.py`:

$$
E_{t+1} = \lambda E_t + (1 - \lambda)\,\tanh\big(\text{event} + \text{pe}\big)
$$

with $\lambda = 0.95$.

Read this carefully, because the *shape* is the entire point. The retained
state and the fresh drive are **convex-combined**, with weights summing to one.
Both inputs lie in $[-1, 1]$; therefore so does the output. **Provably**, not
empirically:

$$
\lambda E_t + (1-\lambda) d \in [-1, 1] \quad \text{for } E_t, d \in [-1, 1]
$$

There is no final $\tanh$, because none is needed. Compare the old rule:

$$
E_{t+1} = \tanh(\lambda E_t + \tanh(\text{ev}) + \tanh(\text{pe}))
$$

whose pre-squash argument reaches $\lambda + 1 + 1 = 2.95$, which $\tanh$ maps
to $0.9945$. Anything above $|1.472|$ already yields $|E| > 0.9$. The old rule
was pinned near its asymptote across most of its dynamic range *by
construction*.

The code asserts the bound rather than observing it:

```python
assert E_next.abs().max() <= 1.0 + 1e-6, (
    f"gated update escaped [-1, 1]: max|E_next|={E_next.abs().max()}"
)
```

**This is the central architectural idea in the project:** prefer a
reparameterization that makes the failure mode unrepresentable over a penalty
that discourages it. A clamp can be traded against the loss; a convex
combination cannot be violated at all.

### Why 0.95, and what it costs

With $\lambda = 0.95$ the effective memory half-life is
$-\ln 2 / \ln 0.95 \approx 13.5$ steps. The state reaches about
$1 - 0.95^{13.5} \approx 0.50$ of the way toward a new drive in that span.

The cost is real and worth naming: the new event's full influence is only
$(1-\lambda) = 0.05$ per step. To move a drive meaningfully takes tens of steps.
That is the trade the design makes deliberately — slow-moving state, but state
that cannot rail. Section 14 shows this is where `threat` eventually gets into
trouble anyway, by a different route.

---

## 4. Feature-wise modulation (FiLM)

FiLM is how the state reaches the loss. In `rda.py`:

```python
scale = 1.0 + _unit_rms(self.film_scale(E_t), self.film_rms)
shift = _unit_rms(self.film_shift(E_t), self.film_rms)
return x * scale + shift
```

With $E_t \in \mathbb{R}^{8}$, `film_scale` maps it to $\mathbb{R}^{d_{model}}$,
so this is a per-position affine transform of the residual stream:

$$
\text{scale} = 1 + \gamma \cdot \frac{W_s E_t}{\lVert W_s E_t \rVert_{\mathrm{RMS}}}, \qquad
\text{shift} = \gamma \cdot \frac{W_b E_t}{\lVert W_b E_t \rVert_{\mathrm{RMS}}}
$$

with $\gamma = \text{film\_rms} = 0.5$ (fixed, non-learnable).

### Zero-init, and why it matters

```python
for layer in (self.film_scale, self.film_shift):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
```

**All** of weight and bias. If only the weights were zeroed, $\text{scale} = 1 + b
\ne 1$ and `modulate()` would not be an exact identity at init — the model would
start already perturbed.

Zero-init buys a specific guarantee, and this project depends on it heavily:

> **The gated update is a fixed point.** Since
> $\frac{\partial \mathcal{L}}{\partial E} \propto W_s = 0$ at init, the drive
> receives **exactly zero** gradient at step 0, for all 8 dimensions.

Measured with a standalone `retain_grad` test: gradient is not `None`,
absmax exactly `0.0`, nonzero-frac `0.0000`.

That is not a bug — it is correct, and arm C is built on top of it (Section 13).
But it has a consequence that caused a real measurement error: a probe that
never steps the optimizer will report the drive gradient as zero *forever*, and
one may mistake the zero-init for a saddle. See Section 15.

### Per-position, not per-sequence

`modulate()` takes $E_t$ of shape $(B, T, 8)$ — one state per position. An
earlier version accepted $(B, 8)$ and broadcast, which meant position 0 was
modulated by a state computed from the *last* token: a **1022-token lookahead**
at $T = 1024$, and architecturally impossible to reproduce at inference, where
an autoregressive step has no "final state" to draw on.

---

## 5. Boundedness: why tanh saturates and why that matters

### The derivative is the story

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

At $|x| = 1$: $\tanh' = 0.42$. At $|x| = 2$: $\tanh' = 0.07$. At $|x| = 3$:
$\tanh' = 0.0099$.

The gradient signal dies as the argument grows. Since the state is read back
through FiLM and the FiLM gain is itself driven by the state, saturation
compounds: a saturated state cannot be corrected, and it modulates the stream
with whatever it holds.

### The concrete measurement

The pre-tanh argument decomposes as

$$
\text{arg} = \text{event\_drive\_scale}\cdot \mathrm{ev} + W_{pe}\,\frac{\epsilon}{10} + b_{pe}
$$

Identity verified to worst residual $4.6\times10^{-7}$ over 3000 steps.

At step 2999 on the threat dimension:

| component | value | share of $\text{arg}_p$
|---|---|---|
| $b_{pe}$ (bias) | −1.71124 | **80%** |
| $W_{pe}$ (PE pathway) | −0.355 | 16.6% |
| $\mathrm{ev}$ (event pathway) | −0.074 | 3.5% |

`arg_p50` crossed $-2.0$ at step 1131 and **never** crossed $-3.0$; it plateaued
near $-2.11$. Mean $\tanh$ derivative fell $0.943 \to 0.059$ with minimum
$0.0507$ — marginal, not dead. The late slope was *positive*
($+2.1\times10^{-7}$/step), i.e. it was relaxing back.

So the honest characterisation is neither "runaway" nor "dead channel". It is a
**converged bias offset** — an equilibrium reached around steps 1300–1800.

### The mechanism: pathways swapped

The correlation structure tells the real story:

- $\mathrm{corr}(\text{drive}, \text{event term})$: $+0.9996 \to +0.0353$ (collapsed)
- $\mathrm{corr}(\text{drive}, \text{PE term})$: $+0.0099 \to +0.9628$ (rose)

And the sign flip that enabled it: `pe_proj.weight[threat]` **crossed zero at
step 569/570** ($+0.00078 \to -0.00186$), ending at $-0.575$. Before that step
the PE pathway *opposed* the event pathway; after it, it *reinforced*.

Of the eight dimensions, `threat` was the **only** one to saturate. Only two
weights ever crossed zero: `threat` (step 546) and `integrity` (step 1881,
ending $+0.0025$). No dimension had a one-signed bias gradient (dominant
fractions 0.505–0.537, all near 50/50) — so `threat`'s specialness is not
gradient sign but bias **magnitude** (−0.9486, 6–20× the others) and its share
of the argument.

### Bias sign is not a ratchet

A tempting story is "the bias accumulated in one direction until it hit the
rail". The gradient data refuses it: 1623 positive-signed bias gradients versus
1376 negative. The bias **overshot** to −1.756 at step 1750 and is now relaxing
back upward. It converged, it did not run away.

---

## 6. Reparameterization as a stability tool

Three places in `rda.py` apply the same idea. Worth studying together, because
they form a coherent design philosophy.

### (a) Bounded state — convex combination
Covered in Section 3. Makes $\lvert E \rvert \le 1$ unrepresentable.

### (b) Bounded event projection — output RMS normalization

```python
ev = self.event_proj(self.h_norm(hidden_state))
ev = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
event_term = self.event_drive_scale * ev   # fixed 0.1
```

The chain of reasoning here is the best diagnostic story in the codebase, and it
is worth following step by step.

`h_norm` (LayerNorm) was added first, intending to stop base-model activation
growth from inflating the drive. Measured over 30 real steps: LayerNorm output
rms stayed at exactly 1.000 and `h_norm.weight` at exactly 1.0 for the whole
run. **It was never the failure.**

The actual cause: $\lVert h_{norm} \rVert = \sqrt{d} = 27.71$ is a *ceiling*, not
a value. So $\text{event\_proj}(h)$ can still reach $\lVert w_{row}\rVert \cdot
27.71$, and $\lVert w_{row}\rVert$ is **learned and unbounded**:

| quantity | 30 steps, seed 42 | ratio |
|---|---|---|
| `event_proj` $\lVert W \rVert$ | 1.63 → 3.51 | ×2.15 |
| event $\lvert\max\rvert$ | 0.25 → 1.26 | ×5.03 |
| LayerNorm out rms | 1.00 → 1.00 | ×1.00 |

The drive grew because the **projection** grew, not because the input did.

The decisive control: freeze the entire base model, so `h` is byte-identical
every step. $\lVert W \rVert$ still went 1.63 → 3.51, and the event term still
reached 1.26. That isolates the cause to the gradient on `event_proj` itself and
rules out both hidden-state drift *and* any FiLM feedback loop as the driver.

**The lesson:** normalizing the *input* gives you invariance to the input's
magnitude. It does nothing about a learned *weight* that grows. Apply the
normalization where the magnitude actually lives.

This is scale-invariance, not a clamp: it does not saturate, does not flatten
gradients, and leaves the argument smooth. It also preserves the learned
relative scale *between* drives exactly — the sign and pattern of the eight
components is the signal.

### (c) Bounded FiLM gain — RMS budget

```python
def _unit_rms(v, rms):
    return v / v.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * rms
```

The FiLM gain ran away for the same reason as (b) — self-excitation. Measured
over 250 real steps: $\lvert1 + \text{film\_scale}(E)\rvert$ grew $1.00 \to 2.23$
(trainable base) and $1.00 \to 9.57$ (frozen base). Two ablations pinned it on
the gain term and not on the PE projection:

- **FiLM disabled entirely:** the state never crosses $|E| = 0.98$, even though
  PE still grows to 19.6. So PE growth is a *consequence* of FiLM, not its cause.
- **Freezing `pe_proj`:** saturation still occurs, at the same step.
- **Detaching PE's gradient:** max $|E|$ and the crossing step are unchanged — so
  this is not a differentiable-loop pathology either.

With a frozen base there is no external signal at all, yet the loop ran away.
That is the definition of self-excitation.

The fix removes the one degree of freedom that lets the loop amplify itself —
the overall *magnitude* of the gain — while keeping the part that carries
information: its **direction** across the $d_{model}$ axis.

At init the film weights are exactly zero, so this is $0/\epsilon = 0$ and
`modulate()` is the exact identity.

### The pattern

| Mechanism | Unbounded degree of freedom removed |
|---|---|
| Convex-combination update | sum of saturating terms |
| Output RMS on `event_proj` | learned $\lVert W_{row}\rVert$ |
| RMS budget on FiLM | overall gain magnitude |

All three keep a direction and discard a magnitude. All three are
reparameterizations, not penalties — nothing can be traded against the loss.

---

## 7. Closed-form recurrences

### The unrolling

$$
E_{t+1} = \lambda E_t + (1-\lambda) d_{t+1}
$$

Unrolling over a sequence of length $L$:

$$
E[t] = \lambda^{t+1} E_{\text{start}} + (1-\lambda)\sum_{s \le t}\lambda^{t-s}\, d[s]
$$

The sum is a decay-weighted prefix sum, which factorizes. Let
$w[t] = \lambda^t$ and $g = (1-\lambda)d$:

$$
E[t] = w[t]\Big(\lambda E_{\text{start}} + \textstyle\sum_{s\le t}\frac{g[s]}{w[s]}\Big)
$$

so the whole recurrence becomes a `cumsum` and a multiply. ~20,000 sequential
kernels collapse to a handful.

### The bug that caught the missing factor

An earlier version used $E_{\text{start}} + \text{cumsum}(\cdot)$ — which
corresponds to seeding the recurrence with $E_{\text{start}}/\lambda$. That is
the same recurrence run from the **wrong initial state**. It was caught by
`diagnostics/verify_vectorized_update.py` disagreeing with the sequential
reference loop by up to $2.9\times10^{-2}$ in fp32.

The `λ` in front of `E_start` falls out of the unrolling and is easy to drop.
The reference loop is preserved verbatim in `_update_loop_reference()` purely so
this class of error stays catchable.

### Why float64 accumulation

The cumsum divides by $w[t] = \lambda^t$, so its reciprocal $\lambda^{-t}$ must
stay finite. The limit is computed, not hardcoded:

$$
T_{\max} = \frac{\ln(\text{finfo}_{\max})}{\ln(1/\lambda)}
$$

| dtype | $T_{\max}$ at $\lambda = 0.95$ |
|---|---|
| float64 | 13,837 |
| float32 | 1,729 |

Training uses $T = 1024$ — 13.5× inside the float64 limit — but it is **not**
unbounded. Hence `_max_safe_T()`, which computes the limit from the dtype and
the actual decay, and a loud failure rather than silent chunking:

```python
if L > limit:
    if not chunk:
        raise ValueError(
            f"update_seq: L={L} exceeds the {limit} that "
            f"decay**-T stays finite in {acc_dtype} (decay={d}). ..."
        )
    return self._update_seq_chunked(E_start, drive, limit)
```

fp64 costs nothing measurable here: these tensors are $(B, T, 8)$, tiny beside
the base model, and no matmul is involved — which matters, since fp64 throughput
is poor on consumer GPUs. The result is cast back.

### One caveat, stated honestly

The vectorized path omits `update()`'s defensive `clamp(E_prev, -1, 1)`. Since
$E_{t+1}$ convex-combines two values already in $[-1,1]$, it lies in $[-1,1]$ in
exact arithmetic, so the clamp can only fire within one ulp ($\sim 6\times10^{-8}$)
of the bound, altering the value by at most that much — then damped by $\lambda$
at every subsequent step. Applying it would defeat the closed form. The
discrepancy is bounded by $\sim 10^{-7}$, far below the loop's own accumulated
rounding error.

---

## 8. Causality and the label-leak surface

This section covers the project's most consequential retraction. It is here
because it is the clearest worked example of the failure mode this whole
document is about: **checking the index, then mislabelling what lives at that
index.**

### The setup

The dataloader packs multiple documents per row and slides two views:

```python
x = ...
y = x[1:]      # within a row, y[p] == x[p+1] EXACTLY
```

`gpt.py:532` scores with **no shift**:

```python
cross_entropy(logits[p], targets[p])
```

Therefore `y[p]` is the label attached to position `p`'s logits. Verified:
fraction exactly `1.000000`.

### The requirement

For position `p$` 's logits not to depend on its own label, the FiLM input at
position $p$ must satisfy

$$
\text{film\_state}[p] \text{ may depend on } y[j] \text{ only for } j < p
$$

Equivalently, since $y[j] = x[j+1]$: it may depend on $x[0..p]$ and nothing
beyond — exactly what a causal position is entitled to.

### The bug

The prediction error was built from `targets[:, 1:]`:

```python
pe_seq = self.state.prediction_error_from_logits(logits, targets[:, 1:])
```

So $\epsilon[j]$ scored $y[j+1] = x[j+2]$. Then:

$$
E_{\text{next\_seq}}[j] \text{ depends on } x[j+2], \quad
\text{film\_state}[p] = E_{\text{next\_seq}}[p-1] \text{ depends on } x[p+1]
$$

and `logits[p]` is graded against `targets[p] = x[p+1]`. **The logits at
position $p$ were built from the label they are scored on.**

The fix is one character:

```python
pe_seq = self.state.prediction_error_from_logits(
    _safe_logits(self.base, hidden, pre_outcome_hidden), targets[:, :-1]
)
```

Now $\epsilon[j]$ scores $y[j] = x[j+1]$, so
`film_state[p] = E_next_seq[p-1]` depends on $y[0..p-1] = x[1..p] \subseteq x[0..p]$.
Satisfied.

### Why the original reasoning failed

The comment that justified `targets[:, 1:]` read, in essence:

> "`E_next_seq[p-1]` depends on `pe_seq[:, p-1]`, which depends on token `p` —
> strictly causal."

Both halves were correct about the **mechanism** and wrong about the
**arithmetic**. `pe_seq[:, p-1]` did depend on `targets[p]` — but
`targets[p]` **is** `x[p+1]`, not `x[p]`. The reasoning verified the index and
then mislabelled what lives at that index.

The one-token lookahead the comment worried about (using `E_next_seq[p]` instead
of `E_next_seq[p-1]`) is real and is still avoided. The bug was in the PE index
underneath it, not in that shift.

### Dynamic proof

Perturbing `y[:,k]` alone:

- hidden and logits at $p < k$: change **exactly 0**
- hidden and logits at $p \ge k$: all move (cumulative recurrence)

$\max\lvert dE \rvert_{p<k} \approx 4.5\times10^{-2}$

Note this also invalidates the natural shortcut test. Asserting
`max|dE| < k > 0` as a leak verdict is **wrong**: $E_{\text{next\_seq}}[p]$
legitimately *records* token $p+1$. The bound belongs on hidden/logits, not on
$E$.

### Retracted counterfactuals

Two "label-blindness" tests were also withdrawn:

- **"Shuffle `y` and re-score."** Invalid: the *baseline* also moves (+1.748),
  because scoring wrong answers is simply harder. It confounds the cheat with
  difficulty.
- **"PE off via `targets=None`"** (4.29 BPB). Invalid counterfactual: it trains
  FiLM in an unseen state distribution.

### The corrected measurements

After the fix, 24 identical validation batches:

| configuration | BPB |
|---|---|
| baseline (seed 42) | 1.242340 |
| **B' (post-fix)** | 1.137101 |
| FiLM zeroed | 2.990616 |
| PE zeroed | 4.371596 |
| event zeroed | 1.596395 |

`B'-s42 val BPB 1.226305 vs baseline 1.301217` is **not valid** and was
retracted. It predates the fix.

### The durable gate

`diagnostics/test_no_label_leak.py` fails on the buggy version and passes on the
fixed one, and the driver's PRE-FLIGHT runs it before any sweep. A leak found
after 12 runs costs far more than one second of preflight.

---

## 9. Experimental design: what a control arm is for

### The question

B' beats baseline by 0.018 BPB (three seeds, Section 14). Three explanations are
consistent with that number:

1. **The state information helps.** The 8 values carry something useful.
2. **The extra parameters help.** +27,760 parameters is a capacity increase, and
   more capacity can improve a model regardless of what it computes.
3. **The extra compute helps.** B' costs ~2.1 s/step versus baseline's ~1.75,
   and a deeper-effective network is a better network.

B' alone cannot distinguish these. That is the whole reason a control exists.

### The design

**Arm C** keeps every RDA parameter and every RDA computation, but replaces the
FiLM input with a learned constant:

```python
if self.const_state_vec is not None:
    film_state = self.const_state_vec.to(film_state.dtype).expand_as(film_state)
```

`expand_as` returns a **view** — zero allocation. `.to(dtype)` is a no-op when
dtypes match. So arm C has:

- identical parameter count (+27,768 vs B'’s +27,760 — the extra 8 are
  `const_state_vec` itself),
- identical FLOPs: the drive, the recurrence, the predictor and `dpe` all still
  run,
- **no state information** reaching FiLM.

If C matches B', the gain was capacity or compute, and the affective state is
decorative. If C is worse than B', the information matters.

### Three preconditions, each verified

A control is only as good as its preconditions. Each was checked by measurement
before the A/B was launched, not assumed.

**(1) The saddle.** With zero-init FiLM weights, $\partial\mathcal{L}/\partial
c \propto W_s = 0$ at step 0. If it stayed zero, arm C would be a
reparameterisation of the identity and the comparison would be vacuous.

| backward | $\lVert\partial\mathcal{L}/\partial c\rVert$ | $\lVert\partial\mathcal{L}/\partial W_s\rVert$ |
|---|---|---|
| 1 | 0.0000000000e+00 *(zero by design)* | 2.9452124238e-01 |
| 2 | 2.3130022455e-03 | 1.2724294662e+00 |
| 3 | 1.6563907266e-02 | 2.4881629944e+00 |
| 4 | 1.1805012822e-02 | 1.3823006153e+00 |

Nonzero from backward 2 onward. The zero at backward 1 is the zero-init; it
clears once the FiLM weights move (absmax 0 → 1.0e-3 → 3.9e-3).

**(2) Forward-call parity.** If C skipped work B' does, it would not be a
control. Measured per step, identical in both arms:

| function | calls/step (C) | calls/step (B') |
|---|---|---|
| `update_seq` | 1 | 1 |
| `predict_seq` | 1 | 1 |
| `event_proj` | 1 | 1 |
| `drive_seq` | 1 | 1 |
| `modulate` | 1 | 1 |

**(3) RNG isolation.** C's extra parameter must not shift the data order or any
shared init. `const_state_vec` is drawn from a **private CPU generator** seeded
`seed + 1000003`, so it is independent of the global stream. Verified: 78 shared
parameters bit-identical, 0 mismatches, and global draws after B' vs C
construction identical (`1.877206, -0.264268, -0.713366`).

### Why the state *moves* matters, and was checked

A control whose control parameter never updates is not a control. End-of-run
logging reports `const_state_vec` init, final and delta by dimension. At 3 steps
all eight deltas are $\pm 1.16\times10^{-2}$ — equal magnitude, sign following
the init sign, which is Adam's near-$\text{sign}(g)$ first-step behaviour. At 60
steps max$|delta| = 1.59$.

### A subtlety worth stating plainly

In arm C the three drive projections receive **no gradient at all** (autograd
pruned; `p.grad is None`, not a zero tensor). They are added to the grad-free
list only so `MuonAdamW._compute_adamw` does not crash calling `grad.float()`
on `None` (`optim.py:73`).

They are **not frozen**. `grad_free` is read *only* by the zero-grad priming at
`base_train.py:935-940`; `optim.py` never sees the key, and
`adamw_step_fused` applies decoupled decay unconditionally (`optim.py:75`,
`p32.mul_(1 - lr*wd)`) before examining the gradient.

Measured over 3 steps at lr 3e-3, wd 0.01, all six tensors shrank by exactly
$(1 - 3\times10^{-5})^3$; observed ratio 0.999910178 vs predicted 0.999910003,
matching to $1.8\times10^{-7}$. `h_norm.bias` did not move only because it is
$0.0$ to begin with.

The correct description is **"untrained and decayed"**, not "at init". Getting
this wrong would have made the drift check unfalsifiable.

---

# Part II — This project

## 10. What RDA is

**RDA — Regulatory Drive Architecture** — wraps a nanochat GPT with a small
recurrent affective state and modulates the residual stream through FiLM.

### The claim being tested

> A language model with a persistent, low-dimensional internal state that
> carries information across positions — and that modulates computation via that
> state — learns better than one without.

### The motivation, stated honestly

The plausible mechanism is **contextual adaptation**. The 8-dimensional state
summarises "what kind of situation is this" — novelty, uncertainty, threat,
integrity, agency, learning progress, arousal, homeostatic deviation — and
modulates the residual stream accordingly. Rather than recomputing an
appropriate transformation at every position from scratch, the model amortizes
it through a recurrent state.

The honest caveat, stated in Section 14's limitations: the drive inputs are
almost entirely functions of the prediction error, which is itself a function of
the labels. So the "situational" content is largely "how surprised was the
model recently". That is a *legitimate* signal for a language model, but it is
narrower than the affective framing suggests, and it is a direct path from the
label to the state. That path is exactly what the label leak exploited, and
fixing it (Section 8) is what made the current A/B possible.

### Scale

| | |
|---|---|
| Base model | nanochat GPT, depth 8, $d_{model} = 768$, 6 heads, head_dim 128 |
| Attention | sliding window (`window_pattern="L"`) |
| Sequence length | 1,024 |
| Base parameters | 207,618,378 |
| B' parameters | 207,646,138 (+27,760) |
| C parameters | 207,646,146 (+27,768) |
| RDA added | +0.0134% |
| Drives | 8 |
| Decay $\lambda$ | 0.95 |
| Batch / tokens | 32,768 / step; 19.7M over 600 steps |
| Hardware | one RTX 5060 Ti (sm_120, Blackwell), 8 GB |

**The addition is 0.0134% of the model.** Any large effect would be
suspicious; a small one is what a capacity explanation predicts.

---

## 11. The state module, line by line

### 11.1 Parameters

| name | shape | init | role |
|---|---|---|---|
| `h_norm` | LayerNorm(768) | 1, 0 | direction of the residual stream, not magnitude |
| `event_proj` | Linear(768 → 8) | default | hidden state → 8 event drives |
| `pe_proj` | Linear(1 → 8) | default | prediction error → 8 drives |
| `film_scale` | Linear(8 → 768) | **zeros** | state → per-unit gain |
| `film_shift` | Linear(8 → 768) | **zeros** | state → per-unit bias |
| `e_init` | (8,) | zeros | initial state (grad-free by design) |
| `event_drive_scale` | scalar | 0.1 | fixed, **not** a Parameter |
| `film_rms` | scalar | 0.5 | fixed, **not** a Parameter |

The two fixed scalars are deliberate: not registered as parameters, so they
never appear in `parameters()` and never affect checkpoint content. They are
inspection aids and hard architectural constraints respectively.

### 11.2 The event pathway

```python
ev = self.event_proj(self.h_norm(hidden_state))
ev = ev / ev.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
event_term = self.event_drive_scale * ev          # × 0.1
```

Three layers of bounding, each closing a hole found by measurement:

- `h_norm` — closes *input* magnitude growth (verified working: rms exactly 1.000
  for 30 steps).
- output RMS — closes *learned weight* growth ($\lVert W \rVert$ ×2.15 in 30
  steps).
- `event_drive_scale = 0.1` — makes the event term a small perturbation of the
  tanh sum rather than something that can saturate it alone.

### 11.3 The prediction-error pathway

```python
pe_scaled = pe / self.pe_reference_scale            # ÷ 10.0
drive = torch.tanh(event_term + self.pe_proj(pe_scaled))
```

The division is not cosmetic. A token-level NLL is $O(10)$ at init and
**always positive** — the NLL of an observed token is never negative. Fed
directly into `pe_proj`, that produced a term ~100× larger than the event
drive, **with a sign fixed forever**. The drive therefore had a near-constant
sign and the state was pinned at a rail.

Dividing by $\ln(V) \approx 10.4$ makes the input $O(1)$ and zero-centred, so it
can actually *oppose* the event drive. The reference scale is a property of the
NLL definition — fixed, not tuned.

### 11.4 The recurrence

```python
E_prev = torch.clamp(E_t, -1.0, 1.0)
E_next = self.decay * E_prev + (1.0 - self.decay) * drive
assert E_next.abs().max() <= 1.0 + 1e-6
```

Defensive clamp on the way in (float drift must never break the bound), and an
assert rather than a log — if a future change reintroduces saturation, the
training run *stops* instead of quietly producing 3000 steps of meaningless
data.

### 11.5 No recurrence in `drive_seq` — and why that is the key fact

```python
def drive_seq(self, hidden_seq, prediction_error_seq):
    """There is deliberately NO recurrence here."""
```

The drive is a function of base hidden states and the prediction error only.
**Neither can depend on $E$.** So all $T$ positions can be computed
independently — which is precisely what makes the closed form in
`update_seq()` possible.

This is the architectural fact that makes the whole design tractable: the
recurrence is only over $E$, and $E$'s inputs are loop-free. Had the drive been
a function of the modulated hidden state, the closed form would be impossible
and the vectorization undone.

---

## 12. The forward pass, end to end

### The data path

```
x, y                                   (B, T)      y[p] == x[p+1] exactly
  │
  ├─ base.forward(return_hidden=True) ──► hidden    (B, T, 768)
  │        └─ loss from UNMODULATED states (discarded below)
  │
  ├─ slices:  pre_outcome_hidden = hidden[:, :-1]     (B, T-1, 768)
  │           pooled_post_outcome = hidden[:, 1:]     (B, T-1, 768)
  │
  ├─ pe_seq = PE(logits(pre_outcome_hidden), targets[:, :-1])   ← THE FIX
  │                                                     (B, T-1)
  ├─ drive = state.drive_seq(pre_outcome_hidden, pe_seq)        (B, T-1, 8)
  ├─ E_next_seq = state.update_seq(E_start, drive)              (B, T-1, 8)
  ├─ E_in = cat([E_start, E_next_seq[:, :-1]])        (B, T-1, 8)
  ├─ E_hat_seq = predictor.predict_seq(E_in, pre_outcome_hidden)
  ├─ dpe = drive_prediction_error(E_next_seq, E_hat_seq)        ← LOGGED ONLY
  │
  ├─ film_state = cat([E_start.unsqueeze(1), E_next_seq])       (B, T, 8)
  ├─ [arm C: film_state = const_state_vec.expand_as(film_state)]
  │
  ├─ hidden_mod = state.modulate(hidden, film_state)            (B, T, 768)
  ├─ lm_logits = base.lm_head(hidden_mod) → softcap(15·tanh(·/15))
  └─ loss = cross_entropy(lm_logits, targets)                   ← NO SHIFT
```

### Three things worth noticing

**(1) The base loss is discarded.** `base.forward` returns a loss computed from
*unmodulated* hidden states. The LM head and cross-entropy are **re-applied** to
the modulated states. At step 0 FiLM is the exact identity, so `hidden_mod ==
hidden` bit-for-bit and B' starts identical to baseline — the two diverge only
as the film weights receive gradient.

**(2) `dpe` is never consumed.** It is computed, returned, logged, and that is
all. Nothing about memory, learning rate, or attention is modulated by it.

> **The state influences the model only through FiLM.**

This is the single most important architectural fact for interpreting results.
The predictor is currently pure instrumentation.

**(3) `film_state` is shifted by one, deliberately.**

```python
film_state = torch.cat([E_start.unsqueeze(1), E_next_seq], dim=1)
```

Position $p$ is modulated by the state *before* the update that consumed
position $p-1$. A single-token decode has no "last state" to draw on, so
modulating with $E_{\text{next\_seq}}[p]$ would not be reproducible
autoregressively. This shift is load-bearing for inference, not just for
causality.

### The single-token decode branch

```python
else:  # T == 1 with a KV cache
    h_new = hidden[:, 0, :]
    E_next = self.state.update(E_t, h_new, torch.zeros(B, ...))
```

The prediction error is the missing piece — the next token does not exist yet —
so it is passed as $0$, the neutral value. Without this branch the state sits
frozen for an entire generation (measured: the norm stayed at its prefill value
across every decode step).

### FiLM receives `film_state`, not `E_seq`

A measurement trap worth recording: the probe's `E_seq` is `E_next_seq`, shaped
$(B, T-1, 8)$. FiLM actually receives `film_state`, shaped $(B, T, 8)$ — **shifted
by one**. Any reconstruction that uses `E_seq` will be subtly wrong.

Related: `modulate` is a **method**, so `register_forward_hook` does not catch
it. Wrap the bound method on the instance and return the original output
unchanged.

---

## 13. Arm C: the control that tests the idea

Covered conceptually in Section 9. Here is what C is *not*, which matters for
reading its result:

| | B' | C |
|---|---|---|
| Parameters | +27,760 | +27,768 |
| Drive computed | yes | **yes** |
| Recurrence run | yes | **yes** |
| Predictor run | yes | **yes** |
| `dpe` computed | yes | **yes** |
| FiLM input | per-position $E$ | constant vector |
| Backward through drive path | yes | **no** (pruned) |

The three differences between C and B', which the final report must state:

1. **Information reaching FiLM.** B' feeds a per-position state; C feeds one
   learned constant. This is the intended variable.
2. **RDA modules untrained at init.** In C, `event_proj`, `pe_proj` and `h_norm`
   receive no gradient and are only decayed.
3. **No backward through the drive path.** Autograd prunes it, because nothing
   in the loss depends on $E$.

Points 2 and 3 are not independent of point 1 — they are consequences of it.
But they mean C is not *only* "B' with the information removed"; it also removes
the learning of those modules. If C underperforms B', that is consistent with
either the state information mattering *or* the drive modules needing to be
trained to be useful.

### The pace anomaly

C runs **2.5× slower** than B' per step (5.18 vs 2.06 s/step, reproduced in two
pairs) *despite doing less backward work*. Cause not identified; candidates are
wasted forward work building a dead graph, or host-side synchronisation.
Section 16.

Note this does **not** affect validity: BPB at a fixed step count is independent
of wall-clock pace.

---

## 14. Measurement history: what we learned

### The A/B that established the effect is real

`ab_v2`, 600 steps, three seeds, post-leak-fix. Metric is validation BPB at the
**final** step, not best checkpoint.

| seed | baseline | B' | B' − baseline |
|---|---|---|---|
| 42 | 1.300643 | 1.278289 | −0.022354 |
| 43 | 1.287503 | 1.274878 | −0.012625 |
| 44 | 1.294832 | 1.275739 | −0.019093 |

Mean |diff| = **0.018024**.

The noise floor is essential and was measured, not assumed: a baseline *repeat*
at seed 42 gave 1.300643 vs 1.300576 — a difference of **0.000067**. So the
0.018024 effect is **269× the repeat noise** and above the 0.0137 threshold
discussed in Section 16.

Per-seed spread of the paired diffs is 0.0097, comparable to the effect itself.
That is a real limitation — three seeds is not many.

### The saturation investigation

| question | answer | method |
|---|---|---|
| Is the state saturating? | yes, `threat` only | `arg_p50` crosses −2.0 at step 1131, never −3.0 |
| Is it a runaway? | **no** | bias overshot to −1.756 at 1750, now relaxing |
| Is it a dead channel? | **no** | mean tanh deriv min 0.0507, late slope **positive** |
| Which pathway owns it? | the PE bias | 80% of `arg` at step 2999 |
| Why did it switch? | weight crossed zero | step 569/570, +0.00078 → −0.00186 |

### What the FiLM analysis actually showed

A long, careful investigation produced several claims that were later
**corrected**. The corrections are more instructive than the claims, so both are
recorded.

The reliable finding: FiLM is linear in $E$, so dimension $d$'s contribution is
exactly $W[:,d] \cdot E[:,d]$. Measuring the **offset/varying** ratio across
positions — assumption-free, since the $\lVert W[:,d]\rVert_{rms}$ factor cancels
— ranked the dimensions:

| dimension | $\lvert E_{mean}\rvert / E_{std}$ at step 2999 |
|---|---|
| **threat** | **436.1** |
| novelty | 139.5 |
| uncertainty | 11.6 |
| arousal | 5.8 |
| learning_progress | 3.4 |
| homeostatic_dev | 1.2 |
| integrity | 0.36 |
| agency | 0.21 |

`threat` is **1st of 8** in offset dominance — the most DC-dominated, not the
least.

The correct varied-magnitude ranking $\lvert col \rvert \cdot E_{std}$ puts
`threat` **last** (0.0114, against `agency` 0.5449). Both are true: `threat`
carries a large constant offset *and* very little variation. That combination is
the signature of a railed channel.

### A methodological lesson: absmax is unsigned

Two retractions in this project came from reading an unsigned field as if it
were signed. `E_perdim`, `pe_perdim` and every `*_absmax` in the logging are
**magnitudes**. They cannot show which bound a value is pinned against.

> A drive collapsed to the lower bound and a drive collapsed to the upper bound
> have the **same** absmax.

Always log signed min/max. And never label an absmax table "mean".

### The other axis error

On a $(768, B, T)$ tensor, `.std()`/`.mean()` **without an explicit `dim`**
reduces over units × batch × positions, so

$$
\text{std}_{\text{flat}}^2 = \mathbb{E}_j[W_j^2]\,\text{var}_p(E) + \bar{E}^2\,\text{var}_j(W)
$$

The second term is spread along the *column* direction, which dominates when
$|\bar E| \approx 1$. This produced a confident "threat is 6th of 8" reading
that was simply wrong.

> Always pass `dim` explicitly, reduce over positions only, then RMS over units.

### TF32: a tolerance lesson

Reconstructing the FiLM output from its captured inputs gave residuals of
$3.9\times10^{-4}$ — and a bit-exact match when replayed under the forward
pass's *own* settings.

The forward runs fp32 matmuls with TF32 **enabled**
(`float32_matmul_precision='high'`, not `'highest'`). TF32 carries 10 mantissa
bits → $\sim 4.9\times10^{-4}$ relative resolution. On outputs of magnitude
0.82/1.33 that predicts $4.0\times10^{-4}$/$6.5\times10^{-4}$ absolute; observed
$2.81\times10^{-4}$/$3.91\times10^{-4}$ are **below** those. Confirmed on
synthetic data: relative gap between (high, TF32) and fp64 = $4.33\times10^{-4}$.

The diagnostic tell: float64 and `'highest'`-without-TF32 both give the
*identical* $3.9\times10^{-4}$. Two different arithmetics producing the same
number means "less accurate view of the same one", not "more accurate view".

> **A $10^{-4}$ absolute tolerance is finer than TF32 can represent.** Any gate
> comparing a TF32 forward against an fp64 or fp32 reconstruction must use a
> *relative* tolerance of about $10^{-3}\cdot\max\lvert out\rvert$, or it will
> reject correct code. This gate once rejected a bit-exact decomposition.

### Empty-norm trap

`E.norm()` on shape $(B, n_{drives})$ returns the **Frobenius** norm over all
$B \cdot n$ elements, so it scales as $\sqrt{B}$ above the per-vector ceiling
$\sqrt{n_{drives}}$. At $B=2, n=8$ the per-row ceiling is 2.828 while `.norm()`
legitimately reaches ~3.5. This was misread as a tanh bound violation. It was
not.

> Use `.norm(dim=-1)` for per-vector state norms. Plain `.norm()` is valid only
> for a single unbatched $(n_{drives},)$ tensor.

---

## 15. Retractions and corrections

Kept deliberately. In a project with this many measurement hazards, the
corrections carry more information than the original claims.

| # | Claim | Status | What was wrong |
|---|---|---|---|
| 1 | `B'-s42 BPB 1.226305 vs baseline 1.301217` | **retracted** | label leak (Section 8) |
| 2 | "shuffle `y` and re-score" as label-blindness | **retracted** | baseline also moves (+1.748); confounds cheat with difficulty |
| 3 | "PE off via `targets=None`" = 4.29 | **retracted** | trains FiLM in an unseen state distribution |
| 4 | `max abs(dE)` at $p<k$ as leak test | **wrong test** | $E[p]$ legitimately records token $p+1$; bound belongs on hidden/logits |
| 5 | threat is "6th of 8", signal-dominated | **wrong** | axis error — missing `dim` (Section 14) |
| 6 | threat's FiLM col norm "grows 13.4%→10.6%, rank 3rd→5th" | superseded | ranks disagree between scale and shift projections; report both |
| 7 | grad-free modules "neither updated nor decayed" | **wrong** | `optim.py:75` applies decay unconditionally |
| 8 | "bit-identical to init" for the three modules | **artifact** | probe excluded them at `wd=0`; not the training config |
| 9 | "memory pressure" explains C's 2.5× slowdown | **retracted** | VRAM is static and identical across arms |
| 10 | C at 60 steps "not what you'd see if states contribute" | **withdrawn** | it *is* that direction; size negligible at 60 steps |
| 11 | "gradient pruning means no drive work in C" | **wrong** | prunes backward only; forward still runs every step |
| 12 | C projects past a 90-min cap | **wrong** | came from a block-buffering log reporting stale steps |

### Pattern in the retractions

Three distinct error classes, and the defences against each:

**1. Verifying the index, mislabelling the content.** (#1, #4.) The index
arithmetic was checked; what *lives* at that index was assumed. Defence: derive
the requirement as an equation and check each dependency explicitly.

**2. Wrong measurement setup.** (#5, #8, #9, #12.) The number was read correctly
from a measurement that did not mean what I thought. Defence: verify that the
probe measures what its name says, and that its configuration matches the
system under test.

**3. Invalid counterfactual.** (#2, #3, #10.) The comparison arm itself was
confounded. Defence: ask what a *correct* implementation would also do under the
same manipulation.

---

## 16. Open questions

### Does the state information help? — unresolved

This is what arm C exists to answer, and it is still open. The protocol:

| verdict | criterion |
|---|---|
| C matches B' | $\lvert C - B' \rvert < 0.0137$ on **all three** seeds |
| states contribute | C worse than B' by $> 0.0137$ on **all three** seeds |
| otherwise | inconclusive |

**The 0.0137 threshold is a proposal, not a calibrated value.** It is
approximately $4\times$ the measured repeat noise (0.000067) and comparable to
the per-seed spread of the paired B'-baseline diffs (0.0097). It has not been
derived from a power calculation, and with three seeds a result near the
threshold should not be trusted in either direction.

### Why is arm C 2.5× slower per step?

Open. C does *less* backward work than B' and is 2.5× slower. Measured, not
guessed:

- VRAM: both arms reach ~7.7 GB within ~30 s then hold **flat**; peaks 7727 vs
  7744 MiB; no growth. So not allocation pressure.
- No sysmem spill (Shared Usage 14.5 MiB).
- Not thermal (63–66 °C, near-max clocks).
- Utilisation 93.9% vs 88.8%, clocks 2835 vs 2738 MHz, power **70.9 vs 105.1 W**.

High utilisation + high clocks + **low** power + halved throughput reads as
waiting, not computing. Candidates: wasted forward work building a dead graph;
host-side synchronisation. **Neither confirmed** — separating them needs a
profile that was not run.

Not blocking: BPB at a fixed step count does not depend on wall-clock pace.

### Does the drive survive longer than 600 steps?

Unknown, and the question the current design makes sharpest. The convex
combination guarantees $|E| \le 1$ — but "bounded" is not "healthy". `threat`
reached a *converged rail* inside $[-1,1]$: `arg_p50` plateaued near −2.11 with
the bias supplying 80% of the argument.

So the range argument guarantees no saturation, and the diagnostic shows
saturation-of-meaning anyway: a drive that sits at one value conveys nothing,
because FiLM's contribution is $\propto W[:,d]E[:,d]$ and a constant $E[:,d]$
contributes only a DC offset.

**A bounded state can still be informationally dead.** Distinguishing "bounded"
from "informative" is the open design question, and the $o/v$ ratio from
Section 14 is the measurement that exposes it: `threat` at 436 means almost
pure DC.

### Where the parameters go

`dpe` is computed and never consumed. The predictor (4 tensors) is trained by
nothing. An end-to-end ablation would tell us whether the predictor is dead
weight or an unused capability — but that is a separate experiment from arm C,
and conflating them would repeat the mistake this project keeps having to
retract.

### Environment and tooling debts

- **`Start-Process -PassThru` returns `$null` ExitCode** unless `$p.Handle` is
  cached first. Verified: `cmd /c exit 5` → `[]` uncached, `[5]` cached. Unfixed,
  *every* run looks like it had no exit code.
- **`Wait-Process -Timeout` misfires** on already-exited processes (raises when
  the PID is gone), so `cmd /c exit 0` was reported as a *timeout*. Use a
  `HasExited` polling loop.
- **Buffered stdout makes progress unmeasurable.** Python block-buffers a
  redirected stream in ~8 KB chunks; a log once showed "step 200" while the
  trainer was at outer step 123, and a 45 s poll saw zero new bytes. That
  produced a 1.8× pace error and a phantom stall. Set `PYTHONUNBUFFERED=1`.
- **`2>&1` + `$ErrorActionPreference='Stop'`** dies on benign stderr. Use
  `Start-Process` with separated redirection.
- **Triton unavailable on Windows**, so `torch.compile` is disabled
  (`base_train.py:332`) and runs are eager. Not a bug — but it means all timings
  are unoptimised.

---

## 17. Glossary

| term | meaning in this project |
|---|---|
| **BPB** | bits per byte; the validation metric. Lower is better |
| **Drive** | one of the 8 state dimensions |
| **$E$** | the state vector, $\mathbb{R}^8$, bounded in $[-1,1]$ |
| **$E_t$** | state entering step $t$; $E_{t+1}$ after update |
| **$\lambda$ (decay)** | 0.95; retention coefficient |
| **Event drive** | from the residual stream via `event_proj` |
| **PE** | prediction error; token-level NLL, $O(10)$, always positive |
| **FiLM** | per-position affine modulation, the *only* state→loss path |
| **Drive projection** | `film_scale` / `film_shift`, both zero-init |
| **Offset/varying ($o/v$)** | $\lvert E_{mean}\rvert / E_{std}$ over positions; DC vs signal |
| **Convex-combination update** | $E_{t+1} = \lambda E_t + (1-\lambda)\text{drive}$; makes $\lvert E\rvert \le 1$ structural |
| **Ratchet** | self-exciting loop where modulation amplifies its own input |
| **Decay-only persistence** | state is never hard-reset at a document boundary; $\lambda$ alone limits carry-over |
| **Saddle** | parameter whose gradient is exactly zero at step 2+ — makes an arm vacuous |
| **Grad-free** | receives no gradient; **not** the same as frozen (see #7) |
| **Arm C** | control: RDA present, state information replaced by a learned constant |
| **TF32** | 10-mantissa-bit matmul; bounds achievable reconstruction precision |
| **Stall** | high GPU utilisation with low throughput — synchronisation, not compute |

### File map

| file | role |
|---|---|
| `nanochat/rda.py` | state module, predictor, `GPTWithRDA` wrapper (~1200 lines) |
| `scripts/base_train.py` | training entry; optimizer groups, RDA enrolment, leak gate |
| `nanochat/optim.py` | `MuonAdamW`; Muon for matrices, fused AdamW for the rest |
| `nanochat/gpt.py` | base transformer |
| `nanochat/dataloader.py` | best-fit packing; `y[p] == x[p+1]` |
| `run_ab_v2.ps1` | the A/B that established the effect |
| `run_ab_v3.ps1` | A/B + control arm C, with the 90-minute timeout guard |
| `run_ab_v4_preview.ps1` | 200-step, three-arm preview |
| `diagnostics/test_no_label_leak.py` | the causal gate; runs in PRE-FLIGHT |
| `diagnostics/verify_vectorized_update.py` | closed form vs reference loop |
| `diagnostics/probe_arm_c.py` | gradient norms, call counts, drift |
| `diagnostics/probe_arm_c_decay.py` | skipped-vs-stepped mechanism |
| `diagnostics/diag_gpu_sample.ps1` | nvidia-smi sampler → CSV |
| `research/EXPERIMENT_LOG.md` | full chronological record |

---

## Where to go next

If you want to *understand* the architecture, read Sections 1–7 in order, then
skim Section 12 for the wiring. If you want to *work on the project*, read
Sections 9, 13 and 16 — those are the open question, the control, and the
things that will waste your time if you do not know them.

The single highest-value habit this project has taught: **when a number
surprises you, check the measurement before believing the number.** Roughly half
the retractions in Section 15 were not wrong mathematics — they were
instrumentation reporting faithfully on the wrong thing.

---

## Concept Notes

This guide explains the architecture; the concept notes below each isolate one
mechanism. Read the guide for the reasoning, the note for the specific.

### Architecture

- [[RDA and Persistent State]] — the system this guide describes
- [[Residual Stream]] — what FiLM modulates and the event pathway reads
- [[FiLM Modulation]] — the only state-to-loss route
- [[Event Signal]] — the hidden-state drive
- [[Prediction Error]] — the outcome drive, and the leak surface
- [[State Saturation]] — why the recurrence is a convex combination
- [[State Prediction]] — the predictor, and why its output is unused
- [[Prediction vs Actual State]] — `dpe`, which is not prediction error
- [[Online Learning]] — decay-only persistence

### Experiments

- [[BPB]] — the metric, and the noise floor that makes it interpretable
- [[Ablation Study]] — removing one component
- [[Control Arm]] — isolating one variable
- [[Label Leakage]] — the bug that invalidated a generation of results
- [[Anomaly Evaluation]] — untested

### Tooling

- [[Python for ML]] — tensors, autograd, and the Windows/PowerShell traps
- [[Optimizer State]] — param groups and the grad-free subtlety
- [[Weight Decay]] — which acts even on grad-free parameters

### Records

- [[CURRENT_STATE]] — verified architecture description
- [[DECISIONS]] — six decisions with evidence
- [[FAILED_EXPERIMENTS]] — what went wrong
- [[EXPERIMENT_LOG]] — chronological record
- [[RESEARCH_QUESTIONS]] — the open questions
- [[HYPOTHESES]] — claims separated from facts

---

## Learning Path

A concrete route through this vault, ordered so each step assumes only the ones
before it. Six stages. Skip or reorder stages 5-6 if you only want to
understand the architecture; stages 1-4 are the minimum for working on the code.

### Stage 1 — Python and tooling (2-3 hours)

Nothing else is readable without this.

1. [[Python for ML]] — **Tensors and shapes**, then **Classes and
   inheritance**, then **The training loop idiom**. Read the whole file; it is
   short.
2. Back to the file for **Views versus copies**, **Type hints and casts**, and
   **Assertions**. Those four cover the mistakes this codebase actually makes.
3. **Numerical precision — the TF32 trap** and **Buffering and logging**. Read
   the buffering section even if you never touch the drivers — it explains why
   a log once showed step 200 when the trainer was at step 123.
4. **PowerShell pitfalls on Windows**, last, and only if you will run anything.

**Checkpoint.** You can read `forward()` in `rda.py` and predict what
`hidden[:, :-1, :]` produces.

### Stage 2 — The base architecture (3-4 hours)

1. [[RDA_ARCHITECTURE_GUIDE]] §1, *The transformer as a residual stream*. Read
   "The cost: the stream has no bounded scale" twice — it is the setup for
   everything that failed later.
2. Guide §2, *Attention, and what mixing means*. Only the subsection "One
   crucial detail for a recurrent architecture" matters for RDA; the rest is
   background.
3. [[Residual Stream]] — the concept note version, shorter.
4. Guide §4, *Feature-wise Modulation*, if you already know FiLM from elsewhere.
   Otherwise read [[FiLM Modulation]] instead.

**Checkpoint.** You can explain why the event pathway reads the *unmodulated*
stream and why that ordering is load-bearing.

### Stage 3 — The RDA mechanism (4-5 hours, the core)

Read in this order; each note assumes the previous.

1. [[RDA and Persistent State]] — the whole system, all three pathways. This is
   the map.
2. [[Event Signal]] — the hidden-state drive. Pay attention to "Layer 2" and the
   freeze-base control; it is the best diagnostic story in the project.
3. [[Prediction Error]] — the outcome drive. Two failure modes, both from
   properties of NLL you can reason about directly.
4. Guide §3, *The training loop's hidden state*, for the recurrence's failure
   modes and the convex-combination fix.
5. [[State Saturation]] — the `threat` investigation, and the section "The open
   problem — bounded is not healthy".
6. [[FiLM Modulation]] — the RMS budget and the three ablations that pinned the
   self-excitation.
7. Guide §6, *Reparameterization as a stability tool*, to see all three bounding
   mechanisms together as one design philosophy.
8. [[State Prediction]] and [[Prediction vs Actual State]] — the unused
   predictor and the `dpe` signal. Read the second before the first if you tend
   to confuse them.

**Checkpoint.** You can predict the effect of a change to `decay` or to
`event_drive_scale`, and say what evidence would falsify your prediction.

### Stage 4 — Causality and the failure record (3-4 hours)

This is the part that makes you useful on this project, because the codebase is
now correct and the *reasoning* is where mistakes still get made.

1. Guide §8, *Causality and the label-leak surface*, worked through in full.
2. [[Label Leakage]] — the same content as a standalone note, with the two
   invalid counterfactuals.
3. Guide §7, *Closed-form recurrences* — the missing-`decay` bug is the best
   example of a reference implementation earning its keep.
4. [[FAILED_EXPERIMENTS]] — read F4 and F5 closely. Their shared signature,
   *a result that came back exactly unchanged when something should have
   changed*, is the most useful heuristic this project produced.
5. Guide §15, *Retractions and corrections* — the 12-row table, grouped into
   three error classes.

**Checkpoint.** You can look at a `.norm()` or `.std()` call without a `dim`
argument and know immediately that it is probably wrong.

### Stage 5 — Measurement (3 hours)

1. [[BPB]] — the metric, the noise floor, and why the 0.0137 threshold is a
   proposal rather than a calibrated value.
2. [[Ablation Study]] — C1-C6, the per-dimension run, and what an ablation
   cannot tell you.
3. [[Control Arm]] — arm C, its three verified preconditions, and the two
   behaviours that are consequences of the intended one.
4. Guide §9, *Experimental design*, if you want the general framing.

**Checkpoint.** Given a proposed change, you can say which experiment
distinguishes its effect from a capacity effect.

### Stage 6 — Results and open questions (2 hours)

1. Guide §14, *Measurement history* — the A/B, the saturation investigation, and
   four methodological lessons that each cost a retraction.
2. [[Control Arm]] §"Status: unresolved" and the 200-step preview.
3. Guide §16, *Open questions* — and read [[State Saturation]] §"The open
   problem" again, because it is the sharpest one.
4. Guide §17, *Glossary*, as a reference rather than a read.

**Checkpoint.** You can state the primary question, what would answer it, and
why the current evidence does not.

### Working path, once you are through

| Goal | Go to |
|---|---|
| understand a mechanism | the concept note, then Guide §1-7 |
| understand a result | [[EXPERIMENT_LOG]], then [[FAILED_EXPERIMENTS]] if retracted |
| check what is true right now | [[CURRENT_STATE]], then [[HANDOVER]] |
| change the architecture | [[DECISIONS]] first — six choices already settled, with evidence |
| see what is still open | [[RESEARCH_QUESTIONS]] and [[HYPOTHESES]] |

### The shortest useful path

If you have 90 minutes: [[RDA and Persistent State]] → [[Event Signal]] →
[[FiLM Modulation]] → [[Control Arm]] → [[Label Leakage]].

That is the architecture, both drive pathways, the only path to the loss, the
experiment that tests whether any of it matters, and the bug that invalidated
most of the project's earlier results. Everything else is detail.
