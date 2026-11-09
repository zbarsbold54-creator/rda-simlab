"""PERMANENT REGRESSION TEST: RDA must never see the label it is scored on.

THE INVARIANT
-------------
The dataloader makes `y[p] == x[p+1]` exactly (verified, fraction 1.000000),
and `gpt.py` scores `cross_entropy(logits[p], targets[p])` with NO shift. So
`logits[p]` is the distribution for token `x[p+1]`, and it is graded against
`y[p] = x[p+1]`.

Therefore, for the model to be honest:

    film_state[p] may depend on y[j] only for j < p.

Since y[j] == x[j+1], that is equivalent to saying `film_state[p]` may depend
on `x[0..p]` and nothing beyond -- i.e. exactly what a causal transformer
position is allowed to see.

`film_state` is the ONLY route from the drive state to the logits, so this one
condition is the whole leak surface.

THE THREE CHECKS
----------------
1. y-perturbation. Fix x, change y[:,k] alone. `logits[:, :k+1]` must be
   bit-identical. This is the check that failed before the fix.
2. x-perturbation. Fix y, change x[:,k]. `logits[:, :k]` must be
   bit-identical -- the ordinary causality property, which passed even while
   the leak was present. Kept so a future "fix" that breaks plain causality
   is caught.
3. Label-blindness. Replace every label at positions >= p with random tokens.
   `logits[:, :p+1]` must be bit-identical to the real-label run. This is the
   vectorized form of check 1 and is the property the metric depends on.

WHY THE MODEL IS BUILT WITH NON-ZERO FiLM WEIGHTS
-------------------------------------------------
This is the whole test. `GPTWithRDA` zero-initializes `film_scale` /
`film_shift` so that step 0 is bit-identical to the baseline. A model left at
that initialization has `modulate()` == identity, the drive state reaches the
logits NOWHERE, and every check here passes VACUOUSLY -- including on the
leaking code. An earlier draft of this audit was fooled exactly that way.

So this test randomizes the FiLM weights and then ASSERTS they are non-zero
before running any check. A vacuous pass is a failure.

Run:  python diagnostics/test_no_label_leak.py
Exit: 0 = invariant holds, 1 = LEAK (or a vacuous test)
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

# Small but real: the leak is a property of the INDEXING, not of width, so a
# tiny model exercises the identical code path in a fraction of the time.
D_MODEL = 64
N_LAYER = 2
N_HEAD = 2
SEQ = 128
BATCH = 2
VOCAB = 512
SEED = 1234

# Token ids used for perturbation. Kept well inside the vocab and away from 0
# so a bug cannot hide behind a coincidence with a zero-init or pad id.
POKE = 7
POKE2 = 11

# Positions to probe. Includes 0 (the E_start path), 1 and 2 (the boundary the
# fix changes), the middle, and the last position.
PROBE = (0, 1, 2, 3, 17, 64, 126, 127)


def say(s=""):
    print(s, flush=True)


def build(device, decay=None):
    torch.manual_seed(SEED)
    cfg = GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=N_LAYER,
                    n_head=N_HEAD, n_kv_head=N_HEAD, n_embd=D_MODEL,
                    window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0,
                   decay=decay if decay is not None else 0.95).to(device)
    # Break the zero-init identity. Without this the test proves nothing.
    with torch.no_grad():
        m.state.film_scale.weight.normal_(0.0, 0.5)
        m.state.film_scale.bias.normal_(0.0, 0.5)
        m.state.film_shift.weight.normal_(0.0, 0.5)
        m.state.film_shift.bias.normal_(0.0, 0.5)
        m.state.pe_proj.bias.normal_(0.0, 0.5)
    return m.to(device).eval()


def make_io(device):
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    # Real token ids, never 0, so nothing can be confused with a special id.
    x = torch.randint(4, VOCAB, (BATCH, SEQ), generator=g).to(device)
    y = torch.roll(x, shifts=-1, dims=1).contiguous()   # y[p] == x[p+1]
    return x, y


def logits_of(m, x, y):
    with torch.no_grad():
        out = m(x, y)
    return out["hidden_mod"]


def head_logits(m, hidden_mod):
    with torch.no_grad():
        lg = m.base.lm_head(hidden_mod)
        lg = lg[..., :m.base.config.vocab_size].float()
        return 15 * torch.tanh(lg / 15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rda-decays", default=None,
                        help="optional comma-separated decay values, one per drive")
    args = parser.parse_args()
    decay = None
    if args.rda_decays is not None:
        try:
            decay = [float(value) for value in args.rda_decays.split(",")]
        except ValueError as exc:
            parser.error(f"invalid --rda-decays: {exc}")
        if len(decay) != 8 or any(not 0.0 <= value < 1.0 for value in decay):
            parser.error("--rda-decays must contain exactly 8 values in [0, 1)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    say("=" * 70)
    say("RDA LABEL-LEAK REGRESSION TEST")
    say("=" * 70)
    say("device=%s  d_model=%d layers=%d T=%d B=%d vocab=%d"
        % (device, D_MODEL, N_LAYER, SEQ, BATCH, VOCAB))

    m = build(device, decay=decay)
    x, y = make_io(device)

    # ---- the test's own preconditions ---- #
    fs = float(m.state.film_scale.weight.abs().sum().detach())
    fh = float(m.state.film_shift.weight.abs().sum().detach())
    say("  |film_scale.W|=%.4f  |film_shift.W|=%.4f" % (fs, fh))
    if fs == 0.0 or fh == 0.0:
        say("")
        say("FATAL: FiLM weights are zero, so modulate() is the identity and")
        say("every check below would pass VACUOUSLY. Refusing to report.")
        return 1
    yy = (y[:, :-1] == x[:, 1:]).float().mean().item()
    say("  y[p]==x[p+1] fraction: %.6f" % yy)
    if yy < 0.999:
        say("FATAL: precondition y[p]==x[p+1] does not hold; the invariant")
        say("below is stated in terms of it. Refusing to report.")
        return 1
    say("  preconditions OK (non-vacuous)")
    say("")

    failures = []

    # ---- CHECK 1: y-perturbation ---- #
    say("--- CHECK 1: change y[:,k] alone; logits[:, :k+1] must be identical ---")
    lg_ref = head_logits(m, logits_of(m, x, y))
    for k in PROBE:
        y2 = y.clone()
        y2[:, k] = POKE if int(y[0, k]) != POKE else POKE2
        lg = head_logits(m, logits_of(m, x, y2))
        same = torch.equal(lg[:, :k + 1], lg_ref[:, :k + 1])
        moved = float((lg[:, k + 1:] - lg_ref[:, k + 1:]).abs().max()) if k + 1 < SEQ else 0.0
        say("  k=%-4d logits[:, :%d] bit-identical: %-5s | pos>k moved: %.3e"
            % (k, k + 1, "YES" if same else "NO", moved))
        if not same:
            failures.append("y-perturbation at k=%d changed logits[:, :%d]" % (k, k + 1))
    say("")

    # ---- CHECK 2: x-perturbation ---- #
    say("--- CHECK 2: change x[:,k] alone; logits[:, :k] must be identical ---")
    for k in PROBE:
        if k == 0:
            continue                      # position 0 IS the changed position
        x2 = x.clone()
        x2[:, k] = POKE if int(x[0, k]) != POKE else POKE2
        lg = head_logits(m, logits_of(m, x2, y))
        same = torch.equal(lg[:, :k], lg_ref[:, :k])
        say("  k=%-4d logits[:, :%d] bit-identical: %s" % (k, k, "YES" if same else "NO"))
        if not same:
            failures.append("x-perturbation at k=%d changed logits[:, :%d]" % (k, k))
    say("")

    # ---- CHECK 3: label-blindness, vectorized ---- #
    say("--- CHECK 3: randomize all labels at positions >= p; ---")
    say("---           logits[:, :p+1] must be identical                 ---")
    g = torch.Generator(device="cpu").manual_seed(SEED + 2)
    for p in (1, 2, 17, 64, 127):
        yr = y.clone()
        rnd = torch.randint(4, VOCAB, (BATCH, SEQ - p), generator=g)
        yr[:, p:] = rnd.to(device)
        lg = head_logits(m, logits_of(m, x, yr))
        same = torch.equal(lg[:, :p + 1], lg_ref[:, :p + 1])
        say("  p=%-4d all labels >= %d randomized; logits[:, :%d] identical: %s"
            % (p, p, p + 1, "YES" if same else "NO"))
        if not same:
            failures.append("randomizing labels >= %d changed logits[:, :%d]" % (p, p + 1))
    say("")

    say("=" * 70)
    if failures:
        say("RESULT: FAIL -- %d violation(s)" % len(failures))
        for f in failures:
            say("  * %s" % f)
        say("")
        say("The drive state is reading the label it is scored on.")
        return 1
    say("RESULT: PASS -- invariant holds at every probed position.")
    say("  film_state[p] depends on y[j] only for j < p.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
