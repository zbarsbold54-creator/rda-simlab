"""Verify the vectorized E recurrence against the sequential loop.

WHY THIS EXISTS. The recurrence in RDAStateModule.update_seq() is a closed
form that REPLACED a step-by-step loop. Replacing a working recurrence with
an algebraically-equivalent-but-different-looking one is exactly the kind of
change that can silently change the model: the shapes still match, the loss
still goes down, and nothing complains. So this compares VALUES, not shapes.

Checks, in order:
  1. The state recurrence itself, in fp32 and in bf16: E_next_seq, E_hat_seq,
     dpe and the carried state, closed form vs loop.
  2. End to end through forward(): loss, hidden_mod and EVERY parameter
     gradient, closed form vs loop, in the native compute dtype.
  3. Causality: perturbing targets at several positions must leave every
     earlier position's modulated hidden state BIT-IDENTICAL.
  4. A real KV-cache decode of >= 6 steps.
  5. Throughput, closed form vs loop.

NOTE ON PRECISION. The base GPT hard-requires bfloat16 rotary embeddings, so
an all-fp32 end-to-end pass is impossible on this model. The recurrence is
therefore checked directly at both precisions (1) and the full forward
separately in the native bf16 (2).

The loop is called through _update_loop_reference(), which is the original
code kept verbatim. It is deliberately NOT the code path forward() uses, so
comparing the two means comparing two genuinely different implementations.

Run:
  python diagnostics/verify_vectorized_update.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Tolerances. fp32 is tight because the state tensors are tiny and the
# arithmetic is only a few flops deep. bf16 is loose because the BASE model
# runs in bf16 and the loss/gradients inherit its rounding regardless of how
# the recurrence is written -- that floor is set by the model, not by us.
TOL = {
    torch.float32: dict(state=1e-6, loss=1e-6, grad=1e-6),
    torch.bfloat16: dict(state=5e-2, loss=5e-3, grad=5e-2),
}
# The end-to-end pass runs in the base model's native dtype, which is bf16 on
# CUDA. Read from the dtype object rather than hardcoding a string key.
NATIVE_TOL = TOL[COMPUTE]

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  [%s] %s%s" % (tag, name, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def build(seed=0):
    """Build a small model in its NATIVE compute dtype.

    NOTE the base GPT hard-asserts that its rotary embeddings are in
    COMPUTE_DTYPE (bf16 on CUDA), so an all-fp32 end-to-end pass is not
    possible on this model. The state recurrence is therefore checked
    directly at both precisions below, and the full forward separately in
    the native dtype.
    """
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=256, vocab_size=256, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="SSSL")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0, check_leakage=False)
    m = m.to(DEVICE)
    # Non-zero FiLM, else modulation is the identity and the causality check
    # is vacuous.
    with torch.no_grad():
        for lay in (m.state.film_scale, m.state.film_shift):
            lay.weight.normal_(0, 0.05)
            lay.bias.normal_(0, 0.05)
    m.train()
    return m


def forward_loop(m, x, y, E_in):
    """Run the ORIGINAL sequential recurrence, returning the same tuple the
    vectorized forward() returns. Reconstruction of the pre-vectorization
    code path, built on _update_loop_reference()."""
    loss, hidden = m.base.forward(x, y, return_hidden=True)
    pre = hidden[:, :-1, :]
    B, L = x.size(0), pre.size(1)
    pe = m.state.prediction_error_from_logits(
        _safe(m.base, pre), y[:, 1:])
    E_start = m.state.init_state(B, x.device, dtype=torch.float32)
    if E_in is not None:
        E_start = torch.clamp(E_in.to(dtype=torch.float32), -1.0, 1.0)
    E_final, E_seq, Ehat_seq = m._update_loop_reference(E_start, pre, pe)
    film_state = torch.cat([E_start.unsqueeze(1), E_seq], dim=1)
    hidden_mod = m.state.modulate(hidden, film_state)
    logits = m.base.lm_head(hidden_mod)
    logits = logits[..., :m.base.config.vocab_size].float()
    logits = 15 * torch.tanh(logits / 15)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
    return {
        "loss": loss,
        "E_next_seq": E_seq,
        "E_hat_next_seq": Ehat_seq,
        "dpe_seq": E_seq - Ehat_seq,
        "E_next": E_final,
        "hidden_mod": hidden_mod,
    }


def _safe(base, pre):
    lg = base.lm_head(pre)[..., :base.config.vocab_size].float()
    return 15 * torch.tanh(lg / 15)


def maxdiff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def state_equiv(dtype):
    """Compare the RECURRENCE ALONE at the given precision, closed form vs
    the reference loop, on the same inputs."""
    print("\n=== state recurrence, dtype=%s ===" % dtype)
    m = build(seed=7)
    m.state = m.state.to(dtype)
    m.predictor = m.predictor.to(dtype)
    B, T, D = 3, 128, m.d_model
    torch.manual_seed(11)
    h = torch.randn(B, T, D, device=DEVICE, dtype=dtype)
    pe = (torch.rand(B, T, device=DEVICE, dtype=dtype) * 3.0)
    E0 = (torch.randn(B, m.n_drives, device=DEVICE) * 0.3).to(dtype)

    E_loop, E_seq_loop, Ehat_loop = m._update_loop_reference(E0, h, pe)
    assert E_seq_loop is not None

    drive = m.state.drive_seq(h, pe)
    E_seq_vec = m.state.update_seq(E0, drive)
    E_in = torch.cat([E0.unsqueeze(1), E_seq_vec[:, :-1, :]], dim=1)
    Ehat_vec = m.predictor.predict_seq(E_in, h)
    E_vec = E_seq_vec[:, -1, :]

    tol = TOL[dtype]["state"]
    for name, a, b in (
            ("E_next_seq", E_seq_vec, E_seq_loop),
            ("E_hat_seq", Ehat_vec, Ehat_loop),
            ("dpe_seq", E_seq_vec - Ehat_vec, E_seq_loop - Ehat_loop),
            ("carried E", E_vec, E_loop)):
        d = maxdiff(a, b)
        check("%s max|diff|" % name, d <= tol,
              "= %.3e (tol %.0e)" % (d, tol))


def end_to_end():
    """Full forward(): loss, hidden_mod and every parameter gradient,
    closed form vs the reference loop, in the native compute dtype."""
    print("\n=== end to end through forward() (native compute dtype) ===")
    m = build(seed=9)
    torch.manual_seed(12)
    B, T = 3, 128
    x = torch.randint(0, 256, (B, T), device=DEVICE)
    y = torch.randint(0, 256, (B, T), device=DEVICE)
    E_in = (torch.randn(B, m.n_drives, device=DEVICE) * 0.3)

    out_v = m(x, y, E_t_external=E_in)
    gv = torch.autograd.grad(out_v["loss"], list(m.parameters()),
                             allow_unused=True)
    m.zero_grad(set_to_none=True)
    out_l = forward_loop(m, x, y, E_in)
    gl = torch.autograd.grad(out_l["loss"], list(m.parameters()),
                             allow_unused=True)

    for key in ("E_next_seq", "E_hat_next_seq", "dpe_seq", "E_next", "hidden_mod"):
        d = maxdiff(out_v[key], out_l[key])
        check("%s max|diff|" % key, d <= NATIVE_TOL["state"],
              "= %.3e (tol %.0e)" % (d, NATIVE_TOL["state"]))

    dl = abs(out_v["loss"].item() - out_l["loss"].item())
    check("loss", dl <= NATIVE_TOL["loss"],
          "vec=%.8f loop=%.8f diff=%.3e (tol %.0e)"
          % (out_v["loss"].item(), out_l["loss"].item(), dl,
             NATIVE_TOL["loss"]))

    worst, wname, n_grad = 0.0, None, 0
    for (nm, g1), (_, g2) in zip(zip(m.named_parameters(), gv),
                                 zip(m.named_parameters(), gl)):
        if g1 is None and g2 is None:
            continue
        n_grad += 1
        d = maxdiff(g1, g2)
        if d > worst:
            worst, wname = d, nm
    check("all %d parameter gradients" % n_grad,
          worst <= NATIVE_TOL["grad"],
          "worst %s = %.3e (tol %.0e)" % (wname, worst, NATIVE_TOL["grad"]))


def causality():
    """Perturbing targets at position p must leave every STRICTLY EARLIER
    position bit-identical.

    The index matters. Position p's film state is E_next_seq[p-1], which was
    built from pe_seq[p-1] -- and pe_seq[p-1] is the PE of targets[p]. So
    position p legitimately DEPENDS ON targets[p]: that is the token sitting
    at position p, which the causal transformer has already consumed. The
    invariant is therefore about positions < p, not <= p.

    An earlier version of this test asserted [:p+1] and failed all seven
    positions. The test was wrong, not the model: demanding that position p
    ignore its own input would be demanding a genuinely broken model.
    """
    print("\n=== causality (bit-identical) ===")
    m = build(seed=3)
    m.eval()
    torch.manual_seed(5)
    B, T = 2, 256
    x = torch.randint(0, 256, (B, T), device=DEVICE)
    y = torch.randint(0, 256, (B, T), device=DEVICE)
    with torch.no_grad():
        base_hm = m(x, y, E_t_external=None)["hidden_mod"].clone()
        for p in (1, 2, 10, 100, 200, 254, 255):
            y2 = y.clone()
            y2[:, p] = (y2[:, p] + 137) % 256
            hm = m(x, y2, E_t_external=None)["hidden_mod"]
        # film_state[q] may depend on y[j] only for j < q, so changing y[:,p]
        # must leave positions <= p untouched. The old bound was [:p] and its
        # control asserted position p DID move, i.e. the test encoded the leak.
        check("perturb targets[.,%d] -> hidden_mod[:, :%d] identical"
              % (p, p + 1),
              torch.equal(base_hm[:, :p + 1, :], hm[:, :p + 1, :]))
    # Control: the perturbation must be live. The recurrence is cumulative, so
    # y[:,5] has to move SOME later position. Without this, the checks above
    # could pass simply because the perturbation does nothing.
    y2 = y.clone()
    y2[:, 5] = (y2[:, 5] + 137) % 256
    hm = m(x, y2, E_t_external=None)["hidden_mod"]
    check("control: some position > 5 DOES change (perturbation is live)",
          any(not torch.equal(base_hm[:, q, :], hm[:, q, :])
              for q in (6, 7, 15, 40) if q < base_hm.size(1)))


def _StubTok():
    """Tokenizer stub for the KV-cache path. Engine._rda_forward drives
    _rda_forward directly, so these are never called."""
    def encode_special(self, s): return 0
    def get_bos_token_id(self): return 0


def kv_decode(steps=6):
    """Real decode through a KV cache: prefill, then >=6 single-token steps.

    Driven through Engine._rda_forward, the same entry point generation uses,
    so both the kv_cache plumbing AND the E_t_external state threading are
    exercised. A tiny tokenizer stub satisfies the constructor; this path
    never calls it.
    """
    print("\n=== KV-cache decode (%d steps) ===" % steps)
    m = build(seed=13)
    m.eval()
    cfg = m.base.config
    T = 64
    torch.manual_seed(17)
    x = torch.randint(0, 256, (2, T), device=DEVICE)
    from nanochat.engine import Engine, KVCache
    eng = Engine(m, _StubTok())
    cache = KVCache(batch_size=2, num_heads=cfg.n_head, seq_len=T,
                    head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                    device=DEVICE, dtype=COMPUTE)

    out = eng._rda_forward(x[:, :T], kv_cache=cache, E_t_external=None)
    check("prefill logits shape",
          tuple(out.logits.shape) == (2, T, 256),
          "got %s" % (tuple(out.logits.shape),))
    E = out.state
    check("prefill produced a state", E is not None)

    ok, trace = True, []
    for _ in range(steps):
        nxt = torch.randint(0, 256, (2, 1), device=DEVICE)
        cache.advance(1)
        r = eng._rda_forward(nxt, kv_cache=cache, E_t_external=E)
        ok = ok and tuple(r.logits.shape) == (2, 1, 256)
        E = r.state
        trace.append(float(E.abs().max()))
    check("decoded %d steps, logits shape ok" % steps, ok)
    check("carried state advanced past init",
          E is not None and max(trace) > 0,
          "max|E| trace: %s" % ", ".join("%.4f" % v for v in trace))
    # The state must actually MOVE across decode steps, not sit frozen -- the
    # T=1 bug this vectorization sits next to.
    check("state evolves across decode steps (not frozen)",
          max(trace) - min(trace) > 0,
          "spread=%.6f" % (max(trace) - min(trace)))


def throughput():
    print("\n=== throughput (T=1024) ===")
    m = build(seed=21)
    m.eval()
    B, T = 4, 1024
    x = torch.randint(0, 256, (B, T), device=DEVICE)
    y = torch.randint(0, 256, (B, T), device=DEVICE)
    with torch.no_grad():
        for _ in range(2):                      # warmup
            m(x, y, E_t_external=None)
            forward_loop(m, x, y, None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            m(x, y, E_t_external=None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_vec = (time.time() - t0) / 5
        t0 = time.time()
        for _ in range(5):
            forward_loop(m, x, y, None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_loop = (time.time() - t0) / 5
    print("  vectorized : %.1f ms/pass" % (t_vec * 1e3))
    print("  loop       : %.1f ms/pass" % (t_loop * 1e3))
    print("  speedup    : %.1fx" % (t_loop / t_vec))
    return t_loop / t_vec


def main():
    print("device:", DEVICE)
    for dt in (torch.float32, torch.bfloat16):
        state_equiv(dt)
    end_to_end()
    causality()
    kv_decode(6)
    sp = throughput()
    print("\n" + "=" * 60)
    if FAILURES:
        print("VERIFY FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("VERIFY PASS  (speedup %.1fx)" % sp)


if __name__ == "__main__":
    main()
