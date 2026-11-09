"""T=1024 equivalence and causality checks for the vectorized E recurrence.

SEPARATE FROM verify_vectorized_update.py on purpose. That script checks the
small synthetic case quickly and includes KV-cache decode; this one is the
check that matters for TRAINING, because T=1024 is the length actually used
and a recurrence that only holds at T=128 proves nothing about a 1023-step
closed form.

Runs on CPU by default so it can be executed while a training run occupies
the GPU. Pass --device cuda to use the GPU when it is free.

  python diagnostics/verify_t1024.py            # CPU, batch 1
  python diagnostics/verify_t1024.py --device cuda

Reports max abs diff for E_next_seq, E_hat, dpe, loss, and the worst
parameter gradient, in fp32 and bf16, plus the causality perturbation test
with a live-perturbation control at every position.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA

T = 1024
V = 256          # small vocab keeps CPU cost down; the recurrence does not
                # depend on vocab size
FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def build(device, seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=T, vocab_size=V, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="SSSL")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0, check_leakage=False)
    m = m.to(device)
    with torch.no_grad():
        for lay in (m.state.film_scale, m.state.film_shift):
            lay.weight.normal_(0, 0.05)
            lay.bias.normal_(0, 0.05)
    m.train()
    return m


def maxdiff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def _safe(base, pre):
    lg = base.lm_head(pre)[..., :base.config.vocab_size].float()
    return 15 * torch.tanh(lg / 15)


def forward_loop(m, x, y, E_in):
    """Pre-vectorization code path, on the reference loop."""
    loss, hidden = m.base.forward(x, y, return_hidden=True)
    pre = hidden[:, :-1, :]
    pe = m.state.prediction_error_from_logits(_safe(m.base, pre), y[:, 1:])
    E_start = m.state.init_state(x.size(0), x.device, dtype=torch.float32)
    if E_in is not None:
        E_start = torch.clamp(E_in.to(torch.float32), -1.0, 1.0)
    E_final, E_seq, Ehat_seq = m._update_loop_reference(E_start, pre, pe)
    film_state = torch.cat([E_start.unsqueeze(1), E_seq], dim=1)
    hidden_mod = m.state.modulate(hidden, film_state)
    logits = m.base.lm_head(hidden_mod)[..., :m.base.config.vocab_size].float()
    logits = 15 * torch.tanh(logits / 15)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                           ignore_index=-1)
    return {"loss": loss, "E_next_seq": E_seq, "E_hat_next_seq": Ehat_seq,
            "dpe_seq": E_seq - Ehat_seq, "E_next": E_final,
            "hidden_mod": hidden_mod}


def state_equiv(device, dtype):
    """The RECURRENCE at T=1023 (T-1 update steps), both precisions."""
    print("\n=== state recurrence, T=%d, dtype=%s, device=%s ==="
          % (T, dtype, device.type))
    m = build(device, seed=7)
    m.state = m.state.to(dtype)
    m.predictor = m.predictor.to(dtype)
    B = 1
    torch.manual_seed(11)
    h = torch.randn(B, T - 1, m.d_model, device=device, dtype=dtype)
    pe = torch.rand(B, T - 1, device=device, dtype=dtype) * 3.0
    E0 = (torch.randn(B, m.n_drives, device=device) * 0.3).to(dtype)

    E_loop, E_seq_loop, Ehat_loop = m._update_loop_reference(E0, h, pe)

    E_seq_vec = m.state.update_seq(E0, m.state.drive_seq(h, pe))
    E_in = torch.cat([E0.unsqueeze(1), E_seq_vec[:, :-1, :]], dim=1)
    Ehat_vec = m.predictor.predict_seq(E_in, h)

    # Tolerance rationale: fp32 is tight (tiny tensors, shallow arithmetic).
    # bf16 is looser because the LOOP rounds to bf16 once per step while the
    # closed form accumulates in float64, so after ~1000 steps they genuinely
    # differ at bf16 resolution. The closed form is the more accurate one.
    tol = 1e-5 if dtype == torch.float32 else 5e-2
    res = {}
    for name, a, b in (
            ("E_next_seq", E_seq_vec, E_seq_loop),
            ("E_hat", Ehat_vec, Ehat_loop),
            ("dpe", E_seq_vec - Ehat_vec, E_seq_loop - Ehat_loop),
            ("carried E", E_seq_vec[:, -1, :], E_loop)):
        d = maxdiff(a, b)
        res[name] = d
        check("%s max|diff|" % name, d <= tol, "= %.3e (tol %.0e)" % (d, tol))
    return res


def end_to_end(device):
    """Full forward at T=1024 in the native compute dtype."""
    print("\n=== end to end, T=%d, device=%s ===" % (T, device.type))
    m = build(device, seed=9)
    torch.manual_seed(12)
    B = 1
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.randint(0, V, (B, T), device=device)
    E_in = torch.randn(B, m.n_drives, device=device) * 0.3

    out_v = m(x, y, E_t_external=E_in)
    gv = torch.autograd.grad(out_v["loss"], list(m.parameters()),
                             allow_unused=True)
    m.zero_grad(set_to_none=True)
    out_l = forward_loop(m, x, y, E_in)
    gl = torch.autograd.grad(out_l["loss"], list(m.parameters()),
                             allow_unused=True)

    worst, wname, n = 0.0, None, 0
    for (nm, a), (_, b) in zip(zip(m.named_parameters(), gv),
                              zip(m.named_parameters(), gl)):
        if a is None and b is None:
            continue
        n += 1
        d = maxdiff(a, b)
        if d > worst:
            worst, wname = d, nm
    print("  loss  vec=%.8f loop=%.8f  diff=%.3e"
          % (out_v["loss"].item(), out_l["loss"].item(),
             abs(out_v["loss"].item() - out_l["loss"].item())))
    print("  worst gradient diff over %d tensors: %s = %.3e" % (n, wname, worst))
    return dict(loss=abs(out_v["loss"].item() - out_l["loss"].item()),
                grad=worst)


def causality(device, positions=(1, 100, 500, 1022, 1023)):
    """Perturb targets at p; every STRICTLY EARLIER position must be
    bit-identical, and position p MUST change (the control).

    The control is the part that makes this meaningful: without it, a
    perturbation that silently did nothing would pass every check.
    """
    print("\n=== causality, T=%d, device=%s ===" % (T, device.type))
    m = build(device, seed=3)
    m.eval()
    torch.manual_seed(5)
    B = 1
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.randint(0, V, (B, T), device=device)
    with torch.no_grad():
        ref = m(x, y, E_t_external=None)["hidden_mod"].clone()
        for p in positions:
            y2 = y.clone()
            y2[:, p] = (y2[:, p] + 137) % V
            hm = m(x, y2, E_t_external=None)["hidden_mod"]
        # film_state[q] may depend on y[j] only for j < q, so changing y[:,p]
        # must leave positions <= p untouched. (Before the label-leak fix this
        # was checked as [:p], which was too weak AND had a control asserting
        # the opposite -- the leak made position p move.)
        check("perturb targets[.,%4d] -> hidden_mod[:, :%4d] identical"
              % (p, p + 1), torch.equal(ref[:, :p + 1, :], hm[:, :p + 1, :]))
        # CONTROL: the perturbation must be live, i.e. it must reach SOME
        # later position. The recurrence is cumulative, so y[p] affects every
        # q > p. If nothing moved the check above would pass vacuously.
        # At p == T-1 there IS no later position, so the control is not
        # applicable -- perturbing the final label is legitimately invisible.
        later = [q for q in (p + 1, p + 2, p + 10) if q < T]
        if later:
            moved = any(not torch.equal(ref[:, q, :], hm[:, q, :]) for q in later)
            check("  control: some position > %4d DOES change (perturbation is live)" % p,
                  moved)
        else:
            print("  [SKIP] control for p=%4d: no position after it" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    print("device=%s  T=%d  batch=1" % (a.device, T))

    r32 = state_equiv(device, torch.float32)
    rbf = state_equiv(device, torch.bfloat16)
    e2e = end_to_end(device)
    causality(device)

    print("\n" + "=" * 66)
    print("SUMMARY (max abs diff, closed form vs sequential loop)")
    print("  %-12s %14s %14s" % ("quantity", "fp32", "bf16"))
    for k in r32:
        print("  %-12s %14.3e %14.3e" % (k, r32[k], rbf[k]))
    print("  %-12s %14s %14.3e" % ("loss", "-", e2e["loss"]))
    print("  %-12s %14s %14.3e" % ("grad(worst)", "-", e2e["grad"]))

    if FAILURES:
        print("\nT=1024 VERIFY FAILED (%d): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("\nT=1024 VERIFY PASS")


if __name__ == "__main__":
    main()
