"""REGRESSION TEST: state.e_init's gradient status must stay understood.

CPU-only, tiny tensors, no training, no checkpoints. Run:

    python diagnostics/test_einit_grad_status.py

WHY THIS EXISTS. The reason `e_init` sits in base_train.py's grad-free list was
stated as "the loss has no gradient path to it". That is FALSE: a path exists
(init_state() is differentiable), and a previous review asserted the opposite
error instead -- that e_init trains normally and receives gradient ~27. Both were
wrong. The measured truth is a combination of two facts:

  1. With E_t_external=None, the path e_init -> init_state() -> E_t ->
     film_state -> modulate() -> loss IS live, but yields EXACTLY 0.0 because
     film_scale/film_shift are zero-initialised and modulate() is the identity.
  2. With E_t_external supplied (what base_train does from step 1 onward),
     forward() DETACHES it, so e_init is unreachable and autograd PRUNES it:
     grad is None.

Net effect in a real run: e_init is structurally learnable but stays at its zero
init. This test pins BOTH halves, because a future change to the persistence
semantics (see docs/decay_semantics.md) or to the zero-init of FiLM would make
the parameter load-bearing, and nobody should discover that from a crash.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from nanochat.rda import GPTWithRDA  # noqa: E402

B, T, V = 2, 8, 32


class _Cfg:
    n_embd, vocab_size = 16, V


class _Stub(nn.Module):
    """Minimal stand-in with GPT's signature; avoids a real GPT + checkpoints."""

    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.emb = nn.Embedding(V, 16)
        self.lm_head = nn.Linear(16, V, bias=False)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean",
                return_hidden=False):
        hid = self.emb(idx)
        lg = 15 * torch.tanh(self.lm_head(hid)[..., :V].float() / 15)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, V), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        return (loss, hid) if return_hidden else lg

    def get_device(self):
        return next(self.parameters()).device

    def estimate_flops(self, *a, **k):
        return 0


def main():
    torch.manual_seed(0)
    x = torch.randint(1, V, (B, T))
    y = torch.randint(1, V, (B, T))
    failures = []

    def check(name, cond, extra=""):
        print("  %-58s %s%s" % (name, "PASS" if cond else "FAIL",
                                ("  " + extra) if extra else ""))
        if not cond:
            failures.append(name)

    print("=" * 78)
    print("e_init gradient status")
    print("=" * 78)

    m = GPTWithRDA(_Stub(), debug_steps=0)

    print("\n[1] no carry (step-0 path): grad is a real ZERO TENSOR, not None")
    m.zero_grad(set_to_none=True)
    m(x, y)["loss"].backward()
    g = m.state.e_init.grad
    check("e_init.grad is not None", g is not None)
    check("e_init.grad is exactly zero", g is not None and float(g.abs().max()) == 0.0,
          "|grad| = %s" % (None if g is None else "%.1e" % g.abs().max()))

    print("\n[2] carry supplied (base_train from step 1): grad is None (pruned)")
    m2 = GPTWithRDA(_Stub(), debug_steps=0)
    with torch.no_grad():   # FiLM no longer the identity, to prove the point
        m2.state.film_scale.weight.normal_(0, 0.5)
        m2.state.film_shift.weight.normal_(0, 0.5)
    m2.zero_grad(set_to_none=True)
    out = m2(x, y, E_t_external=m2.state.init_state(B, torch.device("cpu")))
    out["loss"].backward()
    check("film weights DO get gradient with a carry",
          m2.state.film_scale.weight.grad is not None,
          "|grad| = %.3e" % m2.state.film_scale.weight.grad.abs().max())
    check("e_init.grad is None with a carry", m2.state.e_init.grad is None)

    print("\n[3] init_state() is differentiable (so the path is real)")
    st = m2.state.init_state(B, torch.device("cpu"))
    check("init_state output requires_grad", st.requires_grad)
    check("init_state has a grad_fn", st.grad_fn is not None, str(type(st.grad_fn).__name__))

    print("\n[4] with a NON-ZERO e_init and NO carry, the path produces a real grad")
    m3 = GPTWithRDA(_Stub(), debug_steps=0)
    with torch.no_grad():
        m3.state.film_scale.weight.normal_(0, 0.5)
        m3.state.film_shift.weight.normal_(0, 0.5)
        m3.state.e_init.normal_(0, 0.3)     # a MEANINGFUL initial state
    m3.zero_grad(set_to_none=True)
    m3(x, y)["loss"].backward()
    g3 = m3.state.e_init.grad
    check("non-zero e_init + no carry -> nonzero grad", g3 is not None and g3.abs().max() > 0,
          "|grad| = %s" % ("None" if g3 is None else "%.4e" % g3.abs().max()))
    print("      -> so 'expands a zero vector' was NOT the mechanism. The zero")
    print("         VALUE is irrelevant to whether the graph exists.")

    print("\n[5] end-to-end: reproduce the trainer's call pattern over 5 steps")
    m4 = GPTWithRDA(_Stub(), debug_steps=0)
    opt = torch.optim.AdamW(m4.parameters(), lr=1e-2)
    carry, hist = None, []
    for step in range(5):
        if step == 1:
            with torch.no_grad():
                m4.state.film_scale.weight.normal_(0, 0.5)
                m4.state.film_shift.weight.normal_(0, 0.5)
        opt.zero_grad(set_to_none=True)
        out = m4(x, y, E_t_external=carry)
        out["loss"].backward()
        g = m4.state.e_init.grad
        hist.append("None" if g is None else "%.3e" % g.abs().max())
        carry = out["E_next"].detach()
        if g is None:                       # base_train's zero-priming
            m4.state.e_init.grad = torch.zeros_like(m4.state.e_init)
        opt.step()
    print("      per-step |grad| = %s" % hist)
    print("      e_init after 5 steps = %s" % m4.state.e_init.detach().tolist())
    check("step 0 grad is a zero tensor", hist[0] == "0.000e+00")
    check("steps 1+ grad is None", all(h == "None" for h in hist[1:]))
    check("e_init stayed at its zero init",
          float(m4.state.e_init.detach().abs().max()) == 0.0)

    print("\n" + "=" * 78)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("PASSED -- e_init: structurally learnable, frozen at zero init in practice.")
    print("If the persistence semantics change so a fresh carry can be MEANINGFUL,")
    print("this parameter becomes load-bearing and must leave the grad-free list.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())